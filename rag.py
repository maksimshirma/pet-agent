# -*- coding: utf-8 -*-
"""ДЗ2, Часть 1: RAG поверх своего Habr-корпуса + память агента.

Функции перенесены из seminar02_skeleton_full.ipynb почти без изменений — эталон
семинара уже решён. Источник для каждого блока указан в комментарии.
"""
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import requests
from pydantic import BaseModel, Field
from pymilvus import MilvusClient

import agent as A

DATA = Path(__file__).resolve().parent / "data"
EMBED_URL = "https://openrouter.ai/api/v1/embeddings"
EMBED_MODEL = "openai/text-embedding-3-small"
DIM = 512
CACHE_FILE = DATA / "embeddings_habr.npz"

MILVUS_URI = "http://localhost:19530"
_client = None


def get_client():
    """Ленивое подключение: MilvusClient коннектится сразу в конструкторе и падает,
    если Milvus не поднят — модуль должен импортироваться и без него (для тестов)."""
    global _client
    if _client is None:
        _client = MilvusClient(uri=MILVUS_URI)
    return _client


# --------------------------------------------------------------------------
# Чанкеры (ноутбук, ячейки 5 и 7 — chunk_lines/chunk_chars, здесь под Habr)
# --------------------------------------------------------------------------

def chunk_chars(page, size=300):
    """Наивная нарезка: окна по size символов, без учёта границ предложений/пунктов."""
    text = " ".join(page["text"].split())
    return [{"page": page["page"], "section": "", "url": page["url"], "text": text[i:i + size]}
            for i in range(0, len(text), size) if text[i:i + size].strip()]


def chunk_sections(page):
    """Осмысленная нарезка: один пункт дайджеста/абзац = один кусок, с разделом-заголовком
    как метаданными. Habr-дайджест — это список независимых новостей под заголовками
    разделов (CSS/JS/Браузеры/…); абзац = одна новость = одна проверяемая идея, поэтому
    граница куска проходит по абзацу/пункту, а не по фиксированной длине."""
    section = ""
    chunks = []
    for line in page["text"].splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^==\s*(.+?)\s*==$", line)
        if m:
            section = m.group(1)
            continue
        if len(line) >= 15:
            chunks.append({"page": page["page"], "section": section, "url": page["url"], "text": line})
    return chunks


def load_corpus(path=DATA / "corpus_habr.jsonl"):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


# --------------------------------------------------------------------------
# Эмбеддинги с диск-кэшем (ноутбук, ячейки 11-12)
# --------------------------------------------------------------------------

EMB = {}


def key_of(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def load_cache():
    if CACHE_FILE.exists():
        z = np.load(CACHE_FILE)
        for k, v in zip(z["keys"], z["vectors"]):
            EMB[k] = v.astype(np.float32)


def save_cache():
    if not EMB:
        return
    keys = list(EMB.keys())
    vectors = np.stack([EMB[k] for k in keys]).astype(np.float16)
    np.savez_compressed(CACHE_FILE, keys=np.array(keys), vectors=vectors)


def embed_batch(texts):
    started = time.perf_counter()
    r = requests.post(EMBED_URL, headers=A._headers(), timeout=60,
                       json={"model": EMBED_MODEL, "input": texts, "dimensions": DIM})
    if r.status_code != 200:
        raise RuntimeError(f"эмбеддинги: HTTP {r.status_code} {r.text[:500]}")
    data = r.json()
    A.ledger.add("embed", EMBED_MODEL, data.get("usage") or {}, time.perf_counter() - started)
    return [np.array(d["embedding"], dtype=np.float32) for d in data["data"]]


def embed_cached(texts):
    missing = list(dict.fromkeys(t for t in texts if key_of(t) not in EMB))
    for i in range(0, len(missing), 64):
        batch = missing[i:i + 64]
        for text, vec in zip(batch, embed_batch(batch)):
            EMB[key_of(text)] = vec
    vectors = np.stack([EMB[key_of(t)] for t in texts])
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


# --------------------------------------------------------------------------
# Поиск в numpy для recall@k (ноутбук, ячейки 18-20)
# --------------------------------------------------------------------------

def search_numpy(query_vec, vectors, k=5):
    sims = vectors @ query_vec
    top = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in top]


NUMBERS = {"ноль": "0", "один": "1", "одна": "1", "два": "2", "две": "2", "три": "3", "четыре": "4",
           "пять": "5", "шесть": "6", "семь": "7", "восемь": "8", "девять": "9", "десять": "10"}


def norm(text):
    text = str(text).lower().replace(",", ".")
    text = re.sub(r"[^\w\s.]", " ", text)
    words = [NUMBERS.get(w, w) for w in text.split()]
    return " ".join(words)


def is_gold(chunk, task):
    if not task.get("answerable", True) or not task.get("evidence"):
        return False
    if chunk["page"] != task["page"]:
        return False
    ctext = norm(chunk["text"])
    if norm(task["answer"]) not in ctext:
        return False
    ev_words = [w for w in norm(task["evidence"]).split() if len(w) > 2]
    if not ev_words:
        return False
    hit = sum(1 for w in ev_words if w in ctext)
    return hit / len(ev_words) >= 0.5


def recall_curve(chunks, vectors, tasks, q_vecs, ks=(1, 3, 5, 10)):
    rows = []
    for k in ks:
        hits = 0
        for task, qv in zip(tasks, q_vecs):
            if not task.get("answerable", True):
                continue
            top = search_numpy(qv, vectors, k)
            if any(is_gold(chunks[i], task) for i, _ in top):
                hits += 1
        answerable = sum(1 for t in tasks if t.get("answerable", True))
        rows.append({"k": k, "recall": round(hits / answerable, 3)})
    return rows


# --------------------------------------------------------------------------
# Milvus: индекс и поиск (ноутбук, ячейки 22-24)
# --------------------------------------------------------------------------

def index_to_milvus(name, chunks, vectors):
    client = get_client()
    if client.has_collection(name):
        client.drop_collection(name)
    client.create_collection(name, dimension=vectors.shape[1], metric_type="COSINE")
    rows = [{"id": i, "vector": v.tolist(), **c} for i, (c, v) in enumerate(zip(chunks, vectors))]
    for i in range(0, len(rows), 1000):
        client.insert(name, rows[i:i + 1000])
    client.flush(name)
    return client.get_collection_stats(name)


def search_milvus(name, query, k=5, flt=""):
    vector = embed_cached([query])[0]
    hits = get_client().search(name, data=[vector.tolist()], limit=k, filter=flt,
                                output_fields=["page", "section", "url", "text"])[0]
    return [{**h["entity"], "id": h["id"], "score": round(h["distance"], 4)} for h in hits]


# --------------------------------------------------------------------------
# Гибрид: ключевые слова + RRF (ноутбук, ячейки 41-42)
# --------------------------------------------------------------------------

def tokens(text):
    return re.findall(r"\w+", str(text).lower())


def build_keyword_index(texts):
    docs = [set(tokens(t)) for t in texts]
    n = len(docs)
    df = {}
    for d in docs:
        for w in d:
            df[w] = df.get(w, 0) + 1
    idf = {w: np.log(n / c) for w, c in df.items()}
    return docs, idf


def keyword_search(query, docs, idf, k=20):
    q = set(tokens(query))
    scores = [sum(idf.get(w, 0) for w in q & d) for d in docs]
    top = np.argsort(scores)[::-1][:k]
    return [(int(i), float(scores[i])) for i in top if scores[i] > 0]


def rrf(rankings, k=5, K=60):
    scores = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (K + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)[:k]


# --------------------------------------------------------------------------
# RAG-ответ (ноутбук, ячейка 33)
# --------------------------------------------------------------------------

RAG_SYSTEM = ("Отвечай на вопрос, используя только пронумерованные источники. Ссылайся на "
              "источник в квадратных скобках, например [2]. Если в источниках нет ответа, "
              "ответь ровно NOT_FOUND. Последняя строка обязательно FINAL: <короткий ответ>.")


def rag_answer(question, model, k=5, name="habr_lines"):
    before = A.ledger.total
    hits = search_milvus(name, question, k)
    sources = "\n".join(f"[{i + 1}] ({h['page']}) {h['text']}" for i, h in enumerate(hits))
    msg = A.chat([{"role": "system", "content": RAG_SYSTEM},
                  {"role": "user", "content": f"Источники\n{sources}\n\nВопрос: {question}"}],
                 model, tag="rag")
    return {"answer": msg.get("content") or "", "hits": hits, "cost": A.ledger.total - before}


def rag_run(question, model, k=5, name="habr_lines"):
    """Обёртка rag_answer в форме Run — совместима с measure.py/agent.py (ДЗ1)."""
    before, started = A.ledger.total, time.perf_counter()
    r = rag_answer(question, model, k, name)
    return A.Run(question, r["answer"], 1, [], A.ledger.total - before, time.perf_counter() - started)


# --------------------------------------------------------------------------
# Инструмент агента knowledge_base (ноутбук, ячейка 56)
# --------------------------------------------------------------------------

class KBArgs(BaseModel):
    query: str = Field(description="короткий поисковый запрос по базе знаний (по-русски)")
    page: str = Field(default="", description="точное название дайджеста/статьи для фильтра, например 'Frontend Status #8'; пусто — искать по всем страницам")


def knowledge_base(query: str, page: str = "") -> str:
    try:
        hits = search_milvus("habr_lines", query, 5, f'page == "{page}"' if page else "")
    except Exception as e:
        return f"ошибка базы знаний: {e}"
    if not hits:
        return "ничего не найдено"
    return "\n".join(f"[{h['page']} / {h['section']}] {h['text']}" for h in hits)


A.register(knowledge_base, KBArgs, "Ищет в локальной базе знаний (мои статьи Habr/lawebbox про фронтенд и AI в 2026) пять ближайших фактов с указанием дайджеста и раздела")

AGENT_SYSTEM = ("Ты отвечаешь на вопросы по локальной базе знаний (статьи Habr/lawebbox про "
                 "фронтенд и AI в 2026 году). Всегда сначала вызови knowledge_base, при "
                 "необходимости — с уточнённым запросом или фильтром по странице. Если после "
                 "поиска в базе знаний нет ответа на вопрос, ответь ровно NOT_FOUND. Последняя "
                 "строка ответа всегда FINAL: <короткий ответ или NOT_FOUND>.")


def kb_agent_run(question, model, max_steps=4):
    return A.agent(question, model, ["knowledge_base"], max_steps=max_steps, system=AGENT_SYSTEM)


# --------------------------------------------------------------------------
# Память: эпизодическая (журнал) и семантическая (факты в Milvus)
# (ноутбук, ячейки 47-50)
# --------------------------------------------------------------------------

JOURNAL = Path("episodes.jsonl")
FACTS_FILE = Path("facts.json")

FACTS_SYSTEM = ("Извлеки факты о пользователе из диалога: имя, город, работа, интересы и т.п. "
                 "Верни один JSON-объект вида {\"facts\": [...]} — полный обновлённый список "
                 "фактов, включая уже известные. Если новый факт отменяет старый (например, "
                 "пользователь переехал или сменил интерес), замени старый факт новым, не "
                 "добавляй оба.")


class Facts(BaseModel):
    facts: list[str]


def remember_episode(session, role, text):
    with JOURNAL.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"session": session, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                             "role": role, "text": text}, ensure_ascii=False) + "\n")


def known_facts():
    if FACTS_FILE.exists():
        return json.loads(FACTS_FILE.read_text(encoding="utf-8"))
    return []


def store_facts(facts):
    FACTS_FILE.write_text(json.dumps(facts, ensure_ascii=False, indent=1), encoding="utf-8")
    client = get_client()
    if facts:
        rows = [{"page": "memory", "section": "", "url": "", "text": f} for f in facts]
        index_to_milvus("facts", rows, embed_cached(facts))
    elif client.has_collection("facts"):
        client.drop_collection("facts")
    return facts


def recall_facts(question, k=3):
    if not get_client().has_collection("facts"):
        return []
    return [h["text"] for h in search_milvus("facts", question, k)]


def extract_facts(dialog, known):
    prompt = f"Известные факты: {json.dumps(known, ensure_ascii=False)}\n\nДиалог:\n{dialog}"
    msg = A.chat([{"role": "system", "content": FACTS_SYSTEM}, {"role": "user", "content": prompt}],
                 A.MODELS["cheap"], tag="memory")
    return Facts.model_validate(A.json_from(msg["content"])).facts


def end_session(history):
    dialog = "\n".join(f"{m['role']}: {m['text'][:400]}" for m in history)
    facts = extract_facts(dialog, known_facts())
    return store_facts(facts)


# --------------------------------------------------------------------------
# Диалог с двумя режимами контекста: вся история vs окно + факты
# (ноутбук, ячейка 50, "talk"/"search_with_memory")
# --------------------------------------------------------------------------

ASSISTANT_SYSTEM = ("Ты помогаешь пользователю по базе знаний (статьи Habr/lawebbox про "
                     "фронтенд и AI 2026 года). Если есть источники — отвечай по ним и "
                     "ссылайся на номер источника в квадратных скобках. Отвечай кратко.")


def search_with_memory(user_text, facts, k=5, name="habr_lines"):
    if "?" not in user_text:
        return []
    plain_hits = search_milvus(name, user_text, 20)
    if not facts:
        return plain_hits[:k]
    aug_hits = search_milvus(name, user_text + " " + " ".join(facts), 20)
    by_id = {h["id"]: h for h in plain_hits + aug_hits}
    fused = rrf([[h["id"] for h in plain_hits], [h["id"] for h in aug_hits]], k=k)
    return [by_id[i] for i in fused]


def talk(session, history, user_text, mode="память"):
    remember_episode(session, "user", user_text)
    facts = recall_facts(user_text, 3) if mode == "память" else []
    hits = search_with_memory(user_text, facts) if mode == "память" else \
        (search_milvus("habr_lines", user_text, 5) if "?" in user_text else [])
    sources = ("\n\nИсточники:\n" + "\n".join(f"[{i + 1}] ({h['page']}) {h['text']}" for i, h in enumerate(hits))) if hits else ""

    if mode == "вся история":
        convo = [{"role": m["role"], "content": m["text"]} for m in history]
        system = ASSISTANT_SYSTEM
    else:
        window = history[-2:]
        convo = [{"role": m["role"], "content": m["text"]} for m in window]
        system = ASSISTANT_SYSTEM + (f"\nИзвестно о пользователе: {'; '.join(facts)}" if facts else "")

    messages = [{"role": "system", "content": system}] + convo + [{"role": "user", "content": user_text + sources}]
    msg = A.chat(messages, A.MODELS["cheap"], tag=f"dialog: {mode}")
    reply = msg.get("content") or ""

    remember_episode(session, "assistant", reply)
    history.append({"role": "user", "text": user_text})
    history.append({"role": "assistant", "text": reply})
    return reply
