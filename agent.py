# -*- coding: utf-8 -*-
"""ReAct-агент для датасета свежих вопросов (Домашка 1, Шаг 2).

Ядро (Ledger, chat, agent_loop, register) — из ноутбука семинара, без изменений
по сути. Новое здесь — инструменты со свежим доступом к интернету:

  - web_search(query)      — вступление статьи в англоязычной Википедии (с семинара);
  - web_search_ru(query)   — то же самое в ru.wikipedia.org, для русскоязычных вопросов;
  - tavily_search(query)   — второй источник поиска, Tavily: не только Википедия,
                              подходит для вопросов с фактами из новостей/блогов;
  - page_find(title, kw)   — полный текст статьи Википедии (prop=extracts без exintro),
                              возвращает только строки с ключевыми словами;
  - calculator, python_exec — с семинара, без изменений.

Запуск демонстрации на своём датасете:

    python agent.py --data data/my_fresh_2026.jsonl --n 8 --model cheap
"""
import os
import re
import sys
import json
import time
import ast
import operator
import subprocess
import argparse
from pathlib import Path
from dataclasses import dataclass, field

import requests
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
MODELS = {
    "cheap": "openai/gpt-4o-mini",
    "mid": "anthropic/claude-haiku-4.5",
    "strong": "anthropic/claude-sonnet-4.6",
}

TRACES = Path("traces")


def _headers():
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("нет ключа: положите OPENROUTER_API_KEY в .env")
    return {"Authorization": f"Bearer {key}", "X-Title": "agents-course-hw1-agent"}


# --------------------------------------------------------------------------
# Леджер и вызов модели (с семинара)
# --------------------------------------------------------------------------

@dataclass
class Ledger:
    calls: list = field(default_factory=list)

    def add(self, tag, model, usage, seconds=0.0):
        p, c = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        cost = usage.get("cost") or 0.0
        self.calls.append({"tag": tag, "model": model.split("/")[-1], "prompt": p, "completion": c,
                            "cost": cost, "seconds": round(seconds, 2)})
        return cost

    @property
    def total(self):
        return sum(c["cost"] for c in self.calls)


ledger = Ledger()


def post_with_retry(body, attempts=3):
    """HTTP-запрос к OpenRouter с повтором на 429/5xx."""
    for attempt in range(attempts):
        r = requests.post(CHAT_URL, json=body, headers=_headers(), timeout=120)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503) and attempt < attempts - 1:
            time.sleep(1.5 * (attempt + 1))
            continue
        raise RuntimeError(f"HTTP {r.status_code} : {r.text[:2000]}")


def chat(messages, model, tools=None, tag="chat", temperature=None):
    body = {"model": model, "messages": messages, "usage": {"include": True}}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    if temperature is not None:
        body["temperature"] = temperature
    started = time.perf_counter()
    data = post_with_retry(body)
    ledger.add(tag, model, data.get("usage") or {}, time.perf_counter() - started)
    return data["choices"][0]["message"]


# --------------------------------------------------------------------------
# Инструменты: реестр
# --------------------------------------------------------------------------

TOOLS = {}


def register(fn, args_model, description):
    TOOLS[fn.__name__] = {"fn": fn, "args": args_model, "schema": {"type": "function", "function": {
        "name": fn.__name__, "description": description, "parameters": args_model.model_json_schema()}}}


# --- calculator (с семинара) ------------------------------------------------

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
       ast.Pow: operator.pow, ast.USub: operator.neg, ast.Mod: operator.mod}


def evaluate(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](evaluate(node.left), evaluate(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return OPS[type(node.op)](evaluate(node.operand))
    raise ValueError("допустимы только числа и арифметика")


class CalcArgs(BaseModel):
    expr: str = Field(description="арифметическое выражение, например 17*23+5")


def calculator(expr: str) -> str:
    try:
        val = evaluate(ast.parse(expr.replace(",", "."), mode="eval").body)
        return str(int(val)) if float(val).is_integer() else f"{val:.4f}".rstrip("0")
    except Exception as e:
        return f"ошибка вычисления: {e}"


# --- python_exec (с семинара) -----------------------------------------------

class ExecArgs(BaseModel):
    code: str = Field(description="код на Python; результат надо напечатать через print")


def python_exec(code: str) -> str:
    try:
        r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=5)
    except subprocess.TimeoutExpired:
        return "Код превысил 5 сек"
    out = (r.stdout + r.stderr).strip()
    return out[:5000] if out else "Код исполнился но результат пустой"


# --- Википедия: поиск и чтение страницы -------------------------------------

WIKI_HEADERS = {"User-Agent": "agents-course-hw1/1.0 (https://postypashki.ru; educational project)"}


def wiki(params, domain="en.wikipedia.org", attempts=3):
    url = f"https://{domain}/w/api.php"
    r = None
    for attempt in range(attempts):
        r = requests.get(url, params={**params, "format": "json"}, headers=WIKI_HEADERS, timeout=15)
        if r.status_code == 200 and r.headers.get("content-type", "").startswith("application/json"):
            return r.json()["query"]
        time.sleep(1.0 + attempt)
    raise RuntimeError(f"Википедия ({domain}) ответила {r.status_code if r is not None else '?'}")


class SearchArgs(BaseModel):
    query: str = Field(description="короткий поисковый запрос: имя, название, термин")


def _wiki_search(query: str, domain: str) -> str:
    try:
        hits = wiki({"action": "query", "list": "search", "srsearch": query, "srlimit": 1}, domain=domain)["search"]
    except Exception as e:
        return f"ошибка поиска в Википедии: {e}"
    if not hits:
        return "No data"
    try:
        pages = wiki({"action": "query", "prop": "extracts", "explaintext": 1, "exintro": 1,
                       "titles": hits[0]["title"]}, domain=domain)["pages"]
    except Exception as e:
        return f"ошибка чтения статьи: {e}"
    text = " ".join(next(iter(pages.values())).get("extract", "").split())
    return f"[{hits[0]['title']}] {text[:5000]}"


def web_search(query: str) -> str:
    return _wiki_search(query, "en.wikipedia.org")


def web_search_ru(query: str) -> str:
    return _wiki_search(query, "ru.wikipedia.org")


class PageFindArgs(BaseModel):
    title: str = Field(description="точное название статьи Википедии, как оно приходит из web_search в квадратных скобках")
    keywords: str = Field(description="одно-два ключевых слова через пробел; вернутся только строки статьи, где они встречаются")


def page_find(title: str, keywords: str) -> str:
    """Полный текст статьи (без обрезки на вступлении), отфильтрованный по ключевым словам.

    Вступления часто не хватает: например, факт «в 2026 году в Японии» лежит в теле
    длинной обзорной статьи вроде «2026 in Japan», а не в первом абзаце.
    """
    domain = "ru.wikipedia.org" if re.search(r"[а-яёА-ЯЁ]", title + keywords) else "en.wikipedia.org"
    try:
        pages = wiki({"action": "query", "prop": "extracts", "explaintext": 1, "titles": title}, domain=domain)["pages"]
    except Exception as e:
        return f"ошибка чтения страницы: {e}"
    page = next(iter(pages.values()), {})
    if "missing" in page or not page.get("extract"):
        return "No data"
    words = [w.lower() for w in keywords.split() if w]
    if not words:
        return "No data"
    lines = [ln.strip() for ln in page["extract"].splitlines() if ln.strip() and any(w in ln.lower() for w in words)]
    if not lines:
        return "No data"
    return "\n".join(lines[:40])[:5000]


# --- Tavily: второй источник поиска ------------------------------------------

TAVILY_URL = "https://api.tavily.com/search"

# Мой датасет весь на русском, факты — с habr.com и lawebbox.com
# (data/my_fresh_2026.jsonl). Два независимых фильтра, оба настраиваются env:
#   TAVILY_LANGUAGE  — язык результатов, "" отключает фильтр (по умолчанию "ru");
#   TAVILY_DOMAINS   — домены через запятую, пусто — без ограничения по доменам.
# include_domains сам по себе не режет выдачу: по умолчанию Tavily использует
# include_domains_mode="prefer" (домены в приоритете, но не единственные в ответе),
# поэтому включаем "restrict" явно.
TAVILY_LANGUAGE = os.getenv("TAVILY_LANGUAGE", "ru").strip()
TAVILY_DOMAINS = [d.strip() for d in os.getenv("TAVILY_DOMAINS", "habr.com,lawebbox.com").split(",") if d.strip()]


class TavilyArgs(BaseModel):
    query: str = Field(description="поисковый запрос по русскоязычным техническим сайтам (habr.com, lawebbox.com), не Википедия")
    max_results: int = Field(default=5, description="сколько результатов вернуть, от 1 до 10")


def tavily_search(query: str, max_results: int = 5) -> str:
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        return "ошибка: нет TAVILY_API_KEY"
    body = {
        "api_key": key,
        "query": query,
        "max_results": max(1, min(int(max_results or 5), 10)),
        "search_depth": "basic",
        "include_answer": False,
    }
    if TAVILY_LANGUAGE:
        body["language"] = TAVILY_LANGUAGE
        body["filter_by_language"] = True
    if TAVILY_DOMAINS:
        body["include_domains"] = TAVILY_DOMAINS
        body["include_domains_mode"] = "restrict"
    try:
        r = requests.post(TAVILY_URL, json=body, timeout=20)
    except requests.RequestException as e:
        return f"ошибка Tavily: {e}"
    if r.status_code != 200:
        return f"ошибка Tavily: HTTP {r.status_code} {r.text[:300]}"
    hits = (r.json() or {}).get("results") or []
    if not hits:
        return "No data"
    lines = [f"[{h.get('title', '')}] {h.get('url', '')} — {' '.join((h.get('content') or '').split())[:400]}"
             for h in hits]
    return "\n".join(lines)


register(calculator, CalcArgs, "Считает арифметическое выражение: числа, скобки, + - * / ** %")
register(python_exec, ExecArgs, "Выполняет код на Python в отдельном процессе и возвращает то, что он напечатал")
register(web_search, SearchArgs, "Ищет статью в англоязычной Википедии и возвращает её вступление")
register(web_search_ru, SearchArgs, "Ищет статью в русской Википедии (ru.wikipedia.org) и возвращает её вступление")
register(tavily_search, TavilyArgs, "Ищет через Tavily по русскоязычным техническим сайтам (habr.com, lawebbox.com), не по Википедии")
register(page_find, PageFindArgs, "Читает полный текст статьи Википедии и возвращает строки с ключевыми словами; звать после web_search/web_search_ru, когда во вступлении факта нет")


# --------------------------------------------------------------------------
# Цикл агента (с семинара)
# --------------------------------------------------------------------------

SYSTEM = ("Ты отвечаешь на вопросы о недавних событиях. Если нужно посчитать или найти факт, "
          "вызывай инструменты, а не угадывай. Свежие факты ищи через tavily_search, web_search "
          "или web_search_ru; если во вступлении статьи ответа нет, вызови page_find по названию "
          "статьи и ключевым словам. Когда ответ готов, напиши его последней строкой в формате "
          "FINAL: <ответ>. В FINAL только короткий ответ: имя, число, дата или название.")


@dataclass
class Run:
    question: str
    answer: str
    steps: int
    messages: list
    cost: float = 0.0
    seconds: float = 0.0


def looped(seen, calls):
    keys = [(c["function"]["name"], c["function"]["arguments"]) for c in calls]
    repeated = any(k in seen for k in keys)
    seen.update(keys)
    return repeated


def finish(model, messages, step):
    del messages[-1]
    messages.append({"role": "user", "content": "Инструменты больше недоступны. Ответь по тому, что уже известно, последней строкой FINAL: <ответ>."})
    msg = chat(messages, model, tag="agent")
    messages.append(msg)
    return msg.get("content") or "", step + 1


def run_tool(call):
    name = call["function"]["name"]
    try:
        spec = TOOLS[name]
        args = spec["args"].model_validate_json(call["function"]["arguments"])
        result = spec["fn"](**args.model_dump())
    except Exception as e:
        result = f"НЕ удалось вызвать инструмент {name}: {e}"
    return {"role": "tool", "tool_call_id": call["id"], "content": str(result)[:2000]}


def agent_loop(messages, model, tool_names, max_steps):
    seen = set()
    for step in range(1, max_steps + 1):
        msg = chat(messages, model, tools=[TOOLS[n]["schema"] for n in tool_names] or None, tag="agent")
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        if not calls:
            return msg.get("content") or "", step
        if looped(seen, calls) or step == max_steps:
            return finish(model, messages, step)
        messages += [run_tool(c) for c in calls]


def agent(question, model, tool_names, max_steps=8):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    before, started = ledger.total, time.perf_counter()
    answer, steps = agent_loop(messages, model, tool_names, max_steps)
    return Run(question, answer, steps, messages, ledger.total - before, time.perf_counter() - started)


def show_trace(run):
    for m in run.messages[1:]:
        calls = "; ".join(f"{c['function']['name']}{c['function']['arguments']}" for c in m.get("tool_calls") or [])
        text = " ".join((m.get("content") or "").split())[:180]
        print(f"{m['role']:9s}| {text} {calls}")
    print(f"шагов: {run.steps}, цена: {run.cost * 100:.3f} ¢, время: {run.seconds:.1f} c")


def final_answer(text):
    m = re.search(r"FINAL:\s*(.+)", text or "")
    return m.group(1).strip() if m else (text or "").strip()


def normalize(text):
    text = re.sub(r"[^\w\s]", " ", str(text).lower().replace(",", ""))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def is_correct(item, answer):
    return normalize(item["answer"]) in normalize(answer)


# --------------------------------------------------------------------------
# Демонстрация на датасете
# --------------------------------------------------------------------------

def load_tasks(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    ap = argparse.ArgumentParser(description="Прогон агента на датасете свежих вопросов")
    ap.add_argument("--data", default="data/my_fresh_2026.jsonl")
    ap.add_argument("--n", type=int, default=8, help="сколько вопросов взять с начала файла")
    ap.add_argument("--model", default="cheap", choices=list(MODELS))
    ap.add_argument("--tools", default="calculator,python_exec,web_search,web_search_ru,tavily_search,page_find")
    ap.add_argument("--max-steps", type=int, default=8)
    args = ap.parse_args()

    tasks = load_tasks(args.data)[: args.n]
    tool_names = [t for t in args.tools.split(",") if t]
    model = MODELS[args.model]

    TRACES.mkdir(exist_ok=True)
    out_dir = TRACES / f"demo-{args.model}-{args.tools}"
    out_dir.mkdir(parents=True, exist_ok=True)

    correct = 0
    for task in tasks:
        try:
            run = agent(task["question"], model, tool_names, max_steps=args.max_steps)
            ans = final_answer(run.answer)
            ok = is_correct(task, ans)
        except Exception as e:
            ans, ok, run = f"ошибка: {e}", False, Run(task["question"], f"ошибка: {e}", 0, [])
        correct += ok
        print(f"[{'OK ' if ok else 'НЕТ'}] {task['id']}: {task['question'][:70]}")
        print(f"      ответ агента: {ans[:100]!r}  эталон: {task['answer']!r}  шагов: {run.steps}")
        (out_dir / f"{task['id']}.json").write_text(json.dumps({
            "task": task, "answer": run.answer, "final": ans, "correct": ok,
            "cost": run.cost, "seconds": run.seconds, "steps": run.steps, "messages": run.messages,
        }, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\nверно: {correct}/{len(tasks)} ({correct / len(tasks):.0%}), цена: ${ledger.total:.4f}, "
          f"трейсы: {out_dir}/")


if __name__ == "__main__":
    main()
