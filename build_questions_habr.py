# -*- coding: utf-8 -*-
"""ДЗ2, Шаг 1: вопросы для RAG на моём Habr-корпусе.

Берёт 72 вопроса из ДЗ1 (data/my_fresh_2026.jsonl), оставляет только те, чья цитата
(evidence) реально находится в свежескачанном data/corpus_habr.jsonl — статьи могли
измениться с даты ДЗ1. Добавляет 10 вопросов без ответа: по теме корпуса (свежие новости
фронтенда/AI 2026), но про факты, которых в этих 10 статьях нет (проверено grep'ом).

Запуск: python build_questions_habr.py
Результат: data/questions_habr.jsonl
"""
import json
import re
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"

UNANSWERABLE = [
    "Какая мажорная версия Vue.js вышла в 2026 году?",
    "Что нового появилось в Svelte 6?",
    "Какую версию Deno представили в 2026 году?",
    "Что изменилось в Bun 2.0?",
    "Какая версия Vite вышла вслед за Vite 7?",
    "Какая мажорная версия Tailwind CSS анонсирована в 2026 году?",
    "Какие изменения принесла новая версия ESLint в 2026 году?",
    "Что показала Apple на WWDC 2026 для веб-разработчиков?",
    "Какой браузер первым внедрил поддержку CSS if()?",
    "Сколько разработчиков используют Astro по данным опроса State of JS 2026?",
]


def norm(t):
    return " ".join(re.sub(r"[^\w\s]", " ", str(t).lower()).split())


if __name__ == "__main__":
    corpus = {c["page"]: norm(c["text"]) for c in (json.loads(l) for l in (DATA / "corpus_habr.jsonl").read_text(encoding="utf-8").splitlines())}
    rows = [json.loads(l) for l in (DATA / "my_fresh_2026.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    kept, dropped = [], []
    for r in rows:
        ev = norm(r["evidence"])[:120]
        (kept if ev in corpus.get(r["page"], "") else dropped).append(r)

    for d in dropped:
        print(f"дропнут (нет цитаты в свежей статье): {d['id']} {d['page']} | {d['evidence'][:70]}")

    out = []
    for n, r in enumerate(kept):
        out.append({"id": f"habr-{n}", "source": "habr", "question": r["question"], "answer": r["answer"],
                    "evidence": r["evidence"], "page": r["page"], "url": r["url"], "answerable": True})
    for n, q in enumerate(UNANSWERABLE):
        out.append({"id": f"unans-{n}", "source": "habr", "question": q, "answer": None,
                    "evidence": None, "page": None, "url": None, "answerable": False})

    with (DATA / "questions_habr.jsonl").open("w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nвопросов с ответом: {len(kept)} из {len(rows)}, без ответа: {len(UNANSWERABLE)}, всего: {len(out)}")
