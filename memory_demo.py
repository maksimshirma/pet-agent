# -*- coding: utf-8 -*-
"""ДЗ2, Шаг 4: демонстрация памяти на двух сессиях + график токенов.

Запуск: python memory_demo.py
Результат: episodes.jsonl, facts.json, коллекция Milvus "facts", img/memory_tokens.png,
печать двух сессий и итоговых фактов.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

import agent as A
import rag as R

COLORS = {"violet": "#5436A3", "amber": "#F09000"}


def two_session_demo():
    print("== сессия 1 ==")
    s1 = []
    reply = R.talk("s1", s1,
                    "Привет! Меня зовут Макс, я фронтенд-разработчик, живу в Калининграде и "
                    "увлекаюсь хоккеем. Кстати, какой новый HTML-атрибут предложила команда Edge?")
    print("assistant:", reply[:200])
    print("факты после сессии 1:", R.end_session(s1))

    print("\n== сессия 2 (история сессии 1 не в контексте) ==")
    s2 = []
    reply = R.talk("s2", s2, "Напомни, где я живу и чем увлекаюсь?")
    print("assistant:", reply[:200])

    print("\n== пользователь меняет факт ==")
    reply = R.talk("s2", s2, "Кстати, я переехал в Санкт-Петербург и теперь увлекаюсь футболом, а не хоккеем.")
    print("assistant:", reply[:200])
    print("факты после сессии 2:", R.end_session(s2))


def token_chart(n_turns=12):
    questions = [json.loads(l) for l in Path("data/questions_habr.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    turns = [q["question"] for q in questions if q["answerable"]][:n_turns]

    before = len(A.ledger.calls)
    hist_full, hist_win = [], []
    for q in turns:
        R.talk("bench-full", hist_full, q, mode="вся история")
        R.talk("bench-window", hist_win, q, mode="память")
    calls = A.ledger.calls[before:]

    full_calls = [c for c in calls if c["tag"] == "dialog: вся история"]
    window_calls = [c for c in calls if c["tag"] == "dialog: память"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(range(1, len(full_calls) + 1), [c["prompt"] for c in full_calls], "o-", color=COLORS["amber"], label="вся история в контексте")
    ax.plot(range(1, len(window_calls) + 1), [c["prompt"] for c in window_calls], "o-", color=COLORS["violet"], label="окно + факты")
    ax.set_xlabel("реплика диалога")
    ax.set_ylabel("токенов на входе")
    ax.legend()
    ax.grid(alpha=0.3)
    Path("img").mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig("img/memory_tokens.png", dpi=150)

    cost_full = sum(c["cost"] for c in full_calls)
    cost_window = sum(c["cost"] for c in window_calls)
    print(f"\nвся история: {len(full_calls)} реплик, {sum(c['prompt'] for c in full_calls)} токенов на входе суммарно, ${cost_full:.5f}")
    print(f"окно+факты:  {len(window_calls)} реплик, {sum(c['prompt'] for c in window_calls)} токенов на входе суммарно, ${cost_window:.5f}")


if __name__ == "__main__":
    R.load_cache()
    two_session_demo()
    print("\n" + "=" * 60)
    token_chart()
    R.save_cache()
    print(f"\nитого потрачено на память: ${A.ledger.total:.4f}")
