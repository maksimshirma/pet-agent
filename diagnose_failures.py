# -*- coding: utf-8 -*-
"""ДЗ2, Шаг 3: разбор провалов лучшей конфигурации на две кучки — "поиск не нашёл"
и "нашёл, но ответ неверный" (ноутбук, ячейка 39, diagnose()).

Запуск: python diagnose_failures.py [config] [model]
Без аргументов берёт конфигурацию с максимальной accuracy среди answerable-вопросов.
"""
import json
import sys
from pathlib import Path

import pandas as pd

import rag as R


def diagnose(config, model):
    df = pd.read_csv("results_rag.csv")
    df = df[(df["config"] == config) & (df["model"] == model) & df["answerable"] & ~df["correct"]]
    missed, wrong = [], []
    for _, row in df.iterrows():
        trace = json.loads((Path("traces_rag") / config / model / f"{row['id']}.json").read_text(encoding="utf-8"))
        task = trace["task"]
        hits = R.search_milvus("habr_lines", task["question"], 5)
        gold_found = any(R.is_gold(h, task) for h in hits)
        (wrong if gold_found else missed).append({"row": row, "task": task, "hits": hits})
    return missed, wrong


def show(item, label):
    row, task, hits = item["row"], item["task"], item["hits"]
    print(f"\n--- {label}: {row['id']} ---")
    print("вопрос:", task["question"])
    print("эталон:", task["answer"], "| эталонная страница:", task["page"])
    print("evidence:", task["evidence"])
    print("ответ агента:", row["answer"])
    print("топ-5 из базы:")
    for h in hits:
        mark = "<-- gold" if R.is_gold(h, task) else ""
        print(f"   [{h['page']} / {h['section']}] {h['text'][:100]} {mark}")


if __name__ == "__main__":
    table = pd.read_csv("results_rag.csv")
    scored = table[table["answerable"]].groupby(["config", "model"])["correct"].mean().sort_values(ascending=False)
    config, model = sys.argv[1:3] if len(sys.argv) > 2 else scored.index[0]
    print(f"лучшая конфигурация: {config} ({model}), accuracy={scored.iloc[0]:.3f}\n")

    R.load_cache()
    missed, wrong = diagnose(config, model)
    print(f"поиск не нашёл: {len(missed)}, нашёл-но-неверно: {len(wrong)}")
    if missed:
        show(missed[0], "поиск не нашёл")
    if wrong:
        show(wrong[0], "нашёл, но ответ неверный")
