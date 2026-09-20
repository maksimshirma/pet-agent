# -*- coding: utf-8 -*-
"""Домашка 1

Пять конфигураций, обязательные по заданию:

  1. сильная модель, без инструментов      — ориентир, один вызов на задачу;
  2. дешёвая модель, без инструментов      — честный ноль;
  3. дешёвая модель, только поиск          — базовый агент;
  4. дешёвая модель, поиск и чтение страницы — мой лучший агент на дешёвой;
  5. средняя модель, лучший набор инструментов — сколько добавляет модель посильнее.

Запуск:

    python measure.py --data data/my_fresh_2026.jsonl --n 50

Результат: results_raw.csv, results.txt, img/money_chart.png, трейсы в traces/<конфигурация>/<модель>/<id>.json.
"""
import json
import time
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

import agent as A

COLORS = {"violet": "#5436A3", "amber": "#F09000", "teal": "#00838F", "red": "#C43C3C", "grey": "#787882"}

SEARCH = ["web_search", "web_search_ru", "tavily_search"]
SEARCH_AND_READ = SEARCH + ["page_find"]
BEST_TOOLS = SEARCH_AND_READ + ["calculator", "python_exec"]

CONFIGS = [
    # (config, model_key, tool_names or None for "без инструментов")
    ("сильная, без инструментов", "strong", None),
    ("дешёвая, без инструментов", "cheap", None),
    ("дешёвая, только поиск", "cheap", SEARCH),
    ("дешёвая, поиск и чтение страницы", "cheap", SEARCH_AND_READ),
    ("средняя, лучший набор инструментов", "mid", BEST_TOOLS),
]


def run_one(task, model, tool_names, max_steps=8):
    if tool_names is None:
        return A.no_tools_answer(task["question"], model)
    return A.agent(task["question"], model, tool_names, max_steps=max_steps)


def run_config(tasks, config, model_key, tool_names, max_steps=8):
    model = A.MODELS[model_key]
    folder = A.TRACES / config / model.split("/")[-1]
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, task in enumerate(tasks, 1):
        try:
            run = run_one(task, model, tool_names, max_steps)
            final = A.final_answer(run.answer)
        except Exception as e:
            run, final = A.Run(task["question"], f"ошибка: {e}", 0, []), f"ошибка: {e}"
        ok = A.is_correct(task, final)
        rows.append({"config": config, "model": model.split("/")[-1], "id": task["id"], "correct": ok,
                     "steps": run.steps, "cost": run.cost, "seconds": round(run.seconds, 2),
                     "answer": final[:80], "gold": task["answer"]})
        (folder / f"{task['id']}.json").write_text(json.dumps({
            "task": task, "answer": run.answer, "final": final, "correct": ok,
            "cost": run.cost, "seconds": run.seconds, "steps": run.steps, "messages": run.messages,
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  [{i}/{len(tasks)}] {'OK ' if ok else 'НЕТ'} {task['id']}", flush=True)
    return pd.DataFrame(rows)


def summary(config, model, n, correct, cost, steps, seconds):
    return {"config": config, "model": model, "n": n, "accuracy": round(correct / n, 3),
            "cost_per_task": round(cost / n, 5), "cost_per_correct": round(cost / correct, 5) if correct else float("inf"),
            "avg_steps": round(steps, 2), "avg_seconds": round(seconds, 1)}


def report(results):
    rows = [summary(config, model, len(df), int(df["correct"].sum()), df["cost"].sum(),
                     df["steps"].mean(), df["seconds"].mean())
            for (config, model), df in results.groupby(["config", "model"], sort=False)]
    return pd.DataFrame(rows)


def money_chart(df, path="img/money_chart.png"):
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    x = df["cost_per_task"] * 100
    y = df["accuracy"] * 100
    ax.scatter(x, y, s=150, color=COLORS["violet"], zorder=3)
    # цена задачи растёт на два порядка между конфигурациями (0.005 -- 2 цента):
    # на линейной шкале дешёвые точки слипаются в один пиксель у левого края.
    ax.set_xscale("log")
    offsets = [(10, 8), (10, -14), (10, 10), (10, -18), (-12, 10)]
    aligns = ["left", "left", "left", "left", "right"]
    for (_, r), (dx, dy), ha in zip(df.iterrows(), offsets, aligns):
        ax.annotate(f"{r['config']}\n({r['model']})", (r["cost_per_task"] * 100, r["accuracy"] * 100),
                    xytext=(dx, dy), textcoords="offset points", fontsize=8, ha=ha)
    ax.set_xlabel("цена задачи, центы (логарифмическая шкала)")
    ax.set_ylabel("доля верных, %")
    ax.set_ylim(-8, 108)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    return fig
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/my_fresh_2026.jsonl")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=8)
    args = ap.parse_args()

    tasks = A.load_tasks(args.data)[: args.n]
    print(f"датасет: {args.data}, задач: {len(tasks)}\n")

    frames = []
    for config, model_key, tool_names in CONFIGS:
        print(f"=== {config} ({model_key}) ===")
        started = time.perf_counter()
        frames.append(run_config(tasks, config, model_key, tool_names, args.max_steps))
        print(f"    время конфигурации: {time.perf_counter() - started:.1f} c, "
              f"потрачено всего: ${A.ledger.total:.4f}\n")

    results = pd.concat(frames, ignore_index=True)
    results.to_csv("results_raw.csv", index=False)

    table = report(results)
    Path("results.txt").write_text(table.to_string(index=False), encoding="utf-8")
    print(table.to_string(index=False))

    money_chart(table, "img/money_chart.png")
    print(f"\nитого потрачено: ${A.ledger.total:.4f}")
    print("сохранено: results_raw.csv, results.txt, img/money_chart.png, traces/<конфигурация>/<модель>/<id>.json")


if __name__ == "__main__":
    main()
