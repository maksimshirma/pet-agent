# -*- coding: utf-8 -*-
"""ДЗ2, Шаг 3: замер RAG против живого поиска ДЗ1, на всех вопросах data/questions_habr.jsonl.

Запуск: python measure_rag.py
Результат: results_rag.csv, img/money_rag.png, results_refusals.csv (таблица отказов),
трейсы traces_rag/<конфигурация>/<модель>/<id>.json.
"""
import json
import time
from pathlib import Path

import pandas as pd

import agent as A
import rag as R
import measure as M

TRACES = Path("traces_rag")

CONFIGS = [
    # (config, model_key, kind)
    ("сильная, без инструментов", "strong", "no_tools"),
    ("дешёвая, живой поиск (ДЗ1)", "cheap", "hw1_search"),
    ("дешёвая, RAG всегда", "cheap", "rag"),
    ("дешёвая, агент с knowledge_base", "cheap", "kb_agent"),
    ("средняя, RAG всегда", "mid", "rag"),
]


def run_one(task, model, kind, max_steps=6):
    if kind == "no_tools":
        return A.no_tools_answer(task["question"], model)
    if kind == "hw1_search":
        return A.agent(task["question"], model, M.SEARCH, max_steps=max_steps)
    if kind == "rag":
        return R.rag_run(task["question"], model, k=5, name="habr_lines")
    if kind == "kb_agent":
        return R.kb_agent_run(task["question"], model, max_steps=4)
    raise ValueError(kind)


def run_config(tasks, config, model_key, kind):
    model = A.MODELS[model_key]
    folder = TRACES / config / model.split("/")[-1]
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, task in enumerate(tasks, 1):
        try:
            run = run_one(task, model, kind)
            final = A.final_answer(run.answer)
        except Exception as e:
            run, final = A.Run(task["question"], f"ошибка: {e}", 0, []), f"ошибка: {e}"
        refused = final.strip().upper().startswith("NOT_FOUND")
        answerable = task.get("answerable", True)
        correct = bool(answerable and A.is_correct(task, final))
        rows.append({"config": config, "model": model.split("/")[-1], "id": task["id"],
                     "answerable": answerable, "correct": correct, "refused": refused,
                     "steps": run.steps, "cost": run.cost, "seconds": round(run.seconds, 2),
                     "answer": final[:80], "gold": task.get("answer")})
        (folder / f"{task['id']}.json").write_text(json.dumps({
            "task": task, "answer": run.answer, "final": final, "correct": correct, "refused": refused,
            "cost": run.cost, "seconds": run.seconds, "steps": run.steps, "messages": run.messages,
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  [{i}/{len(tasks)}] {'OK ' if correct else ('ОТК' if refused else 'НЕТ')} {task['id']}", flush=True)
    return pd.DataFrame(rows)


def refusal_table(results):
    rows = []
    for (config, model), df in results.groupby(["config", "model"], sort=False):
        unans = df[~df["answerable"]]
        ans = df[df["answerable"]]
        rows.append({
            "config": config, "model": model,
            "честно_отклонил_из_10": int(unans["refused"].sum()),
            "ложно_отклонил_из_answerable": int(ans["refused"].sum()),
            "answerable_n": len(ans),
        })
    return pd.DataFrame(rows)


def main():
    R.load_cache()
    tasks = [json.loads(l) for l in Path("data/questions_habr.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"вопросов: {len(tasks)} ({sum(t['answerable'] for t in tasks)} с ответом, "
          f"{sum(not t['answerable'] for t in tasks)} без ответа)\n")

    frames = []
    for config, model_key, kind in CONFIGS:
        print(f"=== {config} ({model_key}) ===")
        started = time.perf_counter()
        frames.append(run_config(tasks, config, model_key, kind))
        print(f"    время: {time.perf_counter() - started:.1f} c, потрачено всего: ${A.ledger.total:.4f}\n")

    results = pd.concat(frames, ignore_index=True)
    results.to_csv("results_rag.csv", index=False)

    table = M.report(results[results["answerable"]])
    print(table.to_string(index=False))
    M.money_chart(table, "img/money_rag.png")

    refusals = refusal_table(results)
    refusals.to_csv("results_refusals.csv", index=False)
    print("\n" + refusals.to_string(index=False))

    print(f"\nитого потрачено: ${A.ledger.total:.4f}")
    print("сохранено: results_rag.csv, img/money_rag.png, results_refusals.csv, traces_rag/…")


if __name__ == "__main__":
    main()
