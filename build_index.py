# -*- coding: utf-8 -*-
"""ДЗ2, Шаг 2: индекс в Milvus, кривая recall@k, гибрид RRF.

Запуск: python build_index.py
Результат: коллекции Milvus "habr_chars"/"habr_lines", data/embeddings_habr.npz,
img/recall.png, results_hybrid.csv (таблица «вектор, слова, гибрид»), печать выбранного k.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import rag as R

COLORS = {"violet": "#5436A3", "amber": "#F09000", "teal": "#00838F"}


def emb_text(chunk):
    return f"{chunk['page']}: {chunk['text']}"


def main():
    R.load_cache()
    corpus = R.load_corpus()
    tasks = [json.loads(l) for l in Path("data/questions_habr.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    char_chunks = [c for p in corpus for c in R.chunk_chars(p)]
    line_chunks = [c for p in corpus for c in R.chunk_sections(p)]
    print(f"наивная нарезка: {len(char_chunks)} кусков, осмысленная: {len(line_chunks)} кусков")

    char_vecs = R.embed_cached([emb_text(c) for c in char_chunks])
    line_vecs = R.embed_cached([emb_text(c) for c in line_chunks])
    q_vecs = R.embed_cached([t["question"] for t in tasks])
    R.save_cache()

    print("\n== индекс в Milvus ==")
    print("habr_chars:", R.index_to_milvus("habr_chars", char_chunks, char_vecs))
    print("habr_lines:", R.index_to_milvus("habr_lines", line_chunks, line_vecs))

    print("\n== demo фильтра по странице ==")
    for h in R.search_milvus("habr_lines", "уязвимость безопасность", 3, 'page == "Frontend Status #8"'):
        print(" ", h["page"], "|", h["text"][:70])

    print("\n== recall@k ==")
    ks = (1, 3, 5, 10)
    char_recall = R.recall_curve(char_chunks, char_vecs, tasks, q_vecs, ks)
    line_recall = R.recall_curve(line_chunks, line_vecs, tasks, q_vecs, ks)
    for r in char_recall:
        print("наивная  k=%2d recall=%.3f" % (r["k"], r["recall"]))
    for r in line_recall:
        print("осмысл.  k=%2d recall=%.3f" % (r["k"], r["recall"]))

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(ks, [r["recall"] for r in char_recall], "o-", color=COLORS["amber"], label="наивная (300 симв.)")
    ax.plot(ks, [r["recall"] for r in line_recall], "o-", color=COLORS["violet"], label="осмысленная (абзац/пункт)")
    ax.set_xlabel("k")
    ax.set_ylabel("recall@k")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(ks)
    ax.legend()
    ax.grid(alpha=0.3)
    Path("img").mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig("img/recall.png", dpi=150)

    print("\n== гибрид: вектор / слова / RRF (осмысленная нарезка) ==")
    docs, idf = R.build_keyword_index([c["text"] for c in line_chunks])
    rows = []
    for k in ks:
        v_hits = w_hits = h_hits = 0
        n = 0
        for task, qv in zip(tasks, q_vecs):
            if not task.get("answerable", True):
                continue
            n += 1
            vec_rank = [i for i, _ in R.search_numpy(qv, line_vecs, 20)]
            kw_rank = [i for i, _ in R.keyword_search(task["question"], docs, idf, 20)]
            hybrid = R.rrf([vec_rank, kw_rank], k=k)
            if any(R.is_gold(line_chunks[i], task) for i in vec_rank[:k]):
                v_hits += 1
            if any(R.is_gold(line_chunks[i], task) for i in kw_rank[:k]):
                w_hits += 1
            if any(R.is_gold(line_chunks[i], task) for i in hybrid):
                h_hits += 1
        rows.append({"k": k, "вектор": round(v_hits / n, 3), "слова": round(w_hits / n, 3), "гибрид": round(h_hits / n, 3)})
    hybrid_table = pd.DataFrame(rows)
    hybrid_table.to_csv("results_hybrid.csv", index=False)
    print(hybrid_table.to_string(index=False))


if __name__ == "__main__":
    main()
