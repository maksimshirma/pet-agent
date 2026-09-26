# -*- coding: utf-8 -*-
"""Корпус для ДЗ2: полные тексты статей Habr/lawebbox, на которые ссылаются
вопросы моей первой домашки (data/my_fresh_2026.jsonl). fetch_corpus.py из
семинара дёргает MediaWiki API и для Habr не подходит — здесь свой HTML-парсер.

Запуск: python fetch_habr_corpus.py
Результат: data/corpus_habr.jsonl — {"page", "url", "text"}, где text — статья
построчно, разделы отмечены строками "== Заголовок ==" (как в MediaWiki extract,
чтобы chunk_lines/chunk_sections могли трекать раздел тем же способом, что в семинаре).
"""
import json
import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)
QUESTIONS = DATA / "my_fresh_2026.jsonl"
HEADERS = {"User-Agent": "Mozilla/5.0 (agents-course-hw2/1.0; educational project)"}
CONTENT_SELECTORS = ["div.tm-article-presenter__body", "div.tm-article-body", "article"]
HEADING_TAGS = ("h1", "h2", "h3", "h4")


def fetch_html(url, attempts=4):
    for attempt in range(attempts):
        try:
            r = requests.get(url, headers=HEADERS, timeout=40)
            if r.status_code == 200:
                return r.text
        except requests.RequestException:
            pass
        time.sleep(2 * (attempt + 1))
    return ""


def extract_title(soup):
    title = soup.title.get_text(strip=True) if soup.title else ""
    return re.sub(r"\s*/\s*Хабр\s*$", "", title).strip()


def extract_blocks(soup):
    root = None
    for sel in CONTENT_SELECTORS:
        root = soup.select_one(sel)
        if root:
            break
    if root is None:
        return []
    blocks = []
    for el in root.find_all(list(HEADING_TAGS) + ["p", "li"]):
        if el.name == "p" and el.find_parent("li"):
            continue  # li оборачивает свой текст в <p>, не дублировать
        text = el.get_text(" ", strip=True)
        if not text:
            continue
        if el.name in HEADING_TAGS:
            blocks.append(("heading", text))
        elif len(text) >= 15:
            blocks.append(("item", text))
    return blocks


def page_text(url):
    html = fetch_html(url)
    if not html:
        return "", ""
    soup = BeautifulSoup(html, "lxml")
    title = extract_title(soup)
    lines = []
    for kind, text in extract_blocks(soup):
        lines.append(f"== {text} ==" if kind == "heading" else text)
    return title, "\n".join(lines)


# Дополнительные выпуски дайджеста без вопросов — чистый балласт для корпуса, поднимает
# число кусков заметно выше требуемых 300 на обеих нарезках и делает поиск нетривиальным
# (не на каждой странице корпуса есть вопрос).
EXTRA_PAGES = {
    "https://habr.com/ru/articles/1012160/": "Frontend Status #9",
    "https://habr.com/ru/articles/1018828/": "Frontend Status #11",
}

if __name__ == "__main__":
    tasks = [json.loads(l) for l in QUESTIONS.read_text(encoding="utf-8").splitlines() if l.strip()]
    seen = {}
    for t in tasks:
        seen.setdefault(t["url"], t["page"])
    for url, page in EXTRA_PAGES.items():
        seen.setdefault(url, page)

    corpus = []
    for url, page in seen.items():
        title, text = page_text(url)
        corpus.append({"page": page, "url": url, "title": title, "text": text})
        print(f"{page:28s} {len(text):8d} символов  {url}")
        time.sleep(1)

    with (DATA / "corpus_habr.jsonl").open("w", encoding="utf-8") as f:
        for c in corpus:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"\nстраниц: {len(corpus)}, символов: {sum(len(c['text']) for c in corpus)}")
