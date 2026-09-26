# -*- coding: utf-8 -*-
"""Тесты RAG-инструмента и памяти: три случая на каждый, без модели, без ключа,
без поднятого Milvus (сеть и Milvus-клиент подменены), по образцу test_tools.py.

Запуск: python test_rag.py
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import rag


def ok(name):
    print(f"  ok: {name}")


# --- knowledge_base ------------------------------------------------------------

def test_knowledge_base():
    hits = [{"page": "Frontend Status #8", "section": "🌎 Браузеры", "url": "",
             "text": "Команда Edge предлагает новый HTML-атрибут focusgroup"}]
    with patch("rag.search_milvus", return_value=hits):
        out = rag.knowledge_base("focusgroup")
    assert "Frontend Status #8" in out and "focusgroup" in out                  # нормальный вход

    with patch("rag.search_milvus", return_value=[]):
        out = rag.knowledge_base("qwxzv нонсенс 999999")
    assert out == "ничего не найдено"                                          # пустой результат

    with patch("rag.search_milvus", side_effect=RuntimeError("milvus недоступен")):
        out = rag.knowledge_base("что угодно")
    assert "ошибка" in out                                                     # ошибка
    ok("knowledge_base")


# --- память: recall_facts --------------------------------------------------------

def test_recall_facts():
    with patch("rag.get_client") as fake_get_client:
        fake_get_client.return_value.has_collection.return_value = False
        assert rag.recall_facts("вопрос") == []                                # пустая память

    with patch("rag.get_client") as fake_get_client, patch("rag.search_milvus") as fake_search:
        fake_get_client.return_value.has_collection.return_value = True
        fake_search.return_value = [{"text": "живёт в Калининграде"}, {"text": "работает фронтендером"}]
        found = rag.recall_facts("где живёт пользователь")
    assert found == ["живёт в Калининграде", "работает фронтендером"]           # найденный факт

    with patch("rag.get_client") as fake_get_client:
        fake_get_client.return_value.has_collection.return_value = True
        with patch("rag.search_milvus", return_value=[]):
            assert rag.recall_facts("вопрос") == []                            # факт не найден
    ok("recall_facts")


# --- память: store_facts (запись и замена) ----------------------------------------

def test_store_facts_replaces_not_duplicates():
    tmp_facts = Path("test_facts_tmp.json")
    if tmp_facts.exists():
        tmp_facts.unlink()
    with patch("rag.FACTS_FILE", tmp_facts), \
         patch("rag.get_client"), \
         patch("rag.embed_cached", return_value=__import__("numpy").zeros((1, rag.DIM))), \
         patch("rag.index_to_milvus", return_value={"row_count": 1}):
        rag.store_facts(["живёт в Калининграде"])
        assert rag.known_facts() == ["живёт в Калининграде"]                   # запись факта

        rag.store_facts(["живёт в Санкт-Петербурге"])                          # факт отменяет старый
        assert rag.known_facts() == ["живёт в Санкт-Петербурге"]               # заменённый факт, не оба

    with patch("rag.FACTS_FILE", tmp_facts), patch("rag.get_client") as fake_get_client:
        fake_get_client.return_value.has_collection.return_value = True
        rag.store_facts([])                                                    # пустой список фактов
        assert rag.known_facts() == []
        fake_get_client.return_value.drop_collection.assert_called_once_with("facts")
    tmp_facts.unlink()
    ok("store_facts")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nвсе тесты прошли: {len(tests)}")
    sys.exit(0)
