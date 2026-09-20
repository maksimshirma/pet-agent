# -*- coding: utf-8 -*-
"""Тесты инструментов агента: три случая на каждый — нормальный вход, пустой результат, ошибка.

Работают без модели и без ключей: сеть подменена (`monkeypatch` на requests.get/post),
поэтому `python test_tools.py` можно гонять в CI без .env.
"""
import sys
from unittest.mock import patch

import agent


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text="", content_type="application/json"):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text or str(payload)
        self.headers = {"content-type": content_type}

    def json(self):
        return self._payload


def ok(name):
    print(f"  ok: {name}")


# --- calculator --------------------------------------------------------------

def test_calculator():
    assert agent.calculator("17*23+5") == "396"                 # нормальный вход
    assert agent.calculator("2/4") == "0.5"                     # нормальный вход, дробь
    assert "ошибка" in agent.calculator("__import__('os')")     # ошибка: не арифметика
    ok("calculator")


# --- python_exec ---------------------------------------------------------------

def test_python_exec():
    assert agent.python_exec("print(2 + 2)") == "4"                       # нормальный вход
    assert agent.python_exec("pass") == "Код исполнился но результат пустой"  # пустой результат
    assert "ZeroDivisionError" in agent.python_exec("1/0")                # ошибка
    ok("python_exec")


# --- web_search / web_search_ru (обёртки над wiki()) --------------------------

def test_web_search():
    search_hit = FakeResponse(200, {"query": {"search": [{"title": "Ada Lovelace"}]}})
    extract_hit = FakeResponse(200, {"query": {"pages": {"1": {"extract": "Ada Lovelace was a mathematician."}}}})
    with patch("agent.requests.get", side_effect=[search_hit, extract_hit]):
        out = agent.web_search("Ada Lovelace")
    assert out.startswith("[Ada Lovelace]") and "mathematician" in out       # нормальный вход

    empty_hit = FakeResponse(200, {"query": {"search": []}})
    with patch("agent.requests.get", side_effect=[empty_hit]):
        out = agent.web_search("qwxzv nonsense 12345")
    assert out == "No data"                                                  # пустой результат

    with patch("agent.requests.get", side_effect=RuntimeError("сеть недоступна")):
        out = agent.web_search("что угодно")
    assert "ошибка" in out                                                   # ошибка
    ok("web_search")


def test_web_search_ru():
    search_hit = FakeResponse(200, {"query": {"search": [{"title": "Пётр I"}]}})
    extract_hit = FakeResponse(200, {"query": {"pages": {"1": {"extract": "Пётр I — русский царь."}}}})
    with patch("agent.requests.get", side_effect=[search_hit, extract_hit]):
        out = agent.web_search_ru("Пётр I")
    assert out.startswith("[Пётр I]") and "царь" in out                      # нормальный вход

    empty_hit = FakeResponse(200, {"query": {"search": []}})
    with patch("agent.requests.get", side_effect=[empty_hit]):
        out = agent.web_search_ru("абырвалг несуществующий")
    assert out == "No data"                                                  # пустой результат

    bad_status = FakeResponse(500, {}, text="Internal Error")
    with patch("agent.requests.get", side_effect=[bad_status, bad_status, bad_status]), \
         patch("agent.time.sleep", return_value=None):
        out = agent.web_search_ru("что угодно")
    assert "ошибка" in out                                                   # ошибка
    ok("web_search_ru")


# --- page_find -----------------------------------------------------------------

def test_page_find():
    body = "Вступление статьи.\nВ марте 2026 года запустили ракету.\nДругая строка ни о чём."
    hit = FakeResponse(200, {"query": {"pages": {"1": {"extract": body}}}})
    with patch("agent.requests.get", side_effect=[hit]):
        out = agent.page_find("2026 in spaceflight", "ракету")
    assert "запустили ракету" in out                                         # нормальный вход

    hit_no_match = FakeResponse(200, {"query": {"pages": {"1": {"extract": body}}}})
    with patch("agent.requests.get", side_effect=[hit_no_match]):
        out = agent.page_find("2026 in spaceflight", "динозавры")
    assert out == "No data"                                                  # пустой результат (нет совпадений)

    with patch("agent.requests.get", side_effect=RuntimeError("сеть недоступна")):
        out = agent.page_find("2026 in spaceflight", "ракету")
    assert "ошибка" in out                                                   # ошибка
    ok("page_find")


# --- tavily_search ---------------------------------------------------------------

def test_tavily_search():
    payload = {"results": [
        {"title": "Frontend Status #8", "url": "https://habr.com/ru/articles/1009296/",
         "content": "с 15 марта 2026 года лимит падает с 398 до 200 дней"},
    ]}
    with patch.dict("os.environ", {"TAVILY_API_KEY": "fake-key"}), \
         patch("agent.requests.post", return_value=FakeResponse(200, payload)):
        out = agent.tavily_search("TLS сертификаты 200 дней")
    assert "Frontend Status" in out and "15 марта 2026" in out               # нормальный вход

    with patch.dict("os.environ", {"TAVILY_API_KEY": "fake-key"}), \
         patch("agent.requests.post", return_value=FakeResponse(200, {"results": []})):
        out = agent.tavily_search("qwxzv нонсенс 999999")
    assert out == "No data"                                                  # пустой результат

    with patch.dict("os.environ", {"TAVILY_API_KEY": ""}, clear=False):
        del_key = "TAVILY_API_KEY" in __import__("os").environ
        if del_key:
            del __import__("os").environ["TAVILY_API_KEY"]
        out = agent.tavily_search("что угодно")
    assert "ошибка" in out                                                   # ошибка: нет ключа
    ok("tavily_search")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nвсе тесты прошли: {len(tests)}")
    sys.exit(0)
