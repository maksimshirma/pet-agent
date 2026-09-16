# Pet Agent

Учебный проект по разработке собственного LLM-агента на Python.

## Prerequisites

- Python 3.10+ Проверить версию: python --version
- Аккаунт OpenRouter с API-ключом (формат sk-or-v1-...)

## Quick Start

1. Клонируйте репозиторий и перейдите в его директорию

```bash
   git clone <ссылка-на-репозиторий>
   cd pet-agent
```

2. Установите зависимости

```bash
   pip install requests pydantic python-dotenv
```

3. Настройте переменные окружения

```bash
   cp .env.example .env
```

4. Впишите ваш ключ в .env:
   OPENROUTER_API_KEY=sk-or-v1-...
