# Chess Move Predictor API

API для предсказания шахматных ходов с использованием нейронной сети на базе FastAPI и Docker.

## Возможности

- Предсказание лучших ходов для любой шахматной позиции
- Автоматическая валидация FEN строк
- Поддержка GPU/CPU
- RESTful API с документацией Swagger

## Быстрый запуск

### Использование Docker Compose (рекомендуется)

```bash
# Сборка и запуск
docker-compose up --build

# Запуск в фоновом режиме
docker-compose up -d --build

# Остановка
docker-compose down
```

### Использование Docker

```bash
# Сборка образа
docker build -t chess-predictor .

# Запуск контейнера
docker run -p 8000:8000 -v $(pwd)/notebook/best_chess_model.pt:/app/notebook/best_chess_model.pt:ro chess-predictor
```

### Локальный запуск (для разработки)

```bash
# Установка зависимостей
pip install -r requirements.txt

# Запуск
python run_local.py
```

## API Документация

После запуска приложения документация доступна по адресу:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI схема: http://localhost:8000/openapi.json

## Примеры использования

### POST запрос с FEN позицией

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
       "top_k": 5
     }'
```

### Проверка здоровья сервиса

```bash
curl http://localhost:8000/health
```

## Структура ответа

```json
{
  "position": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  "top_moves": [
    {
      "move": "e2e4",
      "probability": 0.1234,
      "rank": 1
    },
    {
      "move": "d2d4",
      "probability": 0.0987,
      "rank": 2
    }
  ],
  "legal_moves_count": 20
}
```

## Конфигурация

### Переменные окружения

- `PYTHONUNBUFFERED=1` - для корректного логирования в Docker

### Порты

- `8000` - порт API сервера

### Модель

Модель должна находиться в файле `notebook/best_chess_model.pt`. Если файл отсутствует, приложение запустится с предупреждением и будет использовать случайные веса.

## Разработка

### Структура проекта

```
├── src/
│   ├── main.py           # FastAPI приложение
│   ├── mymodel.py        # Архитектура нейронной сети
│   ├── training.py       # Функции обучения и предсказания
│   ├── features.py       # Извлечение признаков из позиции
│   └── move.py          # Генерация возможных ходов
├── notebook/
│   └── best_chess_model.pt  # Обученная модель
├── requirements.txt      # Python зависимости
├── Dockerfile           # Конфигурация Docker
├── docker-compose.yml   # Docker Compose конфигурация
├── .dockerignore        # Исключения для Docker
└── run_local.py         # Скрипт локального запуска
```

### Добавление новой функциональности

1. Измените код в `src/`
2. Обновите `requirements.txt` при необходимости
3. Пересоберите Docker образ: `docker-compose build --no-cache`

## Мониторинг

- Health check: `GET /health`
- Логи: `docker-compose logs -f chess-predictor`

## Производительность

- Автоматическое определение GPU/CPU
- Оптимизация для пакетной обработки
- Поддержка многопоточности


