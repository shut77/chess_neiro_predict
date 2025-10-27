# Chess Move Predictor API - Makefile

.PHONY: help install run-docker run-local test clean build stop


help:
	@echo "Доступные команды:"
	@echo "  install     - Установка зависимостей"
	@echo "  run-local   - Запуск локально для разработки"
	@echo "  run-docker  - Запуск с помощью Docker"
	@echo "  build       - Сборка Docker образа"
	@echo "  test        - Запуск тестов API"
	@echo "  clean       - Остановка и очистка"
	@echo "  stop        - Остановка всех контейнеров"


install:
	pip install -r requirements.txt


run-local:
	python run_local.py


build:
	docker-compose build


run-docker:
	docker-compose up

# Запуск в фоне
run-docker-detached:
	docker-compose up -d


test:
	python test_api.py


stop:
	docker-compose down


clean:
	docker-compose down -v
	docker system prune -f

# Полная очистка и пересборка
rebuild: clean build

# Проверка логов
logs:
	docker-compose logs -f chess-predictor


status:
	docker-compose ps

# Установка зависимостей в Docker контейнере
install-docker:
	docker-compose exec chess-predictor pip install -r requirements.txt

# Выполнение команды в контейнере
exec:
	docker-compose exec chess-predictor /bin/bash

# Обновление модели в контейнере
update-model:
	docker-compose cp notebook/best_chess_model.pt chess-predictor:/app/notebook/best_chess_model.pt

# Проверка здоровья API
health:
	curl -f http://localhost:8000/health || echo "API не отвечает"

# Пример использования API
example-post:
	curl -X POST "http://localhost:8000/predict" \
		-H "Content-Type: application/json" \
		-d '{"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "top_k": 3}'

example-get:
	curl "http://localhost:8000/predict/rnbqkbnr%2Fpppppppp%2F8%2F8%2F8%2F8%2FPPPPPPPP%2FRNBQKBNR%20w%20KQkq%20-%200%201?top_k=3"
