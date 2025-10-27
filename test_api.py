#!/usr/bin/env python3
"""
тест для проверки работоспособности Chess Move Predictor API
"""

import requests
import time

def test_api_health():
    """Тест проверки здоровья API"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("Health check passed")
            return response.json()
        else:
            print(f"Health check failed: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Health check error: {e}")
        return None

def test_api_predict():
    """Тест предсказания ходов"""
    # Стандартная начальная позиция
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    try:
        response = requests.post(
            "http://localhost:8000/predict",
            json={"fen": fen, "top_k": 3},
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            print("Prediction test passed")
            print(f"Position: {data['position']}")
            print(f"Legal moves: {data['legal_moves_count']}")
            print("Top moves:")
            for move in data['top_moves']:
                print(f"  {move['rank']}. {move['move']} ({move['probability']:.4f})")
            return data
        else:
            print(f"Prediction test failed: {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Prediction test error: {e}")
        return None


def test_invalid_fen():
    """Тест с некорректным FEN"""
    try:
        response = requests.post(
            "http://localhost:8000/predict",
            json={"fen": "invalid_fen", "top_k": 3},
            timeout=10
        )

        if response.status_code == 400:
            print("Invalid FEN test passed (correctly rejected)")
            return True
        else:
            print(f"Invalid FEN test failed: expected 400, got {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Invalid FEN test error: {e}")
        return False

def main():
    """Основная функция тестирования"""
    print("Testing Chess Move Predictor API")
    print("=" * 40)

    # Даем время API запуститься
    print("Waiting for API to start...")
    time.sleep(5)

    # Тесты
    tests = [
        ("Health Check", test_api_health),
        ("Move Prediction", test_api_predict),
        ("Invalid FEN", test_invalid_fen),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"Running: {test_name}")
        result = test_func()
        results.append((test_name, result is not None))
        print("")

    # Итоги
    print("=" * 40)
    print("Test Results:")
    passed = sum(1 for _, result in results if result)

    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"  {test_name}: {status}")

    print(f"Overall: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("All tests passed!")
        return True
    else:
        print("Some tests failed")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)