from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import chess
import os
from typing import List, Optional

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mymodel import ChessMovePredictor
from features import fen_one_hot, get_features
from move import generate_all_possible_moves
from training import predict_move

class ChessPosition(BaseModel):
    fen: str
    top_k: Optional[int] = 5

class MovePrediction(BaseModel):
    move: str
    probability: float
    rank: int

class PredictionResponse(BaseModel):
    position: str
    top_moves: List[MovePrediction]
    legal_moves_count: int

app = FastAPI(
    title="Chess Move Predictor API",
    description="API для предсказания шахматных ходов с использованием нейронной сети",
    version="1.0.0"
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])

model = None
device = None
ALL_MOVES = None
MOVE_TO_INDEX = None
INDEX_TO_MOVE = None

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "notebook", "best_chess_model.pt")

def load_model():
    """Загрузка модели и вспомогательных структур"""
    global model, device, ALL_MOVES, MOVE_TO_INDEX, INDEX_TO_MOVE

    try:

        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device('cpu')
        print(f"Используемое устройство: {device}")

        ALL_MOVES = generate_all_possible_moves()
        MOVE_TO_INDEX = {m.uci(): i for i, m in enumerate(ALL_MOVES)}
        INDEX_TO_MOVE = {i: m for i, m in enumerate(ALL_MOVES)}

        print(f"Загружено {len(ALL_MOVES)} возможных ходов")

        num_moves = len(ALL_MOVES)
        model = ChessMovePredictor(num_moves=num_moves, num_residual_blocks=10, channels=256)
        model.to(device)

        # Загрузка весов модели
        if os.path.exists(MODEL_PATH):
            state_dict = torch.load(MODEL_PATH, map_location=device)
            model.load_state_dict(state_dict)
            print("Модель успешно загружена")
        else:
            print(f"Предупреждение: файл модели {MODEL_PATH} не найден")
            print("Модель будет работать со случайными весами")

        model.eval()

    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        raise

def validate_fen(fen: str) -> bool:
    try:
        chess.Board(fen)
        return True
    except:
        return False

@app.on_event("startup")
async def startup_event():
    """Загрузка модели при старте приложения"""
    load_model()

@app.get("/")
async def root():
    """Корневой эндпоинт"""
    return {
        "message": "Chess Move Predictor API",
        "version": "1.0.0",
        "device": str(device),
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_moves(position: ChessPosition):
    """
    Предсказание топ-K лучших ходов для заданной позиции
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Модель не загружена")

    if not validate_fen(position.fen):
        raise HTTPException(status_code=400, detail="Некорректная FEN строка")

    try:
        top_moves = predict_move(
            model, position.fen, device,
            ALL_MOVES, MOVE_TO_INDEX, INDEX_TO_MOVE,
            top_k=position.top_k
        )

        board = chess.Board(position.fen)
        legal_moves_count = len(list(board.legal_moves))

        formatted_moves = [
            MovePrediction(
                move=move_data['ход'],
                probability=move_data['вероятность'],
                rank=move_data['топ']
            )
            for move_data in top_moves
        ]

        return PredictionResponse(
            position=position.fen,
            top_moves=formatted_moves,
            legal_moves_count=legal_moves_count
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при предсказании: {str(e)}")


@app.get("/health")
async def health_check():
    """Проверка здоровья сервиса"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
        "moves_count": len(ALL_MOVES) if ALL_MOVES else 0
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
