#!/usr/bin/env python3
"""
Локальный запуск Chess Move Predictor API для разработки и тестирования
"""

import uvicorn
import os
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        reload_dirs=["src"],
        log_level="info"
    )
