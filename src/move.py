import chess

def generate_all_possible_moves():
    # Генерируем все возможные шахматные ходы (включая невозможные для проверки полноты) Должно быть около ~4672 ходов
    all_moves = []
    
    for from_sq in range(64):
        fx = chess.square_file(from_sq)
        fy = chess.square_rank(from_sq)
        
        # Все возможные целевые клетки (для всех фигур кроме пешек)
        # Это покрывает ферзя, ладью, слона, коня, короля
        for to_sq in range(64):
            if from_sq != to_sq:
                # Обычный ход (без превращения)
                all_moves.append(chess.Move(from_sq, to_sq))
        
        # Пешечные превращения
        # Белые пешки: с 7-й горизонтали (rank 6) на 8-ю (rank 7)
        if fy == 6:  # 7-я горизонталь (0-indexed)
            # Ход вперед с превращением
            to_sq = chess.square(fx, 7)
            for promotion in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                all_moves.append(chess.Move(from_sq, to_sq, promotion=promotion))
            
            # Взятие по диагонали с превращением
            for dx in [-1, 1]:
                tx = fx + dx
                if 0 <= tx < 8:
                    to_sq = chess.square(tx, 7)
                    for promotion in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                        all_moves.append(chess.Move(from_sq, to_sq, promotion=promotion))
        
        # Черные пешки: со 2-й горизонтали (rank 1) на 1-ю (rank 0)
        if fy == 1:  # 2-я горизонталь
            # Ход вперед с превращением
            to_sq = chess.square(fx, 0)
            for promotion in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                all_moves.append(chess.Move(from_sq, to_sq, promotion=promotion))
            
            # Взятие по диагонали с превращением
            for dx in [-1, 1]:
                tx = fx + dx
                if 0 <= tx < 8:
                    to_sq = chess.square(tx, 0)
                    for promotion in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                        all_moves.append(chess.Move(from_sq, to_sq, promotion=promotion))
    
    unique_moves = []
    seen = set()
    for move in all_moves:
        uci = move.uci()
        if uci not in seen:
            seen.add(uci)
            unique_moves.append(move)
    
    return unique_moves