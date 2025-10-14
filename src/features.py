import numpy as np
import chess

def fen_one_hot(fen):
    board = chess.Board(fen)
    #[P,N,B,R,Q,K,p,n,b,r,q,k]
    one_hot = np.zeros((8,8,15), dtype=np.float32)
    dicts = { 'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11}
    white_attacks = set()
    black_attacks = set()
    
    for square in chess.SQUARES:
        if board.is_attacked_by(chess.WHITE, square):
            white_attacks.add(square)
        if board.is_attacked_by(chess.BLACK, square):
            black_attacks.add(square)

    for sq in chess.SQUARES:
        figure = board.piece_at(sq)
        if figure:
            col = sq % 8
            row = 7-sq //8
            if figure:
                chanel = dicts[figure.symbol()]
                one_hot[row, col, chanel] = 1.0
            else:
                one_hot[row, col, 12] = 1.0

            if sq in white_attacks:
                one_hot[row, col, 13] = 1.0
            if sq in black_attacks:
                one_hot[row, col, 14] = 1.0
    return one_hot

def get_features(board):
    features = np.zeros(23, dtype=np.float32)

    features[0] = board.has_kingside_castling_rights(chess.WHITE)
    features[1] = board.has_queenside_castling_rights(chess.WHITE)
    features[2] = board.has_kingside_castling_rights(chess.BLACK)
    features[3] = board.has_queenside_castling_rights(chess.BLACK)

    if board.ep_square:
        features[4] = (chess.square_file(board.ep_square) + 1) / 10.0
    else:
         features[4] = 0.0

    features[5] = 1.0 if board.turn == chess.WHITE else 0.0
    features[6] = board.halfmove_clock / 50.0

    features[7] = len(board.pieces(chess.QUEEN, chess.WHITE))
    features[8] = len(board.pieces(chess.ROOK, chess.WHITE)) / 2.0
    features[9] = len(board.pieces(chess.BISHOP, chess.WHITE)) / 2.0
    features[10] = len(board.pieces(chess.KNIGHT, chess.WHITE)) / 2.0
    features[11] = len(board.pieces(chess.PAWN, chess.WHITE)) / 8.0

    features[12] = len(board.pieces(chess.QUEEN, chess.BLACK))
    features[13] = len(board.pieces(chess.ROOK, chess.BLACK)) / 2.0
    features[14] = len(board.pieces(chess.BISHOP, chess.BLACK)) / 2.0
    features[15] = len(board.pieces(chess.KNIGHT, chess.BLACK)) / 2.0
    features[16] = len(board.pieces(chess.PAWN, chess.BLACK)) / 8.0

    empty_squares = 64 - len(board.piece_map())
    features[17] = empty_squares / 64.0


    center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
    center_control = 0
    for square in center_squares:
        if board.piece_at(square):
            piece = board.piece_at(square)
            if piece.color == chess.WHITE:
                center_control += 1
            else:
                center_control -= 1
    features[18] = center_control / 4.0

    # Безопасность королей (расстояние до края доски)
    white_king = board.king(chess.WHITE)
    black_king = board.king(chess.BLACK)
    
    if white_king:
        white_king_file = chess.square_file(white_king)
        white_king_rank = chess.square_rank(white_king)
        # Король ближе к углу = безопаснее
        white_king_safety = min(white_king_file, 7-white_king_file, white_king_rank, 7-white_king_rank) / 3.5
        features[19] = white_king_safety
    else:
        features[19] = 0.0

    if black_king:
        black_king_file = chess.square_file(black_king)
        black_king_rank = chess.square_rank(black_king)
        black_king_safety = min(black_king_file, 7-black_king_file, black_king_rank, 7-black_king_rank) / 3.5
        features[20] = black_king_safety
    else:
        features[20] = 0.0


    white_king_attack = 0
    black_king_attack = 0
    
    if black_king:
        attackers = board.attackers(chess.WHITE, black_king)
        white_king_attack = len(attackers) / 4.0
    
    if white_king:
        attackers = board.attackers(chess.BLACK, white_king)
        black_king_attack = len(attackers) / 4.0
    
    features[21] = white_king_attack
    features[22] = black_king_attack

    return features



