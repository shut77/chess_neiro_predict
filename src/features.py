import numpy as np
import chess

def fen_one_hot(fen):
    board = chess.Board(fen)
    #[P,N,B,R,Q,K,p,n,b,r,q,k]
    one_hot = np.zeros((8,8,12), dtype=np.float32)
    dicts = { 'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11}
    for sq in chess.SQUARES:
        figure = board.piece_at(sq)
        if figure:
            col = sq % 8
            row = 7-sq //8
            chanel = dicts[figure.symbol()]
            one_hot[row,col,chanel] = 1.0
    return one_hot

def get_features(board):
    features = np.zeros(17, dtype=np.float32)

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

    return features



