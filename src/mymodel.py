import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import chess
from features import fen_one_hot, get_features

def neiro_input(fen, move_uci, all_moves, move_to_index):
    ALL_MOVES, MOVE_TO_INDEX = all_moves, move_to_index 
    board = chess.Board(fen)
    fen_in_ohe = fen_one_hot(fen)
    features = get_features(board)

    board_tensor = torch.tensor(fen_in_ohe, dtype=torch.float32)
    board_tensor = board_tensor.permute(2, 0, 1)
    add_features = torch.tensor(features, dtype=torch.float32)
    

    # индекс хода в ALL_MOVES
    target_move_index = MOVE_TO_INDEX.get(move_uci)
    if target_move_index is None:
        raise ValueError(f"Move {move_uci} not found, FEN: {fen}")
    

    # маска для легальных ходов [len(ALL_MOVES)]
    legal_moves_mask = torch.zeros(len(ALL_MOVES), dtype=torch.bool)
    for legal_move in board.legal_moves:
        move_uci_legal = legal_move.uci()
        idx = MOVE_TO_INDEX.get(move_uci_legal)
        if idx is not None:
            legal_moves_mask[idx] = True
    
    return {
        'board': board_tensor,           # [12, 8, 8]
        'additional': add_features, # [17]
        'target_move': target_move_index, # индекс в ALL_MOVES
        'legal_moves_mask': legal_moves_mask  # [len(ALL_MOVES)]
    }

class ChessDataset(Dataset):
    def __init__(self, df, all_moves, move_to_index):
        self.df = df
        self.all_moves = all_moves
        self.move_to_index = move_to_index

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fen = row['fen']
        move_uci = row['move']

        return neiro_input(fen, move_uci, self.all_moves, self.move_to_index)

class SEBlock1(nn.Module):
    def __init__(self, channels, reduction=32):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid())
    
    def forward(self, x):
        b, c, _, _ = x.shape
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=32):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.SiLU(),  # Swish активация
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        scale = self.fc(x).view(x.size(0), x.size(1), 1, 1)
        return x * scale
    

class ChessResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels) #nn. GroupNorm(4, channels)
        self.bn2 = nn.BatchNorm2d(channels) #nn. GroupNorm(4, channels)
        self.se = SEBlock(channels)
        
    def forward(self, x):
        residual = x
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x))) 
        x = self.se(x)  
        x += residual
        return F.gelu(x)


class ChessMovePredictor(nn.Module):
    def __init__(self, num_moves=4672, num_residual_blocks=10, channels=256):
        super().__init__()
        self.num_moves = num_moves
        
        self.conv_input = nn.Conv2d(15, channels, 5, padding=2)
        self.bn_input = nn.BatchNorm2d(channels) #nn. GroupNorm(4, channels)
        
        
        self.res_blocks = nn.Sequential(
            *[ChessResidualBlock(channels) for _ in range(num_residual_blocks)])

        self.dilated_conv = nn.Conv2d(channels, channels, 3, padding=2, dilation=2)
        self.dilated_norm = nn.BatchNorm2d(channels) #nn. GroupNorm(4, channels)
        
        self.se = SEBlock(channels)
        self.input_se = SEBlock(channels) 
        
        self.spatial_reduce = nn.Sequential(
            nn.Conv2d(channels, channels, 3, stride=2, padding=1),  # 8x8 -> 4x4
            nn.GELU())

        self.add_fc = nn.Sequential(
            nn.Linear(23, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            
        )
        spatial_features = channels * 4 * 4 
        self.combined_fc = nn.Sequential(
            nn.Linear(spatial_features + 256, 1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_moves)  
        )
    
    def forward(self, board_tensor, additional_features, legal_moves_mask):
        """
        Args:
            board_tensor: [batch, 15, 8, 8]
            additional_features: [batch, 23]
            legal_moves_mask: [batch, num_moves] - маска легальных ходов
        
        Returns:
            move_probs: [batch, num_moves] - вероятности ходов
            move_logits: [batch, num_moves] - логиты (для loss)
        """
        batch_size = board_tensor.size(0)
        
        
        x = F.gelu(self.bn_input(self.conv_input(board_tensor)))  # [batch, 256, 8, 8]
        x = self.input_se(x)
        x = self.res_blocks(x)  # [batch, 256, 8, 8]
        x = self.dilated_conv(x)
        x = self.dilated_norm(x)
        x = F.gelu(x) 
        x = self.se(x)
        
        x = self.spatial_reduce(x) 

        x = x.view(batch_size, -1)  # [batch, 256]
        add_features = self.add_fc(additional_features)  # [batch, 128]
        combined = torch.cat([x, add_features], dim=1)  # [batch, 384]
        
        # предикт всех ходов
        move_logits = self.combined_fc(combined)  # [batch, num_moves]
        
        # маска легал ходов
        move_logits = move_logits.masked_fill(~legal_moves_mask, -1e9) # -1e9
        
        # Softmax по легальным ходам
        move_probs = F.softmax(move_logits, dim=-1)
        
        return move_probs, move_logits
    

class ChessMoveLoss(nn.Module):
    #c учетом маски легальных ходов
    def forward(self, move_logits, target_indices, legal_mask):
        masked_logits = move_logits.masked_fill(~legal_mask, -1e9)
        log_probs = F.log_softmax(masked_logits, dim=-1)
        loss = F.nll_loss(log_probs, target_indices)
        return loss