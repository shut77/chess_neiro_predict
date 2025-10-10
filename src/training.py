from tqdm import tqdm
import torch
import chess
from features import fen_one_hot, get_features

def train_epoch(model, dataloader, optimizer, criterion, device, scheduler):
    model.train()
    total_loss, total_acc = 0, 0

    for batch in tqdm(dataloader, desc="Training"):
        board = batch['board'].to(device)
        additional = batch['additional'].to(device)
        target = batch['target_move'].to(device)
        legal_mask = batch['legal_moves_mask'].to(device)

        move_probs, move_logits = model(board, additional, legal_mask)
        loss = criterion(move_logits, target, legal_mask)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # accuracy
        preds = move_logits.argmax(dim=1)
        acc = (preds == target).float().mean()

        total_loss += loss.item()
        total_acc += acc.item()

    return total_loss / len(dataloader), total_acc / len(dataloader)


@torch.no_grad()
def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_acc1 = 0
    total_acc3 = 0
    total_acc5 = 0

    for batch in tqdm(dataloader, desc="Validation"):
        board = batch['board'].to(device)
        additional = batch['additional'].to(device)
        target = batch['target_move'].to(device)
        legal_mask = batch['legal_moves_mask'].to(device)

        move_probs, move_logits = model(board, additional, legal_mask)
        loss = criterion(move_logits, target, legal_mask)

        preds_top1 = move_logits.argmax(dim=1)
        acc1 = (preds_top1 == target).float().mean()
        
        # Top-3 accuracy
        top3 = move_logits.topk(3, dim=1).indices
        acc3 = (top3 == target.unsqueeze(1)).any(dim=1).float().mean()
        
        # Top-5 accuracy
        top5 = move_logits.topk(5, dim=1).indices
        acc5 = (top5 == target.unsqueeze(1)).any(dim=1).float().mean()

        total_loss += loss.item()
        total_acc1 += acc1.item()
        total_acc3 += acc3.item()
        total_acc5 += acc5.item()

    num_batches = len(dataloader)
    metrics = {
        "loss": total_loss / num_batches,
        "acc1": total_acc1 / num_batches,
        "acc3": total_acc3 / num_batches,
        "acc5": total_acc5 / num_batches}
    
    return metrics


def predict_move(model, fen_string, device, ALL_MOVES, MOVE_TO_INDEX, INDEX_TO_MOVE, top_k=5):
    model.eval()
    
    with torch.no_grad():
        
        board = chess.Board(fen_string)
        
        board_tensor = torch.tensor(fen_one_hot(fen_string), dtype=torch.float32)
        board_tensor = board_tensor.permute(2, 0, 1).unsqueeze(0).to(device)  # [1, 12, 8, 8]
        
        add_features = torch.tensor(get_features(board), dtype=torch.float32)
        add_features = add_features.unsqueeze(0).to(device)  # [1, 17]
        
        # все легвльные ходы
        legal_moves_mask = torch.zeros(1, len(ALL_MOVES), dtype=torch.bool)
        legal_moves_list = []
        for legal_move in board.legal_moves:
            idx = MOVE_TO_INDEX.get(legal_move.uci())
            if idx is not None:
                legal_moves_mask[0, idx] = True
                legal_moves_list.append(legal_move)
        legal_moves_mask = legal_moves_mask.to(device)
        
        
        #move_probs, _ = model(board_tensor, add_features, legal_moves_mask)
        #best_move_idx = move_probs[0].argmax().item()
        #best_move_uci = INDEX_TO_MOVE[best_move_idx].uci()
        #best_move_prob = move_probs[0, best_move_idx].item()

        move_probs, _ = model(board_tensor, add_features, legal_moves_mask)
        topk_prob, topk_indices = torch.topk(move_probs[0], k=top_k)
        
        top_moves = []
        for i, (prob, idx) in enumerate(zip(topk_prob, topk_indices)):
            move_idx  = INDEX_TO_MOVE[idx.item()]
            move_uci = move_idx.uci()
            top_moves.append({'топ': i + 1,'ход': move_uci,'вероятность': prob.item()})
        
        return top_moves #best_move_uci, best_move_prob, move_probs[0], 