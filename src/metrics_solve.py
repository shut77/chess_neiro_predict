import numpy as np, pandas as pd, torch
import tqdm as tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score

def top_k_precision(y_true, y_probs, k=3):
    top_k_preds = np.argsort(y_probs, axis=1)[:, -k:]
    
    precision_scores = []
    for i in range(len(y_true)):
        correct_in_top = np.sum(top_k_preds[i] == y_true[i])
        precision_scores.append(correct_in_top / k)
    
    return np.mean(precision_scores)

def mean_reciprocal_rank(y_true, y_probs):
    ranks = []
    for i in range(len(y_true)):
        # Ранжируем предсказания по убыванию вероятности
        sorted_indices = np.argsort(y_probs[i])[::-1]
        rank = np.where(sorted_indices == y_true[i])[0][0] + 1
        ranks.append(1.0 / rank)
    
    return np.mean(ranks)

def top_k_recall(y_true, y_probs, k=3):
    top_k_preds = np.argsort(y_probs, axis=1)[:, -k:]
    correct = []
    for i in range(len(y_true)):
        if y_true[i] in top_k_preds[i]:
            correct.append(1)  
        else:
            correct.append(0)  
    return np.mean(correct)


@torch.no_grad()
def for_metrics(model, test_dataset, criterion):
    all_pred = []
    all_targets = []
    all_prob = []
    total_loss = 0
       
    model.eval()
    
    for batch in tqdm.tqdm(test_dataset, desc="Test"):
        board = batch['board']
        add = batch['additional']
        target = batch['target_move']
        legal_mask = batch['legal_moves_mask']

        move_probs, move_logits = model(board, add, legal_mask)

        loss = criterion(move_logits, target, legal_mask)
        
        preds = move_probs.argmax(dim=1)

        all_pred.append(preds.cpu().numpy())
        all_targets.append(target.cpu().numpy())
        all_prob.append(move_probs.cpu().numpy())
        total_loss += loss.item()
    
    all_prob = np.concatenate(all_prob)
    all_pred = np.concatenate(all_pred)
    all_targets = np.concatenate(all_targets)

    def metrics_solve(y_true, y_pred, y_probs):
        metrics = {}
        metrics["precision"] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics["f1"] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics["accuracy"] = (y_true == y_pred).mean()
        
        metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
        metrics["cohens_kappa"] = cohen_kappa_score(y_true, y_pred)

        actual_classes = np.unique(y_true)
        print(f"Классы в y_true: {actual_classes}")
        print(f"y_probs: {y_probs.shape}")
        for k in [1, 3, 5, 10]:
            metrics[f"recall_top{k}"] = top_k_recall(y_true, y_probs, k=k)
            
        for k in [3, 5, 10]:
            metrics[f"precision_top{k}"] = top_k_precision(y_true, y_probs, k=k)

        metrics["mrr"] = mean_reciprocal_rank(y_true, y_probs)
        return metrics

    metrics = metrics_solve(all_targets, all_pred, all_prob)

    return metrics