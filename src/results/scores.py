import numpy as np

# INFO about the calculation:
# https://www.researchgate.net/publication/303482552_SWIFT-Review_A_text-mining_workbench_for_systematic_review
def calculate_wss(y_true, y_pred, y_scores, recall_target=95):
    # Convert recall target to decimal
    R = recall_target / 100
    
    # Total dataset size N
    N = len(y_true)
    
    # Calculate TP (true positives) at current state
    TP = np.sum((y_true == 1) & (y_pred == 1))
    
    # Calculate total actual positives (TP + FN)
    total_positives = np.sum(y_true == 1)
    
    # Sort by prediction scores
    sorted_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[sorted_indices]
    
    # Find threshold that achieves desired recall
    # R = TP/(TP+FN), so we need TP = R * (TP+FN)
    target_TP = int(np.ceil(R * total_positives))
    cumulative_TP = np.cumsum(y_true_sorted == 1)
    threshold_idx = np.where(cumulative_TP >= target_TP)[0][0]
    
    # Create binary predictions at this threshold
    y_pred_at_threshold = np.zeros_like(y_true)
    y_pred_at_threshold[sorted_indices[:threshold_idx + 1]] = 1
    
    # Calculate confusion matrix elements
    TP = np.sum((y_true == 1) & (y_pred_at_threshold == 1))
    FP = np.sum((y_true == 0) & (y_pred_at_threshold == 1))
    TN = np.sum((y_true == 0) & (y_pred_at_threshold == 0))
    FN = np.sum((y_true == 1) & (y_pred_at_threshold == 0))
    
    # Calculate WSS using the formula from the text
    WSS = (TN + FN)/N - (1.0 - R)
    
    return WSS

# This is a really simple metric, simply sort the docs by ranking
# and check the number of relevant docs in the top 10%
def calculate_rrf(y_true, y_scores, percentage=10):
    # Calculate number of documents to screen
    N = len(y_true)
    n_to_screen = int(np.ceil(N * percentage / 100))
    
    # Sort by prediction scores
    sorted_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[sorted_indices]
    
    # Count relevant documents in the screened portion
    RRF = np.sum(y_true_sorted[:n_to_screen] == 1)
    
    return RRF