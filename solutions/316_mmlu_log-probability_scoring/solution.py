import numpy as np

def mmlu_log_prob_score(log_probs: list, correct_answers: list) -> dict:
    """
    Compute MMLU-style log-probability scoring metrics.
    
    Args:
        log_probs: List of lists, where each inner list contains 
                   log-probabilities for each answer choice
        correct_answers: List of correct answer indices (0-indexed)
    
    Returns:
        Dictionary with 'accuracy', 'predictions', and 'avg_correct_prob'
    """
    predictions, accuracy, probs = [], 0, []
    for iQuestion in range(len(log_probs)):
        answers = log_probs[iQuestion]
        index_predicted = np.argmax(answers)
        index_correct = correct_answers[iQuestion]

        predictions.append(index_predicted)
        accuracy += int(index_predicted == index_correct)
        log_p_max = answers[index_correct]
        p = np.exp(log_p_max)/sum(np.exp(answers))
        probs.append(p)
    
    accuracy /= len(log_probs) 
    avg_correct_prob = sum(probs)/ len(log_probs) 

    return {'accuracy': round(accuracy, 4), 
    'predictions': predictions, 
    'avg_correct_prob': avg_correct_prob}
    