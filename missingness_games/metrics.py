"""Classification endpoints; subject macro accuracy is the primary aggregation."""
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score


def measure(prediction, data):
    probability = np.asarray(prediction['probability'])
    labels = np.asarray(data['y']).astype(int)
    chosen = probability.argmax(1)
    confidence = probability.max(1)
    correct = chosen == labels
    bins = np.minimum((15 * confidence).astype(int), 14)
    ece = sum(np.mean(bins == b) * abs(confidence[bins == b].mean()-correct[bins == b].mean())
              for b in range(15) if np.any(bins == b))
    group_scores = [balanced_accuracy_score(labels[data['group']==group], chosen[data['group']==group])
                    for group in np.unique(data['group'])]
    return dict(accuracy=float(correct.mean()), balanced_accuracy=float(balanced_accuracy_score(labels, chosen)),
        macro_f1=float(f1_score(labels,chosen,average='macro',zero_division=0)),
        nll=float(-np.log(probability[np.arange(len(labels)),labels].clip(1e-10)).mean()),
        brier=float(((probability-np.eye(probability.shape[1])[labels])**2).sum(1).mean()),
        ece=float(ece), subject_macro_accuracy=float(np.mean(group_scores)),
        worst_group_accuracy=float(np.min(group_scores)))
