import torch
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc

class OpenSetDetector:
    def __init__(self, method='msp'):
        self.method = method

    def compute_scores(self, logits):
        if self.method == 'msp':
            # Maximum Softmax Probability
            probs = torch.softmax(logits, dim=1)
            scores = probs.max(dim=1)[0]
        elif self.method == 'energy':
            # Energy score
            scores = torch.logsumexp(logits, dim=1)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return scores.numpy() if torch.is_tensor(scores) else scores

    def evaluate(self, known_logits, unknown_logits):
        # Compute scores
        known_scores = self.compute_scores(known_logits)
        unknown_scores = self.compute_scores(unknown_logits)

        # Create labels (1=known, 0=unknown)
        labels = np.concatenate([
            np.ones(len(known_scores)),
            np.zeros(len(unknown_scores))
        ])
        scores = np.concatenate([known_scores, unknown_scores])

        # Compute metrics
        auroc = roc_auc_score(labels, scores)
        fpr, tpr, thresholds = roc_curve(labels, scores)

        # Find FPR@95% TPR
        idx_95tpr = np.argmax(tpr >= 0.95)
        fpr_at_95tpr = fpr[idx_95tpr]

        # Compute detection accuracy at optimal threshold
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        predictions = (scores >= optimal_threshold).astype(int)
        detection_acc = (predictions == labels).mean()

        # Precision-Recall
        precision, recall, _ = precision_recall_curve(labels, scores)
        aupr = auc(recall, precision)

        return {
            'auroc': auroc,
            'aupr': aupr,
            'fpr_at_95tpr': fpr_at_95tpr,
            'optimal_threshold': optimal_threshold,
            'detection_accuracy': detection_acc,
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'method': self.method
        }

    def save_results(self, results, output_dir):
        import os
        os.makedirs(output_dir, exist_ok=True)

        np.save(os.path.join(output_dir, 'openset_fpr.npy'), results['fpr'])
        np.save(os.path.join(output_dir, 'openset_tpr.npy'), results['tpr'])
        np.save(os.path.join(output_dir, 'openset_thresholds.npy'), results['thresholds'])

        # Save scores and labels
        import json
        metrics = {
            'method': results['method'],
            'auroc': float(results['auroc']),
            'aupr': float(results['aupr']),
            'fpr_at_95tpr': float(results['fpr_at_95tpr']),
            'optimal_threshold': float(results['optimal_threshold']),
            'detection_accuracy': float(results['detection_accuracy'])
        }

        with open(os.path.join(output_dir, 'openset_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
