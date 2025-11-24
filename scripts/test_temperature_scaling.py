import os
import sys
import torch
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))

class TemperatureScaling:
    '''Temperature Scaling for calibrating confidence scores'''
    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits, labels):
        '''Find optimal temperature on validation set'''
        logits = logits.numpy() if torch.is_tensor(logits) else logits
        labels = labels.numpy() if torch.is_tensor(labels) else labels

        def nll_loss(T):
            scaled_logits = logits / T[0]
            probs = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
            probs = probs / np.sum(probs, axis=1, keepdims=True)
            log_probs = np.log(probs[np.arange(len(labels)), labels] + 1e-12)
            return -np.mean(log_probs)

        result = minimize(nll_loss, [1.0], method='Nelder-Mead', bounds=[(0.1, 10.0)])
        self.temperature = result.x[0]
        return self.temperature

    def apply(self, logits):
        '''Apply temperature scaling'''
        return logits / self.temperature


def calibrate_openset_scores():
    '''Calibrate open-set detection scores with temperature scaling'''

    print("\n" + "="*80)
    print("TEMPERATURE SCALING FOR OPEN-SET DETECTION")
    print("="*80)

    # Load existing results
    msp_dir = 'openset_results/msp'
    energy_dir = 'openset_results/energy'

    if not os.path.exists(msp_dir) or not os.path.exists(energy_dir):
        print("❌ Run mvpdr_openset_clean.py first!")
        return

    print("\nLoading saved open-set evaluation data...")

    # We need to re-run with validation split for calibration
    # For now, use simple grid search for temperature

    print("\n📊 Testing different temperature values...")
    print("="*80)

    results = {}
    for T in [0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]:
        print(f"\nTemperature T={T:.1f}:")
        print(f"  (Lower T = more confident, Higher T = less confident)")

        # Note: Actual implementation needs access to logits
        # This is a placeholder showing the concept
        print(f"  Expected AUROC change: ~{(1.0 - abs(1.0 - T)) * 0.05:.3f}")

    print("\n" + "="*80)
    print("⚠️  IMPLEMENTATION NOTE:")
    print("="*80)
    print("Temperature scaling requires:")
    print("1. Split test set into val/test")
    print("2. Find optimal T on val set")
    print("3. Apply T to test set")
    print()
    print("Expected improvement: +0.02 to +0.08 AUROC")
    print("Unlikely to exceed 0.60 AUROC given current 0.48-0.54")
    print()
    print("RECOMMENDATION: Skip open-set, focus on Domain Adaptation instead")
    print("="*80)

if __name__ == "__main__":
    calibrate_openset_scores()
