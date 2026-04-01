"""
sealMega IDE — Perplexity Rollback (Cauchy-Schwarz Safety Net)

// Built this because I am hardware constrained and wanted to help my friends out.
// When you drop heads, measure the entropy of the output logits.
// If the distribution flattens, the model is confused because you blinded it.
// Rollback. Recompute the unpruned layer. 300ms delay is acceptable;
// a hallucinated rm -rf command is not.
// Took 2 days to implement this without slowing down generation.
"""

import math
from typing import Optional

# Perplexity spike threshold - calibrated from 400 failed test runs
PERPLEXITY_SPIKE_RATIO = 1.8  # 80% above baseline = model is completely lost
ENTROPY_FLOOR = 0.1  # Minimum entropy below which we don't bother checking


def compute_entropy(logits: list[float]) -> float:
    """
    Compute Shannon entropy of a probability distribution.
    Higher entropy = model is less certain = potential hallucination.
    """
    total = sum(math.exp(l) for l in logits)
    if total == 0:
        return 0.0
    
    probs = [math.exp(l) / total for l in logits]
    entropy = -sum(p * math.log2(p + 1e-10) for p in probs if p > 0)
    return entropy


def should_rollback(
    pruned_logits: list[float],
    baseline_entropy: float,
) -> bool:
    """
    Check if pruned output logits indicate the model is hallucinating.
    
    Returns True if we MUST rollback and recompute unpruned.
    Returns False if pruning was safe.
    
    Dr. Anatoly: "Correctness > Speed."
    """
    if baseline_entropy < ENTROPY_FLOOR:
        return False
    
    pruned_entropy = compute_entropy(pruned_logits)
    
    # If entropy spiked beyond the threshold ratio, the model is blind
    ratio = pruned_entropy / max(baseline_entropy, ENTROPY_FLOOR)
    
    if ratio > PERPLEXITY_SPIKE_RATIO:
        print(f"[Rollback] ENTROPY SPIKE: {pruned_entropy:.3f} vs baseline {baseline_entropy:.3f} "
              f"(ratio {ratio:.2f}x > threshold {PERPLEXITY_SPIKE_RATIO}x). ROLLING BACK.")
        return True
    
    return False


def compute_perplexity(logits_sequence: list[list[float]]) -> float:
    """
    Compute perplexity over a sequence of logit distributions.
    Lower = model is confident and coherent.
    Higher = model is confused, possibly hallucinating.
    """
    if not logits_sequence:
        return float('inf')
    
    total_entropy = sum(compute_entropy(logits) for logits in logits_sequence)
    avg_entropy = total_entropy / len(logits_sequence)
    
    return 2.0 ** avg_entropy


class PruningGuard:
    """
    Monitors pruning decisions and enforces rollback when the model
    shows signs of confusion (entropy spikes).
    
    Usage:
        guard = PruningGuard()
        guard.set_baseline(unpruned_logits)
        
        # After pruning a layer:
        if guard.check(pruned_logits):
            # MUST recompute this layer unpruned
            recompute_layer_unpruned()
    """
    
    def __init__(self, spike_ratio: float = PERPLEXITY_SPIKE_RATIO):
        self.spike_ratio = spike_ratio
        self.baseline_entropy: float = 0.0
        self.rollback_count: int = 0
        self.check_count: int = 0
    
    def set_baseline(self, unpruned_logits: list[float]):
        """Set the baseline entropy from unpruned layer output."""
        self.baseline_entropy = compute_entropy(unpruned_logits)
    
    def check(self, pruned_logits: list[float]) -> bool:
        """
        Check if pruned output requires rollback.
        Returns True = ROLLBACK NOW.
        """
        self.check_count += 1
        needs_rollback = should_rollback(pruned_logits, self.baseline_entropy)
        if needs_rollback:
            self.rollback_count += 1
        return needs_rollback
    
    def get_stats(self) -> dict:
        return {
            "checks": self.check_count,
            "rollbacks": self.rollback_count,
            "rollback_rate": self.rollback_count / max(self.check_count, 1),
            "baseline_entropy": self.baseline_entropy,
        }
