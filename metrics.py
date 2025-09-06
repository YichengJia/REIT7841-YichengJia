# metrics.py
# Lightweight metrics for intent classification and explanation generation.

from typing import List, Sequence, Tuple
import math
from collections import Counter, defaultdict

def macro_f1(preds: Sequence[int], golds: Sequence[int]) -> float:
    """
    Macro-F1 for intent classification.
    preds/golds: list of integer class ids (same label space).
    """
    assert len(preds) == len(golds)
    labels = set(golds) | set(preds)
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    for p, g in zip(preds, golds):
        if p == g:
            tp[g] += 1
        else:
            fp[p] += 1
            fn[g] += 1
    f1s = []
    for c in labels:
        precision = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        recall    = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        if precision + recall == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * precision * recall / (precision + recall))
    return sum(f1s)/len(f1s) if f1s else 0.0

def _ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))

def corpus_bleu(references: List[List[str]], hypotheses: List[List[str]], max_n: int = 4) -> float:
    """
    BLEU-n (default BLEU-4) without external deps.
    references: list of reference token lists
    hypotheses: list of hypothesis token lists (same length as refs)
    """
    assert len(references) == len(hypotheses)
    weights = [1.0/max_n]*max_n

    # Modified precision
    p_ns = []
    for n in range(1, max_n+1):
        clipped_count = 0
        total_count = 0
        for ref, hyp in zip(references, hypotheses):
            ref_ngrams = _ngrams(ref, n)
            hyp_ngrams = _ngrams(hyp, n)
            total_count += sum(hyp_ngrams.values())
            for ng, cnt in hyp_ngrams.items():
                clipped_count += min(cnt, ref_ngrams.get(ng, 0))
        p_ns.append((clipped_count / total_count) if total_count > 0 else 0.0)

    # Brevity penalty
    ref_len = sum(len(r) for r in references)
    hyp_len = sum(len(h) for h in hypotheses)
    if hyp_len == 0:
        return 0.0
    if hyp_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len / hyp_len) if hyp_len > 0 else 0.0

    # Geometric mean
    s = 0.0
    for w, p in zip(weights, p_ns):
        s += w * (math.log(p) if p > 0 else -9999.0)  # avoid -inf
    bleu = bp * math.exp(s)
    return max(0.0, min(1.0, bleu))
