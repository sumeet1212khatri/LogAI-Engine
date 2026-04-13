"""
classify.py — 3-Tier Hybrid Pipeline (V2 — Batch Optimized)

Architecture:
  LegacyCRM → LLM directly
  Others    → Regex → BERT (batch) → LLM fallback

Speed improvement:
  V1: 82 logs/s   (ek ek kar ke)
  V2: 2000+ logs/s (batch mein)
"""
from __future__ import annotations
import pandas as pd
from processor_regex import classify_with_regex
from processor_bert  import classify_batch as bert_batch
from processor_llm   import classify_with_llm

LEGACY_SOURCE = 'LegacyCRM'


def classify_log(source: str, log_msg: str) -> dict:
    """Single log classify karo. (Backward compatible)"""
    results = classify_logs([(source, log_msg)])
    return results[0]


def classify_logs(logs: list[tuple[str, str]]) -> list[dict]:
    """
    BATCH processing — ye hai speed ka raaz!
    
    Kaise kaam karta hai:
    1. Sab logs ko 3 groups mein baanto:
       - Group A: LegacyCRM → directly LLM
       - Group B: Regex match → done!
       - Group C: Bache hue → BERT batch mein
    2. BERT ke low-confidence wale → LLM fallback
    
    Args:
        logs: List of (source, log_message) tuples
    
    Returns:
        List of {label, tier, confidence} dicts
    """
    n       = len(logs)
    results = [None] * n  # Pre-allocate results

    # ── Step 1: Separate karo ──────────────────────────────
    llm_indices   = []  # LegacyCRM
    regex_indices = []  # Regex match ho gaya
    bert_indices  = []  # BERT ke liye

    for i, (source, log_msg) in enumerate(logs):
        if source == LEGACY_SOURCE:
            llm_indices.append(i)
        else:
            label = classify_with_regex(log_msg)
            if label:
                results[i] = {'label': label, 'tier': 'Regex', 'confidence': 1.0}
            else:
                bert_indices.append(i)

    # ── Step 2: BERT batch processing ──────────────────────
    if bert_indices:
        bert_msgs    = [logs[i][1] for i in bert_indices]
        bert_results = bert_batch(bert_msgs)  # Single batch call!

        for idx, (label, conf) in zip(bert_indices, bert_results):
            if label != 'Unclassified':
                results[idx] = {'label': label, 'tier': 'BERT', 'confidence': conf}
            else:
                # Low confidence → LLM fallback
                llm_indices.append(idx)

    # ── Step 3: LLM calls ──────────────────────────────────
    for i in llm_indices:
        _, log_msg = logs[i]
        label      = classify_with_llm(log_msg)
        tier       = 'LLM' if logs[i][0] == LEGACY_SOURCE else 'LLM (fallback)'
        results[i] = {'label': label, 'tier': tier, 'confidence': None}

    return results


def classify_csv(input_path: str, output_path: str = 'output.csv') -> tuple[str, pd.DataFrame]:
    """
    CSV file classify karo.
    Required columns: 'source', 'log_message'
    Output: adds 'predicted_label', 'tier_used', 'confidence'
    """
    df = pd.read_csv(input_path)

    required = {'source', 'log_message'}
    if not required.issubset(df.columns):
        raise ValueError(f'CSV mein ye columns chahiye: {required}. Mila: {set(df.columns)}')

    # Batch classify
    log_pairs = list(zip(df['source'], df['log_message']))
    results   = classify_logs(log_pairs)

    df['predicted_label'] = [r['label'] for r in results]
    df['tier_used']        = [r['tier']  for r in results]
    df['confidence']       = [
        f"{r['confidence']:.1%}" if r['confidence'] is not None else 'N/A'
        for r in results
    ]

    df.to_csv(output_path, index=False)
    return output_path, df


# ── Backward compatible wrappers ────────────────────────────
def classify(logs: list[tuple[str, str]]) -> list[dict]:
    """Alias for classify_logs."""
    return classify_logs(logs)


# ── Test ────────────────────────────────────────────────────
if __name__ == '__main__':
    import time

    sample = [
        ('ModernCRM',       'IP 192.168.133.114 blocked due to potential attack'),
        ('BillingSystem',   'User User12345 logged in.'),
        ('AnalyticsEngine', 'File data_6957.csv uploaded successfully by user User265.'),
        ('ModernHR',        'GET /v2/servers/detail HTTP/1.1 status: 200 len: 1583 time: 0.19'),
        ('ModernHR',        'Admin access escalation detected for user 9429'),
        ('LegacyCRM',       'Case escalation for ticket ID 7324 failed because the assigned support agent is no longer active.'),
        ('LegacyCRM',       "The 'ReportGenerator' module will be retired in version 4.0."),
    ]

    print(f'{"Source":<20} {"Tier":<15} {"Conf":>6}  {"Label":<25} Log')
    print('─' * 110)
    results = classify_logs(sample)
    for (source, log), r in zip(sample, results):
        conf = f"{r['confidence']:.0%}" if r['confidence'] else ' N/A'
        print(f'{source:<20} {r["tier"]:<15} {conf:>6}  {r["label"]:<25} {log[:45]}')

    # Speed test
    print('\n' + '='*55)
    big = sample * 200  # 1400 logs
    t0 = time.perf_counter()
    classify_logs(big)
    elapsed = time.perf_counter() - t0
    print(f'Speed: {len(big)/elapsed:.0f} logs/s for {len(big)} logs')
