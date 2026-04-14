"""
classify.py — 3-Tier Hybrid Pipeline (V3 — Latency-Tracked & Parallelized)
Architecture:
  LegacyCRM → LLM directly
  Others    → Regex → BERT (batch) → LLM fallback
Changes in V3:
  - Tier-wise latency tracking (regex_ms, bert_ms, llm_ms)
  - Pipeline summary with p50/p95 per tier
  - Defensive: LLM timeout + circuit breaker baked in via processor_llm
  - Parallelized LLM Tier using ThreadPoolExecutor for high throughput
"""
from __future__ import annotations
import time
import statistics
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from processor_regex import classify_with_regex
from processor_bert  import classify_batch as bert_batch
from processor_llm   import classify_with_llm

LEGACY_SOURCE = "LegacyCRM"


# ── Result type ─────────────────────────────────────────────────────────────
def _make_result(label: str, tier: str, confidence, latency_ms: float) -> dict:
    return {
        "label":      label,
        "tier":       tier,
        "confidence": confidence,
        "latency_ms": round(latency_ms, 3),
    }


# ── Single log (backward-compatible) ────────────────────────────────────────
def classify_log(source: str, log_msg: str) -> dict:
    """Classify a single log. Returns label, tier, confidence, and latency_ms."""
    results = classify_logs([(source, log_msg)])
    return results[0]


# ── Batch pipeline (main entry point) ───────────────────────────────────────
def classify_logs(logs: list[tuple[str, str]]) -> list[dict]:
    """
    Batch classify with 3-tier routing + per-result latency.
    Returns list of dicts:
      { label, tier, confidence, latency_ms }
    Tier routing:
      LegacyCRM source → LLM directly
      Regex match      → done (sub-ms)
      Remainder        → BERT batch → LLM if low confidence
    """
    n       = len(logs)
    results = [None] * n

    # ── Step 1: Route to groups ─────────────────────────────────────────────
    llm_indices   = []
    bert_indices  = []

    for i, (source, log_msg) in enumerate(logs):
        if source == LEGACY_SOURCE:
            llm_indices.append(i)
        else:
            t_start = time.perf_counter()
            label = classify_with_regex(log_msg)
            
            if label:
                latency_ms = (time.perf_counter() - t_start) * 1000
                results[i] = _make_result(label, "Regex", 1.0, latency_ms)
            else:
                bert_indices.append(i)

    # ── Step 2: BERT batch ──────────────────────────────────────────────────
    if bert_indices:
        bert_msgs = [logs[i][1] for i in bert_indices]

        t_bert_start = time.perf_counter()
        bert_results = bert_batch(bert_msgs)
        t_bert_end   = time.perf_counter()

        bert_ms_per_log = (t_bert_end - t_bert_start) * 1000 / len(bert_msgs)

        for idx, (label, conf) in zip(bert_indices, bert_results):
            if label != "Unclassified":
                results[idx] = _make_result(label, "BERT", conf, bert_ms_per_log)
            else:
                llm_indices.append(idx)

    # ── Step 3: LLM (Parallel Concurrency) ──────────────────────────────────
    if llm_indices:
        def parallel_llm(idx):
            src, msg = logs[idx]
            t_llm_0 = time.perf_counter()
            label = classify_with_llm(msg)
            t_llm_ms = (time.perf_counter() - t_llm_0) * 1000
            tier = "LLM" if src == LEGACY_SOURCE else "LLM (fallback)"
            return idx, _make_result(label, tier, None, t_llm_ms)

        # Parallelize API calls to prevent pipeline stall, restricted to 4 workers to prevent OOM
        with ThreadPoolExecutor(max_workers=4) as executor:
            llm_results = list(executor.map(parallel_llm, llm_indices))

        for idx, res in llm_results:
            results[idx] = res

    return results


# ── Pipeline summary ─────────────────────────────────────────────────────────
def pipeline_summary(results: list[dict]) -> dict:
    """
    Aggregate stats from classify_logs output.
    Useful for dashboard and benchmark reporting.
    """
    tier_groups: dict[str, list[float]] = {}
    label_counts: dict[str, int] = {}

    for r in results:
        tier = r["tier"]
        tier_groups.setdefault(tier, []).append(r["latency_ms"])
        label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1

    total = len(results)
    tier_stats = {}
    for tier, latencies in tier_groups.items():
        latencies_sorted = sorted(latencies)
        n = len(latencies_sorted)
        tier_stats[tier] = {
            "count":    n,
            "pct":      round(n / total * 100, 1),
            "p50_ms":   round(statistics.median(latencies_sorted), 2),
            "p95_ms":   round(latencies_sorted[min(int(n * 0.95), n - 1)], 2),
            "p99_ms":   round(latencies_sorted[min(int(n * 0.99), n - 1)], 2),
            "mean_ms":  round(statistics.mean(latencies_sorted), 2),
        }

    return {
        "total":        total,
        "tier_stats":   tier_stats,
        "label_counts": label_counts,
    }


# ── CSV batch classify ───────────────────────────────────────────────────────
def classify_csv(input_path: str, output_path: str = "output.csv") -> tuple[str, pd.DataFrame]:
    """
    Process a batch of logs from a CSV file.
    Required columns: 'source', 'log_message'
    Output: appends 'predicted_label', 'tier_used', 'confidence', 'latency_ms'
    """
    df = pd.read_csv(input_path)
    required = {"source", "log_message"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required columns in CSV. Expected: {required}. Found: {set(df.columns)}")

    log_pairs = list(zip(df["source"], df["log_message"]))
    results   = classify_logs(log_pairs)

    df["predicted_label"] = [r["label"]      for r in results]
    df["tier_used"]        = [r["tier"]       for r in results]
    df["latency_ms"]       = [r["latency_ms"] for r in results]
    df["confidence"]       = [
        f"{r['confidence']:.1%}" if r["confidence"] is not None else "N/A"
        for r in results
    ]

    df.to_csv(output_path, index=False)
    return output_path, df


# Aliases
classify = classify_logs


# ── Self-test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = [
        ("ModernCRM",       "IP 192.168.133.114 blocked due to potential attack"),
        ("BillingSystem",   "User User12345 logged in."),
        ("AnalyticsEngine", "File data_6957.csv uploaded successfully by user User265."),
        ("ModernHR",        "GET /v2/servers/detail HTTP/1.1 status: 200 len: 1583 time: 0.19"),
        ("ModernHR",        "Admin access escalation detected for user 9429"),
        ("LegacyCRM",       "Case escalation for ticket ID 7324 failed because the assigned support agent is no longer active."),
        ("LegacyCRM",       "The 'ReportGenerator' module will be retired in version 4.0."),
    ]

    print(f'{"Source":<20} {"Tier":<18} {"Conf":>6} {"Lat(ms)":>8}  {"Label":<25} Log')
    print("─" * 115)
    results = classify_logs(sample)
    for (source, log), r in zip(sample, results):
        conf = f"{r['confidence']:.0%}" if r["confidence"] else "  N/A"
        print(f'{source:<20} {r["tier"]:<18} {conf:>6} {r["latency_ms"]:>8.1f}  {r["label"]:<25} {log[:40]}')

    summary = pipeline_summary(results)
    print("\n📊 Pipeline Summary:")
    for tier, stats in summary["tier_stats"].items():
        print(f"  {tier}: {stats['count']} logs ({stats['pct']}%) | "
              f"p50={stats['p50_ms']}ms p95={stats['p95_ms']}ms p99={stats['p99_ms']}ms")

    print("\n🏷️  Label distribution:")
    for label, count in sorted(summary["label_counts"].items(), key=lambda x: -x[1]):
        print(f"  • {label}: {count}")
