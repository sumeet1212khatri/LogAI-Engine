
from __future__ import annotations
import os
import time
import statistics
import pandas as pd
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from processor_regex import classify_with_regex
from processor_bert  import classify_batch as bert_batch
from processor_llm   import classify_with_llm

# ── Config ──────────────────────────────────────────────────────────────────
LEGACY_SOURCE = os.getenv("LEGACY_SOURCE", "LegacyCRM")


# ── Result type ─────────────────────────────────────────────────────────────
def _make_result(label: str, tier: str, confidence, latency_ms: float) -> dict:
    return {
        "label":      label,
        "tier":       tier,
        "confidence": confidence,
        "latency_ms": round(latency_ms, 4), 
    }


# ── Caching Layer (Max RAM Eater) ───────────────────────────────────────────
@lru_cache(maxsize=500000) 
def cached_llm_call(log_msg: str) -> str:
    """Executes the expensive LLM call only if the string misses the cache."""
    return classify_with_llm(log_msg)


# ── Single log (backward-compatible) ────────────────────────────────────────
def classify_log(source: str, log_msg: str) -> dict:
    results = classify_logs([(source, log_msg)])
    return results[0]


# ── Batch pipeline (main entry point) ───────────────────────────────────────
def classify_logs(logs: list[tuple[str, str]]) -> list[dict]:
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

    # ── Step 2: BERT batch (CPU Bound - No Threads Allowed Here) ────────────
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

    # ── Step 3: LLM (I/O Bound - Threading Applied Here) ────────────────────
    if llm_indices:
        def parallel_llm(idx):
            src, msg = logs[idx]
            
            t_llm_0 = time.perf_counter()
            label = cached_llm_call(msg)
            t_llm_ms = (time.perf_counter() - t_llm_0) * 1000
            
            base_tier = "LLM" if src == LEGACY_SOURCE else "LLM (fallback)"
            tier = f"{base_tier} (Cache Hit)" if t_llm_ms < 5 else f"{base_tier} (API Call)"
            
            return idx, _make_result(label, tier, None, t_llm_ms)

        with ThreadPoolExecutor() as executor:
            llm_results = list(executor.map(parallel_llm, llm_indices))

        for idx, res in llm_results:
            results[idx] = res

    return results


# ── Pipeline summary ─────────────────────────────────────────────────────────
def pipeline_summary(results: list[dict]) -> dict:
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
            "p50_ms":   round(statistics.median(latencies_sorted), 4),
            "p95_ms":   round(latencies_sorted[min(int(n * 0.95), n - 1)], 4),
            "p99_ms":   round(latencies_sorted[min(int(n * 0.99), n - 1)], 4),
            "mean_ms":  round(statistics.mean(latencies_sorted), 4),
            "total_ms": round(sum(latencies_sorted), 4), 
        }

    return {
        "total":        total,
        "tier_stats":   tier_stats,
        "label_counts": label_counts,
    }


# ── CSV batch classify (Hybrid Processing) ───────────────────────────────────
def classify_csv(input_path: str, output_path: str = "output.csv") -> tuple[str, pd.DataFrame]:
    """
    Ultra-Optimized Batch Processing for 2M+ Logs.
    Outer chunks run sequentially (bypasses GIL for BERT, preserves main memory cache).
    Inner LLM calls thread automatically inside classify_logs.
    """
    df = pd.read_csv(input_path)
    required = {"source", "log_message"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required columns in CSV. Expected: {required}. Found: {set(df.columns)}")

    log_pairs = list(zip(df["source"], df["log_message"]))
    total_logs = len(log_pairs)
    
    # Chunk size controls how much data is in RAM at once
    chunk_size = 50000 
    chunks = [log_pairs[i:i + chunk_size] for i in range(0, total_logs, chunk_size)]
    
    results = []
    
    print(f"🔥 Processing {len(chunks)} chunks... (BERT handles CPU batching, LLM handles I/O threads)")
    
    t_start = time.perf_counter()
    # Process sequentially to avoid GIL locks on BERT and keep the cache in one memory block
    for chunk in chunks:
        results.extend(classify_logs(chunk))
    t_end = time.perf_counter()
    
    print(f"⏱️ True Wall-Clock Processing Time: {(t_end - t_start):.2f} seconds")

    df["predicted_label"] = [r["label"]       for r in results]
    df["tier_used"]       = [r["tier"]        for r in results]
    df["latency_ms"]      = [r["latency_ms"]  for r in results]
    df["confidence"]      = [
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
        ("LegacyCRM",       "Case escalation failed due to active timeout."),
    ]

    print("Running quick test...")
    results = classify_logs(sample)
    print("Done. No errors.")
