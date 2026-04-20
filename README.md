# LogAI-Engine

<div align="center">

![LogAI-Engine Banner](https://github.com/user-attachments/assets/32c9f5f8-40b3-4d52-bed7-c44842b703fb)

**A production-grade, 3-tier hybrid log classification pipeline.**  
Classifies **2,000,000** enterprise logs in under **6 minutes** with **92% of traffic** handled at sub-millisecond cost.

[![HuggingFace Space](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-FFD21E?style=for-the-badge)](https://huggingface.co/spaces/NOT-OMEGA/LogAI-Engine)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Stress Test](https://img.shields.io/badge/Stress%20Test-2M%20logs%20✓-brightgreen?style=for-the-badge)](#benchmark-results)

</div>

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Architecture](#architecture)
- [Design Decisions](#design-decisions)
- [Benchmark Results](#benchmark-results)
- [Features](#features)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [CSV Format](#csv-format)
- [Project Structure](#project-structure)
- [How the Tiers Work](#how-the-tiers-work)
- [Training Your Own Classifier](#training-your-own-classifier)
- [Environment Variables](#environment-variables)
- [Dependencies](#dependencies)
- [Live Demo](#live-demo)

---

## Problem Statement

Enterprise systems generate millions of heterogeneous log lines per day. Routing all traffic through a single LLM is accurate but prohibitively expensive and slow at scale. Routing all traffic through regex alone misses ambiguous or context-dependent entries.

LogAI-Engine solves this with a **cascading 3-tier classifier**: each log is routed to the cheapest tier capable of classifying it confidently, and only escalated further when the current tier is insufficient. The outcome is near-LLM accuracy at Regex-level cost for the vast majority of traffic.

---

## Architecture

```
Incoming Log
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│  Router                                                      │
│  source == "LegacyCRM"?  ──YES──► Tier 3 (LLM Direct)      │
└─────────────────────────────────────────────────────────────┘
      │ NO
      ▼
┌─────────────────────────────────────────────────────────────┐
│  Tier 1 — Regex                                             │
│  40+ pre-compiled patterns across 6 label categories        │
│  Latency: < 0.1ms  │  Coverage: ~92% of traffic            │
│  Returns: label  │  None (no match → escalate)             │
└─────────────────────────────────────────────────────────────┘
      │ None
      ▼
┌─────────────────────────────────────────────────────────────┐
│  Tier 2 — BERT / ONNX                                       │
│  Sentence embeddings + sklearn classifier                   │
│  Confidence threshold: 30%                                  │
│  Returns: label  │  "Unclassified" (low confidence → LLM)  │
└─────────────────────────────────────────────────────────────┘
      │ "Unclassified"
      ▼
┌─────────────────────────────────────────────────────────────┐
│  Tier 3 — LLM (Mistral-7B via HuggingFace Inference API)   │
│  MD5 hash-based LRU cache (500k slots)                      │
│  4-thread parallel execution + exponential backoff retry    │
│  Returns: label (always; fallback to "Unclassified")        │
└─────────────────────────────────────────────────────────────┘
```

---

## Design Decisions

**Why cascade by cost, not accuracy?**  
Regex has zero monetary cost and sub-millisecond latency. BERT/ONNX is cheap and runs entirely on CPU. LLM is accurate for ambiguous cases but slow and billed per call. Ordering by cost means you pay the LLM price only for the ~0.01% of traffic that truly requires it.

**Why ONNX for Tier 2?**  
ONNX Runtime on CPU delivers 3–5× faster inference than standard PyTorch for the embedding step, with no GPU dependency. This is critical for keeping Tier 2 viable as a low-cost fallback rather than a bottleneck.

**Why route LegacyCRM logs directly to Tier 3?**  
`LegacyCRM` emits two label categories — `Workflow Error` and `Deprecation Warning` — that are structurally distinct from other sources and too context-dependent for regex or embedding-based matching. Forcing them through earlier tiers would produce high misclassification rates. Direct LLM routing for a known-problematic source is a deliberate and bounded decision, not a general override.

**Why MD5-based LRU caching on Tier 3?**  
Enterprise logs frequently repeat: the same error string, the same status code, the same cron job notification. An LRU cache keyed on the MD5 hash of the log message eliminates redundant API calls entirely for these cases. At 2M log scale, this reduced live LLM calls to just 1,589 out of 6,109 LLM-routed logs — a ~74% cache hit rate on LLM traffic.

**Why `ProcessPoolExecutor` for batch processing?**  
Python's GIL limits CPU-bound parallelism in threads. Tier 1 (regex) and Tier 2 (ONNX) are CPU-bound. `ProcessPoolExecutor` distributes batch chunks across physical cores, enabling true parallel execution and the 4,131 logs/second throughput observed at 2M scale.

---

## Benchmark Results

Stress tested on **2,000,000 real enterprise logs**.

### Throughput

| Metric | Value |
|---|---|
| Total logs | 2,000,000 |
| Wall time | 318.29 seconds |
| Throughput | **~6,286 logs/second** |
| Unclassified rate | 0.31% (6,117 logs) |

### Tier breakdown

| Tier | Logs Handled | Coverage | p50 Latency | Cost |
|---|---|---|---|---|
| 🟢 Regex | 1,844,376 | 92.2% | 0.020ms (p50) | Free |
| 🔵 BERT / ONNX | 149,507 | 7.5% | 14.68ms/log (~2,194s batch) | CPU-only |
| ⚡ LLM (Cache Hit) | 4,520 | 0.2% | 0.0ms | Free (RAM) |
| 🟡 LLM (API Call) | 1,589 | 0.08% | p50=8.5ms, p95=101ms, p99=120ms | Minimal |

### Label distribution (2M logs)

```
Error               ████████████████████  559,443  (28.0%)
Security Alert      ████████████████████  557,463  (27.9%)
System Notification ██████████████        428,609  (21.4%)
User Action         ██████████            298,784  (14.9%)
HTTP Status         █████                 149,584   (7.5%)
Unclassified        ░                       6,117   (0.3%)
```

---

## Features

- **3-tier hybrid pipeline** — Regex → BERT/ONNX → LLM with automatic routing and graceful fallback at each stage
- **ONNX acceleration** — 3–5× faster than standard PyTorch inference on CPU; falls back to PyTorch automatically
- **LRU cache (500k slots)** — MD5-keyed deduplication eliminates redundant LLM API calls
- **Multi-core batch processing** — `ProcessPoolExecutor` across all available CPU cores for true parallelism
- **Per-result telemetry** — every classified log carries its tier, confidence score, and latency
- **Production hardening** — exponential backoff retry (up to 2 retries), configurable timeouts, graceful degradation on API errors
- **Gradio UI** — real-time single-log analyzer and CSV batch upload with live pipeline analytics
- **Docker-ready** — single `docker run` command to deploy anywhere

---

## Quick Start

### Prerequisites

- Python 3.11+
- A HuggingFace account with `HF_TOKEN` set (required only for Tier 3 LLM calls)

### Installation

```bash
git clone https://github.com/NOT-OMEGA/LogAI-Engine.git
cd LogAI-Engine
pip install -r requirements.txt
```

### Run the Gradio UI

```bash
HF_TOKEN=your_token python app_gradio.py
```

Open [http://localhost:7860](http://localhost:7860)

### Run via Docker

```bash
docker build -t logai-engine .
docker run -p 7860:7860 -e HF_TOKEN=your_token logai-engine
```

---

## Usage

### Classify a single log

```python
from classify import classify_log

result = classify_log(
    source="ModernCRM",
    log_msg="IP 192.168.133.114 blocked due to potential attack"
)

# {
#   "label": "Security Alert",
#   "tier": "Regex",
#   "confidence": 1.0,
#   "latency_ms": 0.023
# }
```

### Classify a batch

```python
from classify import classify_logs

logs = [
    ("ModernHR",      "Multiple login failures occurred on user 6454 account"),
    ("BillingSystem", "GET /v2/servers/detail HTTP/1.1 status: 200"),
    ("LegacyCRM",     "Case escalation for ticket ID 7324 failed."),
]

results = classify_logs(logs)
```

### Classify a CSV file

```python
from classify import classify_csv

output_path, df = classify_csv("logs.csv", "classified_output.csv")
```

Input CSV must have columns: `source`, `log_message`.  
Output appends: `predicted_label`, `tier_used`, `confidence`, `latency_ms`.

### Get pipeline analytics

```python
from classify import classify_logs, pipeline_summary

results = classify_logs(logs)
summary = pipeline_summary(results)

# {
#   "total": 3,
#   "tier_stats": { "Regex": { "count": 2, "p50_ms": 0.02, ... }, ... },
#   "label_counts": { "Security Alert": 1, "HTTP Status": 1, ... }
# }
```

---

## CSV Format

```csv
source,log_message
ModernCRM,User User12345 logged in.
ModernHR,Multiple login failures occurred on user 6454 account
BillingSystem,GET /v2/servers/detail HTTP/1.1 status: 200 len: 1583 time: 0.19
AnalyticsEngine,System crashed due to disk I/O failure on node-3
LegacyCRM,Case escalation for ticket ID 7324 failed because the assigned support agent is no longer active.
```

**Supported `source` values:** `ModernCRM`, `ModernHR`, `BillingSystem`, `AnalyticsEngine`, `ThirdPartyAPI`, `LegacyCRM`

Logs with `source == "LegacyCRM"` are routed directly to Tier 3 to handle the `Workflow Error` and `Deprecation Warning` categories, which are too context-dependent for regex or embedding-based classification.

---

## Project Structure

```
LogAI-Engine/
├── classify.py              # Core pipeline: routing logic, batch orchestration, telemetry
├── processor_regex.py       # Tier 1: 40+ compiled regex patterns, coverage benchmark utils
├── processor_bert.py        # Tier 2: ONNX/PyTorch BERT embeddings + sklearn classifier
├── processor_llm.py         # Tier 3: Mistral-7B via HF Inference API, LRU cache, retry logic
├── app_gradio.py            # Gradio UI: real-time analyzer + batch CSV processor
├── models/
│   └── log_classifier.joblib       # Trained sklearn classifier (embeddings → label)
├── onnx_model/
│   ├── model.onnx                  # ONNX-exported BERT weights
│   ├── tokenizer.json
│   └── vocab.txt
├── training/
│   └── training.py                 # Classifier training script (designed for Colab)
├── Dockerfile
└── requirements.txt
```

---

## How the Tiers Work

### Tier 1 — Regex

40+ hand-crafted patterns pre-compiled at import time with `re.compile(..., re.IGNORECASE)`. Patterns are ordered from most-specific to least-specific to prevent mis-labeling across overlapping categories. Covers HTTP requests, authentication events, system operations, network and firewall events, and structured error prefixes.

p50 latency at 2M scale: **0.024ms**.

### Tier 2 — BERT (ONNX)

Logs that match no regex pattern are embedded using a fine-tuned `all-MiniLM-L6-v2` model exported to ONNX for CPU-optimized inference. Embeddings are fed to a trained sklearn classifier. If the classifier's confidence falls below 30%, the log is escalated to Tier 3 rather than returning a potentially wrong label.

ONNX provides 3–5× faster inference than PyTorch on CPU. Falls back to PyTorch automatically if the ONNX model is not present.

### Tier 3 — LLM

Uses `Mistral-7B-Instruct-v0.3` via the HuggingFace Inference API with few-shot prompting. Implementation includes:

- **LRU cache (500k slots)** keyed on MD5 hash of the log message, eliminating duplicate API calls
- **4-thread parallel execution** via `ThreadPoolExecutor`
- **Exponential backoff retry** — up to 2 retries, starting at 1s delay
- **Hard timeout** — 5 seconds per call to prevent pipeline stalls
- **Graceful degradation** — returns `"Unclassified"` on all failure paths; never raises

---

## Labels

| Label | Description |
|---|---|
| `Error` | System crashes, timeouts, connection failures, I/O errors |
| `Security Alert` | Login failures, IP blocks, privilege escalation, brute force attempts |
| `System Notification` | Backups, deployments, health checks, cron jobs, config reloads |
| `User Action` | Login/logout, profile updates, account creation and deletion |
| `HTTP Status` | REST API request/response logs with status codes |
| `Workflow Error` | LegacyCRM-specific: failed escalations, broken workflows *(Tier 3 only)* |
| `Deprecation Warning` | LegacyCRM-specific: feature retirement notices *(Tier 3 only)* |
| `Unclassified` | No tier could classify with sufficient confidence |

---

## Training Your Own Classifier

The BERT-backed sklearn classifier can be retrained on any labeled log dataset.

```bash
# Recommended: Google Colab (free GPU)
training/training.py
```

Training produces `models/log_classifier.joblib`. To export BERT to ONNX for inference speedup:

```python
from optimum.onnxruntime import ORTModelForFeatureExtraction

model = ORTModelForFeatureExtraction.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2",
    export=True
)
model.save_pretrained("onnx_model/")
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `HF_TOKEN` | Yes (Tier 3 only) | HuggingFace API token for Mistral-7B inference |

Without `HF_TOKEN`, Tier 1 and Tier 2 operate normally. Logs routed to Tier 3 are returned as `"Unclassified"`.

---

## Dependencies

| Package | Purpose |
|---|---|
| `sentence-transformers` | BERT embeddings (PyTorch fallback path) |
| `onnxruntime` | Optimized BERT inference on CPU |
| `scikit-learn` | Sklearn classifier trained on sentence embeddings |
| `huggingface-hub` | Mistral-7B inference client |
| `gradio` | Web UI |
| `pandas` | CSV batch processing |
| `joblib` | Model serialization |
| `fastapi` + `uvicorn` | Production REST API (optional) |

---

## Live Demo

Try it on HuggingFace Spaces — no setup required:

**[https://huggingface.co/spaces/NOT-OMEGA/LogAI-Engine](https://huggingface.co/spaces/NOT-OMEGA/LogAI-Engine)**

Upload any CSV with `source` and `log_message` columns to run batch classification at scale.

---

## License

[MIT](LICENSE)

---

<div align="center">
  Built with Python &nbsp;·&nbsp; Deployed on HuggingFace Spaces &nbsp;·&nbsp; Stress tested at 2M logs
</div>
