# LogAI-Engine


<img width="1919" height="943" alt="image" src="https://github.com/user-attachments/assets/ccef8a14-00fc-40c7-a677-4201ff2f1b51" />


<div align="center">

**A production-grade, 3-tier hybrid log classification pipeline.**  
Classifies 2,000,000 enterprise logs in under 9 minutes with 92% of traffic handled at sub-millisecond cost.

[![HuggingFace Space](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-FFD21E?style=for-the-badge)](https://huggingface.co/spaces/NOT-OMEGA/LogAI-Engine)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Stress Test](https://img.shields.io/badge/Stress%20Test-2M%20logs%20✓-brightgreen?style=for-the-badge)](#benchmark-results)

</div>

---

<div align="center">
  <img src="https://github.com/user-attachments/assets/00c8d59b-b312-4f0f-b7b1-dce3e4eb4ff0" alt="LogAI-Engine UI" width="100%" />
</div>

---

## Overview

LogAI-Engine solves a real operational problem: enterprise systems produce millions of heterogeneous log lines per day, and routing them through a single expensive model is slow and wasteful. This project implements a **cascading 3-tier classifier** that routes each log to the cheapest tier that can confidently label it — and only escalates to LLM when strictly necessary.

The result: **4,131 logs/second** throughput with a median classification latency of **0.024ms** for the majority of traffic.

---

## Architecture

```
Incoming Log
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│  Router                                                      │
│  Is source == "LegacyCRM"?  ──YES──► Tier 3 (LLM Direct)   │
└─────────────────────────────────────────────────────────────┘
      │ NO
      ▼
┌─────────────────────────────────────────────────────────────┐
│  Tier 1 — Regex  (< 0.1ms, covers ~92% of traffic)         │
│  40+ pre-compiled patterns across 6 label categories        │
│  Returns: label or None                                      │
└─────────────────────────────────────────────────────────────┘
      │ None (no match)
      ▼
┌─────────────────────────────────────────────────────────────┐
│  Tier 2 — BERT / ONNX  (~37ms amortized per log)           │
│  Sentence embeddings + sklearn classifier                   │
│  Confidence threshold: 30%                                  │
│  Returns: label or "Unclassified"                            │
└─────────────────────────────────────────────────────────────┘
      │ "Unclassified"
      ▼
┌─────────────────────────────────────────────────────────────┐
│  Tier 3 — LLM  (Mistral-7B via HuggingFace Inference API)  │
│  MD5 hash-based LRU cache (500k slots)                      │
│  4-thread parallel execution + exponential backoff retry    │
│  Returns: label (always, with fallback to "Unclassified")   │
└─────────────────────────────────────────────────────────────┘
```

### Why this order?

Each tier is ordered by cost (latency + money), not accuracy. Regex is free and instant. BERT is cheap but has a warm-up cost. LLM is accurate for ambiguous cases but expensive per call. The cascade ensures you only pay for what you need.

---

## Benchmark Results

Stress tested on **2,000,000 real enterprise logs**.

### Throughput

| Metric | Value |
|---|---|
| Total logs | 2,000,000 |
| Wall time | 484.24 seconds |
| Throughput | **4,131 logs/second** |
| Unclassified rate | 0.31% (6,117 logs) |

### Performance by tier

| Tier | Logs Handled | Coverage | p50 Latency | Cost |
|---|---|---|---|---|
| 🟢 Regex | 1,844,376 | **92.2%** | 0.0244ms | Free |
| 🔵 BERT | 149,507 | 7.5% | ~37ms (amortized) | GPU-free (ONNX CPU) |
| ⚡ LLM Cache Hit | 5,922 | 0.3% | 0.0ms | Free (RAM) |
| 🟡 LLM API Call | 195 | 0.01% | 10.1ms | Minimal |

### Label distribution (2M logs)

```
Error              ████████████████████  559,443  (28.0%)
Security Alert     ████████████████████  557,463  (27.9%)
System Notification ██████████████       428,609  (21.4%)
User Action        ██████████           298,784  (14.9%)
HTTP Status        █████                149,584   (7.5%)
Unclassified       ░                      6,117   (0.3%)
```

---

## Features

- **3-tier hybrid pipeline** — Regex → BERT → LLM with automatic routing and fallback
- **ONNX acceleration** — 3–5× faster than standard PyTorch inference on CPU
- **LRU cache (500k slots)** — MD5-based deduplication eliminates redundant LLM API calls
- **Multi-core batch processing** — `ProcessPoolExecutor` across all available CPU cores
- **Per-result telemetry** — every log carries its own tier, confidence score, and latency
- **Production hardening** — retry with exponential backoff, configurable timeouts, graceful fallback on API errors
- **Gradio UI** — real-time single-log analyzer + CSV batch upload with live analytics
- **Docker-ready** — single `docker run` to deploy anywhere

---

## Labels

| Label | Description |
|---|---|
| `Error` | System crashes, timeouts, connection failures, I/O errors |
| `Security Alert` | Login failures, IP blocks, privilege escalation, brute force |
| `System Notification` | Backups, deployments, health checks, cron jobs, config reloads |
| `User Action` | Login/logout, profile updates, account creation/deletion |
| `HTTP Status` | REST API request/response logs with status codes |
| `Workflow Error` | LegacyCRM-specific: failed escalations, broken workflows *(LLM only)* |
| `Deprecation Warning` | LegacyCRM-specific: feature retirement notices *(LLM only)* |
| `Unclassified` | No tier could classify with sufficient confidence |

---

## Quick Start

### Prerequisites

- Python 3.11+
- A HuggingFace account with `HF_TOKEN` set (for Tier 3 LLM calls)

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

### Classify a single log (Python)

```python
from classify import classify_log

result = classify_log(
    source="ModernCRM",
    log_msg="IP 192.168.133.114 blocked due to potential attack"
)

print(result)
# {
#   "label": "Security Alert",
#   "tier": "Regex",
#   "confidence": 1.0,
#   "latency_ms": 0.0231
# }
```

### Classify a batch

```python
from classify import classify_logs

logs = [
    ("ModernHR",    "Multiple login failures occurred on user 6454 account"),
    ("BillingSystem", "GET /v2/servers/detail HTTP/1.1 status: 200"),
    ("LegacyCRM",   "Case escalation for ticket ID 7324 failed."),
]

results = classify_logs(logs)
```

### Classify a CSV file

```python
from classify import classify_csv

output_path, df = classify_csv("logs.csv", "classified_output.csv")
```

Input CSV must have columns: `source`, `log_message`.  
Output adds: `predicted_label`, `tier_used`, `confidence`, `latency_ms`.

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

Supported `source` values: `ModernCRM`, `ModernHR`, `BillingSystem`, `AnalyticsEngine`, `ThirdPartyAPI`, `LegacyCRM`.

Logs from `LegacyCRM` are routed directly to Tier 3 (LLM) to detect `Workflow Error` and `Deprecation Warning` — categories that are too context-dependent for regex or BERT.

---

## Project Structure

```
LogAI-Engine/
├── classify.py           # Main pipeline — routing logic, batch orchestration, telemetry
├── processor_regex.py    # Tier 1 — 40+ compiled regex patterns, coverage benchmark utils
├── processor_bert.py     # Tier 2 — ONNX/PyTorch BERT embeddings + sklearn classifier
├── processor_llm.py      # Tier 3 — Mistral-7B via HF Inference API, cache, retry logic
├── app_gradio.py         # Gradio UI — real-time analyzer + batch CSV processor
├── models/
│   └── log_classifier.joblib   # Trained sklearn classifier (SVM/RandomForest on embeddings)
├── onnx_model/                  # ONNX-exported BERT tokenizer + weights
│   ├── model.onnx
│   ├── tokenizer.json
│   └── vocab.txt
├── training/
│   └── training.py              # Model training script (run on Colab)
├── Dockerfile
└── requirements.txt
```

---

## How the Tiers Work

### Tier 1 — Regex

40+ hand-crafted patterns pre-compiled at import time with `re.compile(..., re.IGNORECASE)`. Patterns are ordered from most-specific to least-specific to prevent mis-labeling. Covers HTTP requests, authentication events, system operations, network/firewall events, and structured error prefixes.

Latency: **sub-millisecond** (p50: 0.024ms at 2M scale).

### Tier 2 — BERT (ONNX)

Logs that don't match any regex pattern are embedded using a fine-tuned `all-MiniLM-L6-v2` model, exported to ONNX format for CPU-optimized inference. Embeddings are classified by a trained sklearn model. If the classifier's confidence is below 30%, the log is escalated to Tier 3 rather than returning a wrong label.

ONNX mode is **3–5× faster** than standard PyTorch on CPU. Falls back to PyTorch automatically if the ONNX model file is not present.

### Tier 3 — LLM

Uses `Mistral-7B-Instruct-v0.3` via the HuggingFace Inference API with few-shot prompting. Includes:
- **LRU cache (500k slots)** keyed on MD5 hash of the log message — eliminates duplicate API calls entirely
- **4-thread parallel execution** via `ThreadPoolExecutor`
- **Exponential backoff retry** (up to 2 retries, starting at 1s delay)
- **Hard timeout** (5 seconds) to prevent pipeline stalls
- **Graceful degradation** — returns `"Unclassified"` on all failure paths, never raises

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `HF_TOKEN` | Yes (for Tier 3) | HuggingFace API token for Mistral-7B inference |

Without `HF_TOKEN`, Tier 1 and Tier 2 function normally. Tier 3 logs are returned as `"Unclassified"`.

---

## Dependencies

| Package | Purpose |
|---|---|
| `sentence-transformers` | BERT embeddings (PyTorch fallback) |
| `onnxruntime` | Optimized BERT inference on CPU |
| `scikit-learn` | Log classifier trained on embeddings |
| `huggingface-hub` | Mistral-7B inference client |
| `gradio` | Web UI |
| `pandas` | CSV batch processing |
| `joblib` | Model serialization |
| `fastapi` + `uvicorn` | Production REST API (optional) |

---

## Training Your Own Classifier

The BERT-backed sklearn classifier can be retrained on your own labeled log data.

```bash
# Open in Google Colab (recommended — free GPU)
training/training.py
```

Training produces `models/log_classifier.joblib`. Export BERT to ONNX for inference speedup:

```python
from optimum.onnxruntime import ORTModelForFeatureExtraction
model = ORTModelForFeatureExtraction.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", export=True)
model.save_pretrained("onnx_model/")
```

---

## Live Demo

Try it on HuggingFace Spaces — no setup required:

**[https://huggingface.co/spaces/NOT-OMEGA/LogAI-Engine](https://huggingface.co/spaces/NOT-OMEGA/LogAI-Engine)**

Upload any CSV with `source` and `log_message` columns to run batch classification.

---

## License

[MIT](LICENSE)

---

<div align="center">
  Built with Python · Deployed on HuggingFace Spaces · Stress tested at 2M logs
</div>
