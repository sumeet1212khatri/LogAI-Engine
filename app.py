"""
Log Classification System — HuggingFace Spaces
Gradio UI for the 3-tier hybrid log classification pipeline.
"""
from __future__ import annotations
import io
import time
import pandas as pd
import gradio as gr
from classify import classify_log, classify_csv

# ── Source options ──────────────────────────────────────────────────────────
SOURCES = [
    "ModernCRM",
    "ModernHR",
    "BillingSystem",
    "AnalyticsEngine",
    "ThirdPartyAPI",
    "LegacyCRM",
]

TIER_COLORS = {
    "Regex":        "🟢",
    "BERT":         "🔵",
    "LLM":          "🟡",
    "LLM (fallback)": "🟠",
}

EXAMPLE_LOGS = [
    ["ModernCRM",       "User User12345 logged in."],
    ["ModernHR",        "Multiple login failures occurred on user 6454 account"],
    ["BillingSystem",   "GET /v2/servers/detail HTTP/1.1 status: 200 len: 1583 time: 0.19"],
    ["AnalyticsEngine", "System crashed due to disk I/O failure on node-3"],
    ["LegacyCRM",       "Case escalation for ticket ID 7324 failed — support agent is no longer active."],
    ["LegacyCRM",       "The 'BulkEmailSender' feature will be deprecated in v5.0. Use 'EmailCampaignManager'."],
]


# ── Single log tab ──────────────────────────────────────────────────────────
def classify_single(source: str, log_message: str):
    if not log_message.strip():
        return "—", "—", "—", "—"

    t0 = time.perf_counter()
    result = classify_log(source, log_message)
    latency_ms = (time.perf_counter() - t0) * 1000

    label      = result["label"]
    tier       = result["tier"]
    confidence = f"{result['confidence']:.1%}" if result["confidence"] is not None else "N/A"
    icon       = TIER_COLORS.get(tier, "⚪")

    return (
        label,
        f"{icon} {tier}",
        confidence,
        f"{latency_ms:.1f} ms",
    )


# ── Batch CSV tab ───────────────────────────────────────────────────────────
def classify_batch(file):
    if file is None:
        return None, "⚠️ Please upload a CSV file."

    try:
        output_path, df = classify_csv(file.name, "/tmp/classified_output.csv")
    except ValueError as e:
        return None, f"⚠️ {e}"
    except Exception as e:
        return None, f"❌ Error: {e}"

    total = len(df)
    tier_counts  = df["tier_used"].value_counts().to_dict()
    label_counts = df["predicted_label"].value_counts().to_dict()

    tier_lines  = "\n".join(f"  {TIER_COLORS.get(k,'⚪')} {k}: {v} ({v/total:.0%})" for k, v in tier_counts.items())
    label_lines = "\n".join(f"  • {k}: {v}" for k, v in label_counts.items())

    stats = (
        f"✅ Classified {total} logs\n\n"
        f"📊 Tier breakdown:\n{tier_lines}\n\n"
        f"🏷️ Label distribution:\n{label_lines}"
    )

    return output_path, stats


# ── UI ──────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Log Classification System", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
# 🔍 Log Classification System
**3-tier hybrid pipeline** → 🟢 Regex · 🔵 BERT + LogReg · 🟡 LLM
Built to mimic production enterprise log monitoring architecture.
""")

    with gr.Tabs():

        # ── Tab 1: Single Log ────────────────────────────────────────────
        with gr.Tab("Single Log"):
            with gr.Row():
                source_input = gr.Dropdown(
                    choices=SOURCES,
                    value="ModernCRM",
                    label="Source System",
                )
                log_input = gr.Textbox(
                    label="Log Message",
                    placeholder="Paste a log message here...",
                    lines=3,
                )

            classify_btn = gr.Button("Classify", variant="primary")

            with gr.Row():
                label_out      = gr.Textbox(label="🏷️ Predicted Label",     interactive=False)
                tier_out       = gr.Textbox(label="⚙️ Tier Used",           interactive=False)
                confidence_out = gr.Textbox(label="📈 Confidence",          interactive=False)
                latency_out    = gr.Textbox(label="⏱️ Latency",             interactive=False)

            classify_btn.click(
                fn=classify_single,
                inputs=[source_input, log_input],
                outputs=[label_out, tier_out, confidence_out, latency_out],
            )

            gr.Examples(
                examples=EXAMPLE_LOGS,
                inputs=[source_input, log_input],
                label="📋 Example Logs (click to try)",
            )

        # ── Tab 2: Batch CSV ─────────────────────────────────────────────
        with gr.Tab("Batch CSV Upload"):
            gr.Markdown("""
Upload a CSV with columns: **`source`**, **`log_message`**  
Download the classified CSV with added columns: `predicted_label`, `tier_used`, `confidence`.
""")
            with gr.Row():
                with gr.Column():
                    csv_input  = gr.File(label="📂 Upload CSV", file_types=[".csv"])
                    batch_btn  = gr.Button("Classify All", variant="primary")
                with gr.Column():
                    csv_output = gr.File(label="📥 Download Classified CSV")
                    stats_out  = gr.Textbox(label="📊 Stats", lines=12, interactive=False)

            batch_btn.click(
                fn=classify_batch,
                inputs=[csv_input],
                outputs=[csv_output, stats_out],
            )

            gr.Markdown("""
**Sample CSV format:**
```
source,log_message
ModernCRM,User User123 logged in.
LegacyCRM,Case escalation for ticket ID 7324 failed.
BillingSystem,GET /api/v2/invoice HTTP/1.1 status: 500
```
""")

        # ── Tab 3: Architecture ──────────────────────────────────────────
        with gr.Tab("Architecture"):
            gr.Markdown("""
## 🏗️ 3-Tier Hybrid Pipeline

| Tier | Method | Coverage | Latency | When Used |
|------|--------|----------|---------|-----------|
| 🟢 Regex | Python `re` patterns | ~21% | < 1ms | Fixed patterns (login, backup, etc.) |
| 🔵 BERT | `all-MiniLM-L6-v2` + LogReg | ~79% | 20–80ms | High-volume categories with 150+ samples |
| 🟡 LLM | HuggingFace Inference API | ~0.3% | 500–2000ms | LegacyCRM logs, rare patterns |

## 📊 Model Performance (from training)
- **BERT + LogReg** trained on 2,410 synthetic enterprise logs
- **Confidence threshold**: 0.5 (below → escalate to LLM)
- **Source-aware routing**: `LegacyCRM` bypasses ML entirely (only 7 training samples)

## 🔑 Environment Variables
| Secret | Required For |
|--------|-------------|
| `HF_TOKEN` | LLM inference (LegacyCRM logs) |
""")

if __name__ == "__main__":
    demo.launch()
