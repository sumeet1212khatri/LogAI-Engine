"""
Log Classification System — HuggingFace Spaces
Ultra-modern 3D UI with custom CSS
"""
from __future__ import annotations
import io
import time
import pandas as pd
import gradio as gr
from classify import classify_log, classify_csv

SOURCES = [
    "ModernCRM", "ModernHR", "BillingSystem",
    "AnalyticsEngine", "ThirdPartyAPI", "LegacyCRM",
]

TIER_COLORS = {
    "Regex":          "🟢",
    "BERT":           "🔵",
    "LLM":            "🟡",
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

# ── Custom CSS — 3D Modern Dark Theme ──────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;700&display=swap');

:root {
    --bg-primary: #050810;
    --bg-secondary: #0a0f1e;
    --bg-card: #0d1425;
    --bg-card-hover: #111a30;
    --accent-cyan: #00d4ff;
    --accent-blue: #0066ff;
    --accent-purple: #7c3aed;
    --accent-green: #00ff88;
    --accent-orange: #ff6b00;
    --text-primary: #e2e8f0;
    --text-secondary: #94a3b8;
    --text-muted: #475569;
    --border-glow: rgba(0, 212, 255, 0.3);
    --shadow-3d: 0 20px 60px rgba(0, 0, 0, 0.8), 0 0 40px rgba(0, 102, 255, 0.15);
    --glow-cyan: 0 0 20px rgba(0, 212, 255, 0.4), 0 0 40px rgba(0, 212, 255, 0.2);
    --glow-blue: 0 0 20px rgba(0, 102, 255, 0.4);
}

/* ── Base ── */
body, .gradio-container {
    background: var(--bg-primary) !important;
    font-family: 'Exo 2', sans-serif !important;
    color: var(--text-primary) !important;
}

.gradio-container {
    background:
        radial-gradient(ellipse at 20% 20%, rgba(0, 102, 255, 0.08) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 80%, rgba(124, 58, 237, 0.08) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 50%, rgba(0, 212, 255, 0.03) 0%, transparent 70%),
        var(--bg-primary) !important;
    min-height: 100vh;
}

/* ── Header ── */
.main-header {
    text-align: center;
    padding: 48px 24px 32px;
    position: relative;
}

.main-header::before {
    content: '';
    position: absolute;
    top: 0; left: 50%;
    transform: translateX(-50%);
    width: 600px; height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent-cyan), var(--accent-blue), transparent);
    box-shadow: var(--glow-cyan);
}

/* ── Tab Navigation ── */
.tab-nav {
    background: rgba(13, 20, 37, 0.8) !important;
    border: 1px solid rgba(0, 212, 255, 0.15) !important;
    border-radius: 16px !important;
    padding: 6px !important;
    backdrop-filter: blur(20px) !important;
    box-shadow: var(--shadow-3d) !important;
}

.tab-nav button {
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    color: var(--text-secondary) !important;
    background: transparent !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 24px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

.tab-nav button.selected {
    color: var(--accent-cyan) !important;
    background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(0, 102, 255, 0.1)) !important;
    box-shadow: 0 0 20px rgba(0, 212, 255, 0.2), inset 0 1px 0 rgba(0, 212, 255, 0.3) !important;
    border: 1px solid rgba(0, 212, 255, 0.3) !important;
}

/* ── Cards / Blocks ── */
.gradio-group, .gr-group {
    background: var(--bg-card) !important;
    border: 1px solid rgba(0, 212, 255, 0.1) !important;
    border-radius: 20px !important;
    box-shadow: var(--shadow-3d), inset 0 1px 0 rgba(255,255,255,0.03) !important;
    transition: all 0.4s ease !important;
    transform: perspective(1000px) rotateX(0deg);
    position: relative;
    overflow: hidden;
}

.gradio-group::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0, 212, 255, 0.5), transparent);
}

.gradio-group:hover {
    border-color: rgba(0, 212, 255, 0.25) !important;
    box-shadow: var(--shadow-3d), var(--glow-cyan) !important;
    transform: perspective(1000px) translateY(-4px) !important;
}

/* ── Labels ── */
label span, .gr-label {
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    font-size: 11px !important;
    color: var(--accent-cyan) !important;
    opacity: 0.85;
}

/* ── Inputs ── */
input, textarea, select, .gr-input {
    background: rgba(5, 8, 16, 0.8) !important;
    border: 1px solid rgba(0, 212, 255, 0.15) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 13px !important;
    transition: all 0.3s ease !important;
    padding: 12px 16px !important;
}

input:focus, textarea:focus {
    border-color: var(--accent-cyan) !important;
    box-shadow: 0 0 0 3px rgba(0, 212, 255, 0.1), var(--glow-cyan) !important;
    outline: none !important;
    background: rgba(0, 212, 255, 0.03) !important;
}

/* ── Dropdown ── */
.gr-dropdown select, .gradio-dropdown {
    background: rgba(5, 8, 16, 0.9) !important;
    border: 1px solid rgba(0, 212, 255, 0.2) !important;
    border-radius: 12px !important;
    color: var(--accent-cyan) !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
}

/* ── Primary Button ── */
button.primary, .gr-button-primary, button[variant="primary"] {
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    background: linear-gradient(135deg, #0066ff 0%, #00d4ff 50%, #0066ff 100%) !important;
    background-size: 200% 200% !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 32px !important;
    color: #fff !important;
    box-shadow:
        0 8px 32px rgba(0, 102, 255, 0.4),
        0 2px 8px rgba(0, 0, 0, 0.5),
        inset 0 1px 0 rgba(255,255,255,0.2) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    animation: gradientShift 3s ease infinite !important;
    position: relative !important;
    overflow: hidden !important;
}

button.primary::before {
    content: '';
    position: absolute;
    top: -50%; left: -60%;
    width: 40%; height: 200%;
    background: rgba(255,255,255,0.1);
    transform: skewX(-20deg);
    transition: left 0.6s ease;
}

button.primary:hover::before {
    left: 120%;
}

button.primary:hover {
    transform: translateY(-3px) scale(1.02) !important;
    box-shadow:
        0 16px 48px rgba(0, 102, 255, 0.5),
        0 0 30px rgba(0, 212, 255, 0.3),
        inset 0 1px 0 rgba(255,255,255,0.3) !important;
}

button.primary:active {
    transform: translateY(0px) scale(0.98) !important;
}

@keyframes gradientShift {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}

/* ── Output Textboxes — 3D Result Cards ── */
.output-card input, .output-card textarea {
    background: linear-gradient(135deg, rgba(0, 212, 255, 0.05), rgba(0, 102, 255, 0.05)) !important;
    border: 1px solid rgba(0, 212, 255, 0.2) !important;
    border-radius: 14px !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 16px !important;
    font-weight: bold !important;
    color: var(--accent-cyan) !important;
    text-align: center !important;
    box-shadow: inset 0 2px 8px rgba(0,0,0,0.3), 0 0 20px rgba(0, 212, 255, 0.1) !important;
}

/* ── Table / DataFrame ── */
table {
    border-collapse: separate !important;
    border-spacing: 0 4px !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 12px !important;
}

th {
    background: rgba(0, 102, 255, 0.2) !important;
    color: var(--accent-cyan) !important;
    font-family: 'Rajdhani', sans-serif !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    font-size: 11px !important;
    padding: 10px 16px !important;
    border: none !important;
}

td {
    background: rgba(13, 20, 37, 0.6) !important;
    color: var(--text-secondary) !important;
    padding: 8px 16px !important;
    border: none !important;
    border-top: 1px solid rgba(0, 212, 255, 0.05) !important;
    transition: background 0.2s ease !important;
}

tr:hover td {
    background: rgba(0, 212, 255, 0.05) !important;
    color: var(--text-primary) !important;
}

/* ── Markdown ── */
.prose, .markdown {
    color: var(--text-secondary) !important;
    font-family: 'Exo 2', sans-serif !important;
}

.prose h1, .markdown h1 {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 3rem !important;
    font-weight: 700 !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    background: linear-gradient(135deg, #ffffff 0%, var(--accent-cyan) 40%, var(--accent-blue) 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    filter: drop-shadow(0 0 30px rgba(0, 212, 255, 0.3)) !important;
    margin-bottom: 8px !important;
}

.prose h2, .markdown h2 {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1.4rem !important;
    font-weight: 600 !important;
    letter-spacing: 2px !important;
    color: var(--accent-cyan) !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid rgba(0, 212, 255, 0.2) !important;
    padding-bottom: 8px !important;
}

.prose p, .markdown p {
    color: var(--text-secondary) !important;
    line-height: 1.7 !important;
    font-size: 14px !important;
}

.prose strong, .markdown strong {
    color: var(--accent-cyan) !important;
}

/* ── Code blocks ── */
code, pre {
    font-family: 'Share Tech Mono', monospace !important;
    background: rgba(0, 212, 255, 0.05) !important;
    border: 1px solid rgba(0, 212, 255, 0.15) !important;
    border-radius: 8px !important;
    color: var(--accent-cyan) !important;
    font-size: 12px !important;
}

/* ── Examples Table ── */
.examples {
    background: var(--bg-card) !important;
    border: 1px solid rgba(0, 212, 255, 0.1) !important;
    border-radius: 14px !important;
    overflow: hidden !important;
}

.examples table th {
    background: rgba(0, 102, 255, 0.15) !important;
}

/* ── File Upload ── */
.gr-file {
    background: rgba(5, 8, 16, 0.8) !important;
    border: 2px dashed rgba(0, 212, 255, 0.25) !important;
    border-radius: 16px !important;
    transition: all 0.3s ease !important;
}

.gr-file:hover {
    border-color: var(--accent-cyan) !important;
    background: rgba(0, 212, 255, 0.03) !important;
    box-shadow: var(--glow-cyan) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-secondary); }
::-webkit-scrollbar-thumb {
    background: linear-gradient(var(--accent-blue), var(--accent-cyan));
    border-radius: 3px;
}

/* ── Pulsing accent line ── */
@keyframes pulse-glow {
    0%, 100% { opacity: 0.4; box-shadow: 0 0 10px rgba(0,212,255,0.3); }
    50% { opacity: 1; box-shadow: 0 0 30px rgba(0,212,255,0.8); }
}

/* ── Tier badge colors ── */
.tier-regex  { color: #00ff88 !important; }
.tier-bert   { color: #00d4ff !important; }
.tier-llm    { color: #ffd700 !important; }
"""

# ── Functions ───────────────────────────────────────────────────────────────
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
    return label, f"{icon} {tier}", confidence, f"{latency_ms:.1f} ms"


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


# ── UI ───────────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="LOG CLASSIFICATION SYSTEM",
    theme=gr.themes.Base(
        primary_hue="blue",
        secondary_hue="cyan",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Exo 2"), "sans-serif"],
        font_mono=[gr.themes.GoogleFont("Share Tech Mono"), "monospace"],
    ).set(
        body_background_fill="#050810",
        body_text_color="#e2e8f0",
        block_background_fill="#0d1425",
        block_border_color="rgba(0,212,255,0.15)",
        block_label_text_color="#00d4ff",
        input_background_fill="#050810",
        input_border_color="rgba(0,212,255,0.2)",
        button_primary_background_fill="linear-gradient(135deg, #0066ff, #00d4ff)",
        button_primary_text_color="#ffffff",
        border_color_accent="#00d4ff",
        color_accent_soft="rgba(0,212,255,0.1)",
    ),
    css=CUSTOM_CSS
) as demo:

    gr.Markdown("""
# 🔍 LOG CLASSIFICATION SYSTEM
**3-tier hybrid pipeline** — 🟢 Regex · 🔵 BERT + ML · 🟡 LLM  
*Enterprise-grade log monitoring at production scale*
""")

    with gr.Tabs():

        # ── Tab 1: Single Log ─────────────────────────────────────────────
        with gr.Tab("⚡ SINGLE LOG"):
            with gr.Row():
                with gr.Column(scale=1):
                    source_input = gr.Dropdown(
                        choices=SOURCES,
                        value="ModernCRM",
                        label="SOURCE SYSTEM",
                    )
                with gr.Column(scale=3):
                    log_input = gr.Textbox(
                        label="LOG MESSAGE",
                        placeholder="Paste a log message here...",
                        lines=3,
                    )

            classify_btn = gr.Button("▶  CLASSIFY LOG", variant="primary", size="lg")

            with gr.Row():
                label_out      = gr.Textbox(label="🏷️ PREDICTED LABEL",  interactive=False)
                tier_out       = gr.Textbox(label="⚙️  TIER USED",        interactive=False)
                confidence_out = gr.Textbox(label="📈 CONFIDENCE",        interactive=False)
                latency_out    = gr.Textbox(label="⏱️  LATENCY",          interactive=False)

            classify_btn.click(
                fn=classify_single,
                inputs=[source_input, log_input],
                outputs=[label_out, tier_out, confidence_out, latency_out],
            )

            gr.Examples(
                examples=EXAMPLE_LOGS,
                inputs=[source_input, log_input],
                label="📋 EXAMPLE LOGS — click to try",
            )

        # ── Tab 2: Batch CSV ──────────────────────────────────────────────
        with gr.Tab("📦 BATCH CSV"):
            gr.Markdown("""
### Bulk Classification
Upload a CSV with columns: **`source`**, **`log_message`**  
Output includes: `predicted_label`, `tier_used`, `confidence`, `latency_ms`
""")
            with gr.Row():
                with gr.Column():
                    csv_input  = gr.File(label="📂 UPLOAD CSV", file_types=[".csv"])
                    batch_btn  = gr.Button("▶  CLASSIFY ALL", variant="primary")
                with gr.Column():
                    csv_output = gr.File(label="📥 DOWNLOAD RESULTS")
                    stats_out  = gr.Textbox(label="📊 STATISTICS", lines=12, interactive=False)

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

        # ── Tab 3: Architecture ───────────────────────────────────────────
        with gr.Tab("🏗️ ARCHITECTURE"):
            gr.Markdown("""
## 3-Tier Hybrid Pipeline

| Tier | Method | Coverage | Latency | Trigger |
|------|--------|----------|---------|---------|
| 🟢 **Regex** | Python `re` patterns | ~21% | < 1ms | Fixed patterns |
| 🔵 **BERT** | `all-MiniLM-L6-v2` + LogReg | ~79% | 20–80ms | High-volume categories |
| 🟡 **LLM** | HuggingFace Inference API | ~0.3% | 500–2000ms | LegacyCRM + rare patterns |

## Model Performance
- **Training data**: 2,410 synthetic enterprise logs
- **Confidence threshold**: 0.5 (below → escalate to LLM)
- **Source-aware routing**: `LegacyCRM` → LLM directly

## Environment Variables
| Secret | Purpose |
|--------|---------|
| `HF_TOKEN` | LLM inference for LegacyCRM logs |
""")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
