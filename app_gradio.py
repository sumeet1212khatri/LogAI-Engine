"""
Log Classification System — HuggingFace Spaces
Ultra-Modern 3D UI | Optimized for Gradio 6.0 & HF Free Tier
"""
from __future__ import annotations
import io
import time
import uuid
import pandas as pd
import numpy as np
import gradio as gr
from classify import classify_log, classify_csv
from processor_bert import preload_models

# ── Preload models (Start loading BERT into RAM immediately) ──
preload_models()

SOURCES = [
    "ModernCRM", "ModernHR", "BillingSystem",
    "AnalyticsEngine", "ThirdPartyAPI", "LegacyCRM",
]

def get_tier_icon(tier_name: str) -> str:
    if "Regex" in tier_name: return "🟢"
    if "BERT" in tier_name: return "🔵"
    if "Cache Hit" in tier_name: return "⚡"
    if "fallback" in tier_name: return "🟠"
    if "LLM" in tier_name: return "🟡"
    return "⚪"

EXAMPLE_LOGS = [
    ["ModernCRM",       "User User12345 logged in."],
    ["ModernHR",        "Multiple login failures occurred on user 6454 account"],
    ["BillingSystem",   "GET /v2/servers/detail HTTP/1.1 status: 200 len: 1583 time: 0.19"],
    ["AnalyticsEngine", "System crashed due to disk I/O failure on node-3"],
    ["LegacyCRM",       "The 'BulkEmailSender' feature will be deprecated in v5.0."],
]

# ── Custom CSS ────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@600;700&family=Share+Tech+Mono&family=Exo+2:wght@400;600&display=swap');

:root {
    --bg-primary: #050810;
    --accent-cyan: #00d4ff;
    --text-primary: #e2e8f0;
}

body, .gradio-container { 
    background: var(--bg-primary) !important; 
    font-family: 'Exo 2', sans-serif !important; 
}

.gradio-group { 
    background: #0d1425 !important; 
    border: 1px solid rgba(0, 212, 255, 0.1) !important; 
    border-radius: 20px !important; 
    box-shadow: 0 10px 30px rgba(0,0,0,0.5) !important;
}

button.primary {
    background: linear-gradient(135deg, #0066ff, #00d4ff) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    letter-spacing: 1.5px !important;
    box-shadow: 0 4px 15px rgba(0, 102, 255, 0.4) !important;
    transition: all 0.2s ease !important;
}

button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(0, 212, 255, 0.5) !important;
}

.output-stats textarea {
    font-family: 'Share Tech Mono', monospace !important;
    background: #050810 !important;
    color: #00ff88 !important;
}
"""

# ── Functions ────────────────────────────────────────────────

def classify_single(source: str, log_message: str):
    from processor_bert import _model_ready
    if not log_message.strip():
        return "—", "—", "—", "—"
    if not _model_ready:
        return "⏳ Loading...", "Warming up", "—", "—"
    
    t0 = time.perf_counter()
    try:
        result = classify_log(source, log_message)
        latency = (time.perf_counter() - t0) * 1000
        icon = get_tier_icon(result["tier"])
        return (
            result["label"], 
            f"{icon} {result['tier']}", 
            f"{result['confidence']:.1%}" if result["confidence"] else "N/A", 
            f"{latency:.4f} ms" 
        )
    except Exception as e:
        return f"Error: {str(e)}", "Fail", "—", "—"

def classify_batch(file, progress=gr.Progress(track_tqdm=True)):
    if file is None: return None, "⚠️ Please upload a CSV file."
    
    progress(0, desc="🚀 Initializing Engine...")
    t0 = time.perf_counter()
    
    try:
        # FIX: Generate a unique output path per user to prevent data bleeding
        unique_id = uuid.uuid4().hex
        safe_output_path = f"/tmp/classified_output_{unique_id}.csv"
        
        output_path, df = classify_csv(file.name, safe_output_path)
        total_time_sec = time.perf_counter() - t0
        
        progress(0.9, desc="📊 Calculating Metrics...")
        
        total = len(df)
        label_counts = df["predicted_label"].value_counts().to_dict()
        tier_counts = df["tier_used"].value_counts().to_dict()
        
        tier_lines = []
        for tier, count in tier_counts.items():
            tier_df = df[df["tier_used"] == tier]
            lats = tier_df["latency_ms"].dropna()
            icon = get_tier_icon(tier)
            pct = count / total
            
            if "BERT" in tier:
                total_ms = lats.sum()
                tier_lines.append(f"  {icon} {tier}: Batch Latency {total_ms:.1f} ms (Over {count} logs)")
            elif "Regex" in tier:
                p50 = np.percentile(lats, 50) if not lats.empty else 0
                tier_lines.append(f"  {icon} {tier}: < 0.1 ms (p50: {p50:.4f} ms) | {count} logs ({pct:.0%})")
            else:
                p50 = np.percentile(lats, 50) if not lats.empty else 0
                p95 = np.percentile(lats, 95) if not lats.empty else 0
                p99 = np.percentile(lats, 99) if not lats.empty else 0
                tier_lines.append(f"  {icon} {tier}: {count} logs ({pct:.0%}) | p50={p50:.1f}ms p95={p95:.1f}ms p99={p99:.1f}ms")
        
        tier_lines_str = "\n".join(tier_lines)
        label_lines = "\n".join([f"  • {k}: {v}" for k, v in label_counts.items()])
        
        stats = (
            f"✅ Classified {total} logs in {total_time_sec:.2f} s\n\n"
            f"📊 Performance by Tier:\n{tier_lines_str}\n\n"
            f"🏷️ Label distribution:\n{label_lines}"
        )
        
        progress(1.0, desc="✅ Success")
        return output_path, stats

    except Exception as e:
        return None, f"❌ System Error: {str(e)}"

# ── Theme & Layout ──────────────────────────────────────────
THEME = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="cyan",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Exo 2")],
)

with gr.Blocks(title="Log AI Engine") as demo:
    gr.HTML("<div style='text-align: center; padding: 20px;'><h1>🔍 LOG CLASSIFICATION SYSTEM</h1></div>")

    with gr.Tabs():
        with gr.Tab("⚡ REAL-TIME ANALYZER"):
            with gr.Row():
                with gr.Column(scale=1):
                    src_in = gr.Dropdown(choices=SOURCES, value="ModernCRM", label="SOURCE")
                with gr.Column(scale=3):
                    msg_in = gr.Textbox(label="LOG MESSAGE", placeholder="Paste raw log string...", lines=3)
            
            run_btn = gr.Button("▶ CLASSIFY LOG", variant="primary")
            
            with gr.Row():
                lbl_out = gr.Textbox(label="PREDICTED LABEL")
                tier_out = gr.Textbox(label="TIER USED")
                conf_out = gr.Textbox(label="CONFIDENCE")
                lat_out = gr.Textbox(label="LATENCY")

            run_btn.click(classify_single, [src_in, msg_in], [lbl_out, tier_out, conf_out, lat_out])
            gr.Examples(examples=EXAMPLE_LOGS, inputs=[src_in, msg_in])

        with gr.Tab("📦 BATCH PROCESSING"):
            with gr.Row():
                with gr.Column():
                    csv_in = gr.File(label="UPLOAD CSV", file_types=[".csv"])
                    batch_btn = gr.Button("▶ START BATCH PROCESS", variant="primary")
                with gr.Column():
                    csv_out = gr.File(label="DOWNLOAD CLASSIFIED DATA")
                    stats_out = gr.Textbox(label="PIPELINE ANALYTICS", lines=16, elem_classes="output-stats")
            
            batch_btn.click(classify_batch, inputs=[csv_in], outputs=[csv_out, stats_out])

demo.queue(default_concurrency_limit=2).launch(
    server_name="0.0.0.0",
    server_port=7860,
    theme=THEME,
    css=CUSTOM_CSS
)
