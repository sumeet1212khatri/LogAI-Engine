# ✨ LOG CLASSIFICATION SYSTEM V2 (FULL CODE & OUTPUT)

### Cell 1: Setup & Installation
**Code:**
```python
!pip install -q sentence-transformers scikit-learn pandas numpy \
    matplotlib seaborn joblib huggingface-hub optimum[onnxruntime] \
    onnxruntime onnx

print('✅ Sab install ho gaya!')
```
**Output:**
```text
✅ Sab install ho gaya!
```

---

### Cell 2: 50,000 Logs Generation
**Code:**
```python
# [Dataset generation code from the notebook...]
import random, pandas as pd, numpy as np
# ... templates defined here ...
print(f'✅ Dataset ready: {len(df):,} logs')
print('\nClass distribution:')
print(df['target_label'].value_counts().to_string())
df.to_csv('synthetic_logs_v2.csv', index=False)
print('\n✅ synthetic_logs_v2.csv saved!')
```
**Output:**
```text
✅ Dataset ready: 50,000 logs

Class distribution:
HTTP Status            15000
Security Alert          9000
System Notification     7500
Resource Usage          6000
Critical Error          5000
Error                   4000
User Action             2500
Workflow Error           500
Deprecation Warning      500

✅ synthetic_logs_v2.csv saved!
```

---

### Cell 3: BERT Embeddings Generation
**Code:**
```python
# [Embedding logic...]
print(f'BERT training data: {len(bert_df):,} logs')
# Load model 'all-MiniLM-L6-v2'
# Generate embeddings in batches of 256
print(f'\n✅ Embedding shape: {X.shape}')
```
**Output:**
```text
BERT training data: 49,000 logs
Loading sentence transformer model...
Using device: cuda
✅ Embedding shape: (49000, 384)
```

---

### Cell 4: Model Training (Logistic Regression)
**Code:**
```python
# [Training and Calibration...]
print(f'\n✅ Training done in {train_time:.2f}s')
print(f'  Accuracy : {acc:.4f} ({acc*100:.1f}%)')
```
**Output:**
```text
Train: 41,650  |  Test: 7,350
Training LogisticRegression...
Calibrating probabilities...
✅ Training done in 4.76s
  Accuracy : 1.0000 (100.0%)
Detailed report shows 1.0 precision/recall for all classes.
```

---

### Cell 8: Speed Benchmark
**Code:**
```python
# Benchmark ONNX vs PyTorch batch=64
```
**Output:**
```text
❌ OLD (PyTorch, 1 at a time): 164 logs/s  (6.1ms/log)
✅ PyTorch (batch=64):     3066 logs/s  (0.3ms/log)
  ONNX (batch=64):         129 logs/s  (7.8ms/log)
```

---

### Cell 9: Final Results Table
**Code:**
```python
# Print resume-ready summary
```
**Output:**
```text
╔═══════════════════════════════════════════════════════╗
║     ═ RESUME-READY NUMBERS (V2 — 50k Dataset)        ║
╠═══════════════════════════════════════════════════════╣
║  Dataset:    50,000 records                       ║
║  Accuracy:   100.0%                               ║
║  Speed:      129 logs/s                           ║
╚═══════════════════════════════════════════════════════╝
```

---

### Cell 10: File Downloads
**Code:**
```python
import shutil
from google.colab import files
# Zip and download models and charts
print('Downloading files...')
```
**Output:**
```text
Downloading files...
✅ Downloaded: log_classifier.joblib, onnx_model.zip, etc.
```
