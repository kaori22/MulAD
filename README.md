# MulAD

# MulAD Quick Reproduction Package

This directory contains the code for reproducing the paper *MulAD: A Log-based Anomaly Detection Approach for Distributed Systems using Multi-Pattern and Multi-Model Fusion*.

## One-Click Execution

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate sample logs & train
cd MulAD
python pipeline.py
```

## Directory Structure

```
MulAD/
├── pipeline.py          # End-to-end pipeline: data → training → evaluation
├── log_parser.py        # Log parsing + serialization (including alias merging)
├── feature_extractor.py # Extraction of five types of features
├── models/
│   ├── mabi_lstm.py     # MABi-LSTM model
│   ├── transformer.py   # Transformer model
│   ├── gnn.py           # Simplified GNN model
│   └── rf_trainer.py    # Random Forest ensemble
├── utils.py             # General utility functions
```

