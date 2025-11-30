# MulAD: A Log-based Anomaly Detection Approach

This repository contains the official implementation of **MulAD** (Multi-Pattern and Multi-Model Fusion), a framework for detecting complex anomalies in distributed systems by fusing five types of log patterns (sequential, semantic, quantitative, temporal, and parametric) and integrating three deep learning models (MABi-LSTM, Transformer, GNN).

# MulAD Quick Reproduction Package

This directory contains the code for reproducing the paper *MulAD: A Log-based Anomaly Detection Approach for Distributed Systems using Multi-Pattern and Multi-Model Fusion*.

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

## One-Click Execution

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate sample logs & train
cd MulAD
python pipeline.py
```

## 🛠️ Requirements
- **Python**: >= 3.8
- **Dependencies**: See `requirements.txt`

To install the required packages:
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start (Reproduction)
To reproduce the experimental pipeline using the sample data provided in the repository (if any) or a specific test file:
```bash
# Run the full pipeline (Parsing -> Feature Extraction -> Training -> Evaluation)
python MulAD_pipeline.py sample_logs.csv
```

## 📖 Usage on Custom Datasets
MulAD is designed to be adaptable to various distributed system logs. To use the tool on your own data ("cases beyond the paper"), please follow these steps:

1. **Data Preparation**  
   Your log data must be formatted as a CSV file containing at least two required columns:  
   - `timestamp`: The time of the log event (e.g., ISO 8601 format or Unix timestamp).  
   - `message`: The raw log content/message.  

   Example (`my_dataset.csv`):
   ```
   timestamp,message
   2023-10-01T12:00:01,User 102 logged in from 192.168.1.10
   2023-10-01T12:00:02,Process started with job_id=99
   2023-10-01T12:00:05,Error: Connection failed to db_shard_01
   ...
   ```

2. **Execution**  
   Run the main pipeline script pointing to your dataset path:
   ```bash
   python MulAD_pipeline.py path/to/your/my_dataset.csv
   ```

3. **Outputs**  
   After execution, the results will be saved in the `output/` directory:  
   - `anomaly_scores.csv`: The predicted anomaly probability for each log sequence.  
   - `metrics.txt`: Evaluation metrics (Precision, Recall, F1-score, AUC) calculated on the test split.  
   - `*.pt` / `*.pkl`: Trained model checkpoints (MABi-LSTM, Transformer, GNN, and Random Forest).

## ⚙️ Configuration & Parameters
You can fine-tune the model performance and behavior by modifying the parameters directly in the Python scripts.

1. **Log Parsing & Interleaving (`MulAD_log_parser.py`)**  
   These parameters control how logs are grouped into tasks based on parameter aliasing.  
   - `alpha` (Default: 3): The minimum number of overlapping values required to identify two parameters as aliases. Increase this for stricter matching.  
   - `beta` (Default: 10): The minimum string length for a value to be considered valid for alias analysis. Increase this to filter out short, trivial tokens.  

   To modify, edit the `alias_merge` function definition in `MulAD_log_parser.py`.

2. **Model Training (`MulAD_pipeline.py`)**  
   These parameters control the deep learning model architecture and training process.  
   - `SEQ_LEN` (Default: 10): The window size ($w$) for constructing log sequences. This determines how many log events are considered in one sequence.  
   - `EPOCHS` (Default: 5): The number of training epochs for the neural networks.  
   - `BATCH_SIZE` (Default: 32): The batch size used during training and inference.  
   - `HIDDEN_DIM` (Default: 128): The size of the hidden layers in MABi-LSTM, Transformer, and GNN.  
   - `LR` (Default: 1e-3): Learning rate for the Adam optimizer.  

   To modify, edit the global variables at the top of `MulAD_pipeline.py`.

## 📂 Project Structure
- `MulAD_pipeline.py`: The main entry point. Orchestrates the entire workflow: data loading, parsing, feature extraction, model training, and evaluation.  
- `MulAD_log_parser.py`: Handles log serialization. Implements the Drain parser and the parameter-based log grouping logic (using alias identification).  
- `MulAD_feature_extractor.py`: Extracts the five log patterns (Sequential, Semantic via BERT, Quantitative, Temporal, Parametric) and fuses them into a synthesized pattern.  
- `MulAD_models_*.py`:  
  - `mabi_lstm.py`: Implementation of the MABi-LSTM model.  
  - `transformer.py`: Implementation of the Transformer model.  
  - `gnn.py`: Implementation of the Graph Neural Network model.  
  - `rf_trainer.py`: Implementation of the Random Forest ensemble integration.  
- `MulAD_utils.py`: Helper functions for file I/O and time conversion.

