# Bias Reduction Tool with Pre- and Post-Intervention Metrics in Large Language Models (LLMs)

This repository focuses on neutralizing bias in large language models (LLMs) and benchmarking their performance using various metrics. It includes scripts for fine-tuning models, evaluating text quality, and calculating bias-related metrics.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
  - [Fine-Tuning Models](#fine-tuning-models)
  - [Running Benchmarks](#running-benchmarks)
- [Metrics and Evaluation](#metrics-and-evaluation)
- [Requirements](#requirements)

---

## Overview

Bias in LLMs can lead to unfair or harmful outputs. This project aims to:
1. Fine-tune LLMs to reduce bias.
2. Evaluate the effectiveness of debiasing using metrics like WEAT, BLEU, ROUGE, and cosine similarity.
3. Provide tools for benchmarking text quality and bias mitigation.

---

## Repository Structure

```
.
├── .env                     # Environment variables (e.g., API tokens)
├── .gitignore               # Files and directories to ignore in version control
├── dataset100.png           # Example dataset visualization
├── training_eval_loss_plot.png # Training evaluation loss plot
├── requirements.txt         # Python dependencies
├── benchmark/               # Benchmarking scripts
│   ├── cosine_similarity.py # Cosine similarity calculation
│   ├── text_quality_metrics.py # BLEU and ROUGE metrics
│   └── metrics/
│       └── weat.py          # WEAT score calculation
├── scripts/                 # Scripts for fine-tuning and evaluation
│   ├── fine_t5_safellmwithBiasdata.py
│   ├── finetune_t5_safellm.py
│   ├── finetune_t5_wnc.py
│   ├── main.py              # Main script for benchmarking
│   └── runfinetune_t5.py    # Script to run fine-tuned models
└── README.md                # Project documentation
```

---

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/Bias-Mitigation-and-benchmarking-LLM.git
   cd Bias-Mitigation-and-benchmarking-LLM
   ```

2. **Install Dependencies**:
   Use the `requirements.txt` file to install the necessary Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables**:
   Add your Hugging Face token to the `.env` file:
   ```
   TOKEN = "your_huggingface_token"
   ```

4. **Download Required Models**:
   Ensure you have the necessary pre-trained models (e.g., `t5-small`, `t5-large`) downloaded via Hugging Face.

---

## Usage

### Fine-Tuning Models

- **Instruction-Safe LLM Fine-Tuning**:
  Run the script `finetune_t5_safellm.py` to fine-tune a T5 model on the instruction-safe dataset:
  ```bash
  python scripts/finetune_t5_safellm.py
  ```

- **Fine-Tuning with Bias Data**:
  Use `fine_t5_safellmwithBiasdata.py` to fine-tune the model on biased data:
  ```bash
  python scripts/fine_t5_safellmwithBiasdata.py
  ```

- **Fine-Tuning with WNC Dataset**:
  Run `finetune_t5_wnc.py` to fine-tune the model using the WNC dataset:
  ```bash
  python scripts/finetune_t5_wnc.py
  ```

### Running Benchmarks

- Use the `main.py` script to evaluate the fine-tuned models on various metrics:
  ```bash
  python scripts/main.py
  ```

- To test the fine-tuned models with specific inputs, use `runfinetune_t5.py`:
  ```bash
  python scripts/runfinetune_t5.py
  ```

---

## Metrics and Evaluation

The repository includes the following metrics for benchmarking:

1. **WEAT (Word Embedding Association Test)**:
   - Measures bias in word embeddings.
   - Implemented in [`benchmark/metrics/weat.py`](benchmark/metrics/weat.py).

2. **Cosine Similarity**:
   - Measures similarity between biased and neutralized sentences.
   - Implemented in [`benchmark/cosine_similarity.py`](benchmark/cosine_similarity.py).

3. **Text Quality Metrics**:
   - BLEU and ROUGE scores for evaluating text quality.
   - Implemented in [`benchmark/text_quality_metrics.py`](benchmark/text_quality_metrics.py).

---

## Requirements

The project requires the following Python packages install them using:
```bash
pip install -r requirements.txt
```

---


