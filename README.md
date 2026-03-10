# Gemma-2b Fine-Tuning with QLoRA

This repository contains a comprehensive Python implementation for fine-tuning Google's **Gemma-2b** Large Language Model. It demonstrates how to use **QLoRA (Quantized Low-Rank Adaptation)** to transform a base model into a specialized instruction-following assistant using the **Databricks Dolly-15k** dataset.



## Project Overview
Base LLMs often produce "hallucinated" or irrelevant content when asked specific questions. This project solves that by:
* **Quantization**: Loading the model in 4-bit precision to save VRAM.
* **PEFT (Parameter-Efficient Fine-Tuning)**: Training only a small subset of "adapter" weights (LoRA) instead of the full 2 billion parameters.
* **Instruction Tuning**: Training on the Dolly-15k dataset to improve response relevance and structure.

---

## 🛠️ Technical Stack
* **Model**: [Google Gemma-2b](https://huggingface.co/google/gemma-2b)
* **Frameworks**: `Transformers`, `PEFT`, `TRL`, `BitsAndBytes`
* **Compute**: `PyTorch` with CUDA (GPU acceleration)
* **Dataset**: `databricks/databricks-dolly-15k`

---

## 📋 Prerequisites
1.  **Hugging Face Access**: You must accept the license for Gemma-2b on Hugging Face.
2.  **API Token**: Set your `HF_TOKEN` as an environment variable.
3.  **Hardware**: An NVIDIA GPU with at least 8GB of VRAM (or use Google Colab with a T4 GPU).

---

## 🔧 Installation
```bash
pip install torch transformers peft trl bitsandbytes accelerate datasets jsonlines
