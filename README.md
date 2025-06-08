# Emotions Classifier

A PyTorch-based Transformer model for text emotion classification with data augmentation and training utilities.

---

## Overview

This project implements an emotion classification pipeline using a Transformer encoder model in PyTorch. It includes:

- A Transformer-based emotion classifier with learnable positional encoding and [CLS] token.
- Data loader with augmentation (synonym replacement and random deletion), tokenization, and padding.
- Training and evaluation handler with checkpointing and learning rate scheduling.

---

## Features

- Transformer encoder architecture with multi-head attention.
- Text data augmentation for improved generalization.
- Early stopping and learning rate reduction on plateau.
- GPU support (automatic if available).
- Detailed classification reports with precision, recall, and F1-score.
- Checkpoint saving and loading.

---

## Installation

```bash
pip install torch torchvision torchaudio scikit-learn pandas nltk

Also, download NLTK data:

```python
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
