# Emotions Classifier

A PyTorch-based Transformer model for text emotion classification with data augmentation and training utilities.


## Overview

This project implements an emotion classification pipeline using a Transformer encoder model in PyTorch. It includes:

- A Transformer-based emotion classifier with learnable positional encoding and [CLS] token.
- Data loader with augmentation (synonym replacement and random deletion), tokenization, and padding.
- Training and evaluation handler with checkpointing and learning rate scheduling.


## Features

- Transformer encoder architecture with multi-head attention.
- Text data augmentation for improved generalization.
- Early stopping and learning rate reduction on plateau.
- GPU support (automatic if available).
- Detailed classification reports with precision, recall, and F1-score.
- Checkpoint saving and loading.


## Installation

````bash
pip install torch torchvision torchaudio scikit-learn pandas nltk
````
Also, download NLTK data:

```python
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
```
## Usage

1. Prepare your dataset as lists of texts and emotion labels.

2. Initialize the data loader and preprocess data:

```python
from EmotionsDataLoader import EmotionsDataLoader

data_loader = EmotionsDataLoader(batch_size=32, max_length=20, augment=True)
train_loader, test_loader, vocab_size, vocab = data_loader.preprocess(texts, labels)
```

3. Create the model:

```python
from EmotionClassifierModel import EmotionClassifierModel

model = EmotionClassifierModel(
    vocab_size=vocab_size,
    embed_dim=128,
    num_heads=8,
    hidden_dim=512,
    num_layers=2,
    output_dim=num_classes,
    dropout=0.1,
    pad_idx=0,
    max_length=20
)
```
4. Train and evaluate the model:

```python
from ModelHandler import ModelHandler

handler = ModelHandler(model, train_loader, test_loader, learning_rate=0.002)
handler.train(epochs=30)
handler.test()
```

5. Save/load checkpoints as needed:

```python
handler.load_checkpoint('checkpoints/model_epoch_30.pt')
```
## License
This project is open-source and free for anyone to use, modify, and distribute without restrictions.
