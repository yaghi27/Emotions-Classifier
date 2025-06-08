import torch
import random
import nltk
import pandas as pd
from nltk.corpus import wordnet
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder

# Download required nltk resources
nltk.download('wordnet')
nltk.download('omw-1.4')

class EmotionsDataLoader:
    def __init__(self, batch_size=32, max_length=20, augment=True):
        self.batch_size = batch_size
        self.max_length = max_length
        self.vocab = {'<pad>': 0}
        self.augment = augment

    def preprocess(self, texts, labels):
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)

        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.25, random_state=42
        )

        # Apply data augmentation to training texts 
        if self.augment:
            augmented_texts = []
            augmented_labels = []
            for text, label in zip(X_train, y_train):
                augmented_texts.append(text)  # Original text
                augmented_labels.append(label)  # Original label
                augmented_texts.append(self.augment_text(text))  # Augmented text 1
                augmented_labels.append(label)  # Augmented label 1
            X_train = augmented_texts
            y_train = augmented_labels

        X_train, X_test = self.tokenize_data(X_train, X_test)
        X_train = self.pad_sequences(X_train)
        X_test = self.pad_sequences(X_test)

        train_data = TensorDataset(torch.tensor(X_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.long))
        test_data = TensorDataset(torch.tensor(X_test, dtype=torch.long), torch.tensor(y_test, dtype=torch.long))

        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader, len(self.vocab), self.vocab

    def augment_text(self, text, alpha_sr=0.1, alpha_rd=0.1):
        words = text.split()
        if len(words) == 0:
            return text

        # Synonym Replacement
        new_words = []
        for word in words:
            if random.uniform(0, 1) < alpha_sr:
                synonyms = self.get_synonyms(word)
                if synonyms:
                    new_word = random.choice(synonyms)
                    new_words.append(new_word)
                else:
                    new_words.append(word)
            else:
                new_words.append(word)

        # Random Deletion
        if len(new_words) > 1:
            new_words = [word for word in new_words if random.uniform(0, 1) > alpha_rd]

        return " ".join(new_words) if new_words else text

    def get_synonyms(self, word): 
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ').lower()
                if synonym != word.lower():
                    synonyms.add(synonym)
        return list(synonyms)

    def tokenize_data(self, X_train, X_test): # Convert characters to numbers for the model to undesrtand
        idx = 1
        for text in X_train:
            for word in text.split():
                if word not in self.vocab:
                    self.vocab[word] = idx
                    idx += 1

        def text_to_indices(text):
            return [self.vocab.get(word, 0) for word in text.split()]

        X_train_tokenized = [text_to_indices(text) for text in X_train]
        X_test_tokenized = [text_to_indices(text) for text in X_test]

        return X_train_tokenized, X_test_tokenized

    def pad_sequences(self, sequences): # Fill shorter tockinezed sentences with 0s to have same length for training
        padded_sequences = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(seq, dtype=torch.long) for seq in sequences],
            batch_first=True,
            padding_value=0
        )

        if padded_sequences.size(1) > self.max_length:
            padded_sequences = padded_sequences[:, :self.max_length]

        return padded_sequences
