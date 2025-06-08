import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import os

class ModelHandler:
    def __init__(self, model, train_loader, test_loader, learning_rate=0.002):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.learning_rate = learning_rate
        self.device = self.get_device()

        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', patience=2, factor=0.5, verbose=True)


    def get_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, epochs, checkpoint_dir='checkpoints'):
        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            correct_predictions = 0
            total_predictions = 0

            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()

                loss.backward()
                self.optimizer.step()

                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)

            avg_loss = running_loss / len(self.train_loader)
            accuracy = (correct_predictions * 100) / total_predictions

            val_accuracy = self.evaluate(self.test_loader) 
            self.scheduler.step(val_accuracy)

            if epoch == 0 or (epoch + 1) % 5 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch [{epoch+1}/{epochs}], Accuracy: {accuracy:.2f} %, Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")

            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pt')
                torch.save(self.model.state_dict(), checkpoint_path)


    def test(self):
        self.model.eval()
        correct_predictions = 0
        total_predictions = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = (correct_predictions * 100) / total_predictions
        print(f"Test Accuracy: {accuracy:.2f} %")
        print("\nClassification Report:\n")
        print(classification_report(all_labels, all_preds))

    def load_checkpoint(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        print(f"Loaded model weights from {checkpoint_path}")


    def evaluate(self, data_loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        accuracy = (correct * 100) / total
        self.model.train()
        return accuracy

