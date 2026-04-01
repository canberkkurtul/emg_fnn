import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import random

#train_model(X, y, X_val, y_val, d_hidden=32, epochs=50, batch_size=500, lr=1e-3) i=16
seed = 16
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

class FNN(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


def train_model(X, y, X_val, y_val, d_hidden=10, epochs=20, batch_size=50, lr=1e-3):
    d_in = X.shape[1]
    d_out = 3

    model = FNN(d_in, d_hidden, d_out)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    N = X.shape[0]

    for epoch in range(epochs):
        model.train()

        perm = torch.randperm(N)
        X_shuffled = X[perm]
        y_shuffled = y[perm]

        epoch_loss = 0.0
        num_batches = 0

        for j in range(0, N, batch_size):
            xb = X_shuffled[j:j+batch_size]
            yb = y_shuffled[j:j+batch_size]

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        train_loss = epoch_loss / num_batches

        model.eval()
        with torch.no_grad():
            logits_val = model(X_val)
            val_loss = criterion(logits_val, y_val).item()

            preds_val = torch.argmax(logits_val, dim=1)
            correct_val = (preds_val == y_val).sum().item()
            total_val = y_val.size(0)
            val_acc = correct_val / total_val

        print(
            f"Epoch {epoch+1}/{epochs}, "
            f"Train Loss = {train_loss:.4f}, "
            f"Val Loss = {val_loss:.4f}, "
            f"Val Acc = {val_acc*100:.2f}%"
        )

    return model


def load_model():
    d_in = 5
    d_hidden = 10
    d_out = 3

    model = FNN(d_in, d_hidden, d_out)
    model.load_state_dict(torch.load("fnn_model.pth"))
    model.eval()
    print("Model loaded.")
    return model


def inference(model, X_val, y_val):
    criterion = nn.CrossEntropyLoss()

    model.eval()
    with torch.no_grad():
        logits = model(X_val)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        val_loss = criterion(logits, y_val).item()
        correct = (preds == y_val).sum().item()
        total = y_val.size(0)
        acc = correct / total

        cm = np.zeros((3, 3), dtype=int)

        for t, p in zip(y_val.cpu().numpy(), preds.cpu().numpy()):
            cm[t, p] += 1

        print("Confusion Matrix:")
        print(cm)

        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        print("Per-class accuracy:")
        for cls, acc_cls in enumerate(per_class_acc):
            print(f"Class {cls}: {acc_cls*100:.2f}%")

        print(f"Final Validation Loss: {val_loss:.4f}")
        print(f"Correct: {correct}/{total}")
        print(f"Validation Accuracy: {acc*100:.2f}%")



if __name__ == "__main__":
    #df = pd.read_csv(r"/Users/canberkkurtul/Desktop/emgdata/3103/training_normalized_bysubject.csv")
    df= pd.read_excel(r"/Users/canberkkurtul/Desktop/emgdata/3103/training_data.xlsx")
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
  

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    #df_val = pd.read_csv(r"/Users/canberkkurtul/Desktop/emgdata/3103/validation_normalized.csv")
    df_val = pd.read_excel(r"/Users/canberkkurtul/Desktop/emgdata/3103/validation_data.xlsx")

    X_val = df_val.iloc[:, :-1].values
    y_val = df_val.iloc[:, -1].values
    #print(np.unique(y_val, return_counts=True))

    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)

    model = train_model(X, y, X_val, y_val, d_hidden=32, epochs=50, batch_size=500, lr=1e-3)
    inference(model, X_val, y_val)

    # torch.save(model.state_dict(), "fnn_model.pth")
    # loaded_model = load_model()
    # inference(loaded_model, X_val, y_val)