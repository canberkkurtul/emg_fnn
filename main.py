import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import random

i=4
torch.manual_seed(i)
random.seed(i)
np.random.seed(i)

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


def train_model(X, y, d_hidden=10, epochs=20, batch_size=50, lr=1e-3):
    d_in = X.shape[1]
    d_out = 3 #len(torch.unique(y))

    model = FNN(d_in, d_hidden, d_out)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    N = X.shape[0]
    print("number of rows:",N)

    for epoch in range(epochs):
        model.train()

        # shuffle
        perm = torch.randperm(N)
        X_shuffled = X[perm]
        y_shuffled = y[perm]

        epoch_loss = 0.0

        for i in range(0, N, batch_size):
            xb = X_shuffled[i:i+batch_size]
            yb = y_shuffled[i:i+batch_size]

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        #print(f"Epoch {epoch+1}/{epochs}, Loss = {epoch_loss:.4f}")

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


def inference(model):
    df_val = pd.read_excel(r"C:\Users\MonsterPC\Desktop\emgdata\validation_mustafa2903.xlsx")
    X_val = df_val.iloc[:, :-1].values
    y_val = df_val.iloc[:, -1].values

    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)

    model.eval()
    with torch.no_grad():
        logits = model(X_val)                 # shape: (84, 3)
        probs = torch.softmax(logits, dim=1) # shape: (84, 3)
        preds = probs.argmax(dim=1)          # shape: (84,)

        correct = (preds == y_val).sum().item()
        total = y_val.size(0)
        acc = correct / total

    print(f"Correct: {correct}/{total}")
    print(f"Validation Accuracy: {acc*100:.2f}%")


if __name__ == "__main__":
    df = pd.read_excel(r"C:\Users\MonsterPC\Desktop\emgdata\training_mustafa2903.xlsx")
    X = df.iloc[:, :-1].values  # ilk 8 sütun
    y = df.iloc[:, -1].values  # son sütun (TRUECLASS)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    model = train_model(X, y, d_hidden=30, epochs=500, batch_size=28, lr=1e-3)
    inference(model)

    #loaded_model = load_model()
    #inference(loaded_model)
