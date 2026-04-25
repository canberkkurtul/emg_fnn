import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import random
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

#6 CHANNEL 7 OUTPUT CLASS
#NORMALIZATION
#caffeinate -i /Users/canberkkurtul/emg_fnn/emg_fnn_venv/bin/python /Users/canberkkurtul/emg_fnn/main6.py
device = torch.device("cpu")
print(f"Using device: {device}")

# =========================
# Reproducibility
# =========================
seed = 18
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =========================
# Model
# =========================
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

# =========================
# Training
# =========================
def train_model(X, y, X_val, y_val, d_hidden=32, epochs=50, batch_size=128, lr=1e-3):
    # Move data to GPU/MPS
    X, y = X.to(device), y.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)

    d_in = X.shape[1] # Automatically detects 24 features
    # Dynamically find the total number of unique classes across both train and val
    d_out = len(torch.unique(torch.cat((y, y_val)))) 
   

    model = FNN(d_in, d_hidden, d_out).to(device) # Move model to GPU/MPS
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    N = X.shape[0]
    best_val_acc = -1.0
    best_state = None

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
            val_acc = (preds_val == y_val).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Print every 10 epochs to keep the terminal clean but show progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

    model.load_state_dict(best_state)
    return model

# =========================
# Evaluation
# =========================
def evaluate_model(model, X_val, y_val, class_names):
    # Move val data to GPU/MPS
    X_val, y_val = X_val.to(device), y_val.to(device)
    criterion = nn.CrossEntropyLoss()
    num_classes = len(class_names) # Dynamically handles 7 classes

    model.eval()
    with torch.no_grad():
        logits = model(X_val)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        val_loss = criterion(logits, y_val).item()
        correct = (preds == y_val).sum().item()
        total = y_val.size(0)
        acc = correct / total

        y_true = y_val.cpu().numpy()
        y_pred = preds.cpu().numpy()

        # Update: labels parameter dynamically handles 0 through 6
        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

        # Update: Per-class accuracy loop now iterates over num_classes (7)
        per_class_acc = np.zeros(num_classes)
        for i in range(num_classes):
            row_sum = cm[i].sum()
            per_class_acc[i] = cm[i, i] / row_sum if row_sum > 0 else 0.0

        print("\nConfusion Matrix:")
        print(cm)

        print("\nPer-class accuracy:")
        for i, a in enumerate(per_class_acc):
            label = class_names[i]
            print(f"{label}: {a*100:.2f}%")

        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {acc*100:.2f}%")

    return {
        "loss": val_loss,
        "acc": acc,
        "cm": cm,
        "per_class_acc": per_class_acc
    }

# =========================
# Helper: load one subject
# =========================
def load_subject_csv(filepath, features_to_drop=None):
    df = pd.read_csv(filepath)  
    df.columns = [col.strip() for col in df.columns]

    # ---> NEW: Drop specific features <---
    if features_to_drop is not None:
        # Find all columns that contain the target string (except the TRUECLASS label)
        cols_to_drop = [col for col in df.columns[:-1] if any(drop_feat in col for drop_feat in features_to_drop)]
        df = df.drop(columns=cols_to_drop)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    return X, y

# =========================
# LOSO Cross Validation
# =========================
def loso_cross_validation(subject_files, d_hidden=32, epochs=50, batch_size=128, lr=1e-3, features_to_drop=None):
    # UPDATE THIS LIST: Replaced the 3 classes with 7 generic classes. 
    # Change these strings to match your actual 7 movement names!
    class_names = [
        "Right biceps", "Right triceps", "Right front arm", "Left biceps", 
        "Left triceps", "Left front arm", "Rest"
    ]

    subject_data = {}
    for subject_name, filepath in subject_files.items():
        # ---> NEW: Pass features_to_drop down to the loader <---
        X_subj, y_subj = load_subject_csv(filepath, features_to_drop=features_to_drop)
        subject_data[subject_name] = (X_subj, y_subj)

        unique, counts = np.unique(y_subj.numpy(), return_counts=True)
        print(f"Loaded {subject_name} | Features: {X_subj.shape[1]} | Samples: {len(y_subj)}")

    fold_results = []
    subject_names = list(subject_files.keys())

    for val_subject in subject_names:
        print("\n" + "="*70)
        print(f"LOSO Fold | Validation Subject: {val_subject}")
        print("="*70)

        X_val, y_val = subject_data[val_subject]

        X_train_list = []
        y_train_list = []

        for train_subject in subject_names:
            if train_subject != val_subject:
                X_s, y_s = subject_data[train_subject]
                X_train_list.append(X_s)
                y_train_list.append(y_s)

        X_train = torch.cat(X_train_list, dim=0)
        y_train = torch.cat(y_train_list, dim=0)

        # 1. Convert PyTorch tensors temporarily back to numpy arrays
        X_train_np = X_train.numpy()
        X_val_np = X_val.numpy()

        # 2. Initialize the scaler
        scaler = StandardScaler()

        # 3. Fit ONLY on training data, then transform both train and val
        X_train_scaled = scaler.fit_transform(X_train_np)
        X_val_scaled = scaler.transform(X_val_np)

        # 4. Convert back to PyTorch tensors
        X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
        X_val = torch.tensor(X_val_scaled, dtype=torch.float32)

        model = train_model(
            X_train, y_train, X_val, y_val,
            d_hidden=d_hidden,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr
        )

        result = evaluate_model(model, X_val, y_val, class_names=class_names)
        result["val_subject"] = val_subject
        fold_results.append(result)

    print("\n" + "#"*70)
    print("FINAL LOSO SUMMARY")
    print("#"*70)

    all_accs = [r["acc"] for r in fold_results]
    all_losses = [r["loss"] for r in fold_results]
    all_per_class = np.array([r["per_class_acc"] for r in fold_results])

    for r in fold_results:
        print(f"{r['val_subject']}: Acc = {r['acc']*100:.2f}%, Loss = {r['loss']:.4f}")

    print("\nOverall Results:")
    print(f"Mean Accuracy: {np.mean(all_accs)*100:.2f}%")
    print(f"Std Accuracy : {np.std(all_accs)*100:.2f}%")
    print(f"Mean Loss    : {np.mean(all_losses):.4f}")

    print("\nMean Per-Class Accuracy Across Folds:")
    for i, cname in enumerate(class_names):
        print(f"{cname}: {np.mean(all_per_class[:, i])*100:.2f}%")

    return fold_results

# =========================
# Main
# =========================
if __name__ == "__main__":
    subject_files = {
        "subject1": r"/Users/canberkkurtul/Desktop/new_training_2304/berkay_combined_2304.csv",
        "subject2": r"/Users/canberkkurtul/Desktop/new_training_2304/canberk_combined_2304.csv",
        "subject3": r"/Users/canberkkurtul/Desktop/new_training_2304/ecem_combined_2304.csv",
        "subject4": r"/Users/canberkkurtul/Desktop/new_training_2304/ekin_combined_2304.csv",
        "subject5": r"/Users/canberkkurtul/Desktop/new_training_2304/emre_combined_2304.csv",
        "subject6": r"/Users/canberkkurtul/Desktop/new_training_2304/yigit_combined_2304.csv",
        "subject7": r"/Users/canberkkurtul/Desktop/new_training_2304/zeynep_combined_2304.csv"
    }

    results = loso_cross_validation(
        subject_files,
        d_hidden=64,
        epochs=100,
        batch_size=58,
        lr=1e-2,
        features_to_drop=["SSC"]  # <--- NEW: List any feature names you want to drop here!
    )