import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import random
from sklearn.metrics import confusion_matrix

#SSC ler atılarak bakılıyor
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
    d_in = X.shape[1]
    d_out = len(torch.unique(y))

    model = FNN(d_in, d_hidden, d_out)
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

        print(
            f"Epoch {epoch+1:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc*100:.2f}%"
        )

    model.load_state_dict(best_state)
    return model

# =========================
# Evaluation
# =========================
def evaluate_model(model, X_val, y_val, class_names=None):
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

        y_true = y_val.cpu().numpy()
        y_pred = preds.cpu().numpy()

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

        per_class_acc = np.zeros(3)
        for i in range(3):
            row_sum = cm[i].sum()
            per_class_acc[i] = cm[i, i] / row_sum if row_sum > 0 else 0.0

        print("Confusion Matrix:")
        print(cm)

        print("Per-class accuracy:")
        for i, a in enumerate(per_class_acc):
            label = class_names[i] if class_names is not None else f"Class {i}"
            print(f"{label}: {a*100:.2f}%")

        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Correct: {correct}/{total}")
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
def load_subject_excel(filepath):
    df = pd.read_excel(filepath)

    # SSC featurelerini çıkar
    cols_to_drop = []#["SSC1","WA2","WL2","SSC2"]

    # eğer büyük-küçük harf farkı olursa diye güvenli hale getiriyoruz
    df.columns = [col.strip() for col in df.columns]

    existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    df = df.drop(columns=existing_cols_to_drop)

    # son sütun TRUECLASS olacak şekilde
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    return X, y

# =========================
# LOSO Cross Validation
# =========================
def loso_cross_validation(subject_files, d_hidden=32, epochs=50, batch_size=128, lr=1e-3):
    class_names = ["biceps contraction", "triceps contraction", "rest"]

    subject_data = {}
    for subject_name, filepath in subject_files.items():
        X_subj, y_subj = load_subject_excel(filepath)
        subject_data[subject_name] = (X_subj, y_subj)

        unique, counts = np.unique(y_subj.numpy(), return_counts=True)
        print(f"\nLoaded {subject_name}")
        print(f"Path: {filepath}")
        print(f"Samples: {len(y_subj)}")
        print(f"Feature dimension: {X_subj.shape[1]}")
        print(f"Class distribution: {dict(zip(unique, counts))}")

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

        print(f"Training subjects: {[s for s in subject_names if s != val_subject]}")
        print(f"Train size: {X_train.shape[0]}")
        print(f"Validation size: {X_val.shape[0]}")
        print(f"Number of features: {X_train.shape[1]}")

        unique_train, counts_train = np.unique(y_train.numpy(), return_counts=True)
        unique_val, counts_val = np.unique(y_val.numpy(), return_counts=True)

        print(f"Train class distribution: {dict(zip(unique_train, counts_train))}")
        print(f"Val class distribution: {dict(zip(unique_val, counts_val))}")

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
        "subject1": r"/Users/canberkkurtul/Desktop/training_1104/berkay_training_1104.xlsx",
        "subject2": r"/Users/canberkkurtul/Desktop/training_1104/canberk_training_1104.xlsx",
        "subject3": r"/Users/canberkkurtul/Desktop/training_1104/emre_training_1104.xlsx",
        "subject4": r"/Users/canberkkurtul/Desktop/training_1104/yigit_training_1104.xlsx",
        "subject5": r"/Users/canberkkurtul/Desktop/training_1104/zeynep_training_1104.xlsx"
    }

    results = loso_cross_validation(
        subject_files,
        d_hidden=32,
        epochs=100,
        batch_size=64,
        lr=1e-2
    )