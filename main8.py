import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import random
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

#NOT LOSO VALOIDATION
#EXTRACTS THE MODEL INTO A PTH FILE

# =========================
# Hardware & Reproducibility
# =========================
device = torch.device("cpu") # Change to "mps" if you want to use Apple Silicon acceleration
print(f"Using device: {device}")

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
    X, y = X.to(device), y.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)

    d_in = X.shape[1]
    d_out = len(torch.unique(torch.cat((y, y_val)))) 

    model = FNN(d_in, d_hidden, d_out).to(device)
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

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

    model.load_state_dict(best_state)
    return model

# =========================
# Evaluation
# =========================
def evaluate_model(model, X_val, y_val, class_names):
    X_val, y_val = X_val.to(device), y_val.to(device)
    criterion = nn.CrossEntropyLoss()
    num_classes = len(class_names)

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

        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
        per_class_acc = np.zeros(num_classes)
        
        for i in range(num_classes):
            row_sum = cm[i].sum()
            per_class_acc[i] = cm[i, i] / row_sum if row_sum > 0 else 0.0

        print("\n" + "-"*40)
        print("FINAL EVALUATION ON VALIDATION SUBJECT")
        print("-"*40)
        print("\nConfusion Matrix:")
        print(cm)

        print("\nPer-class accuracy:")
        for i, a in enumerate(per_class_acc):
            label = class_names[i]
            print(f"{label}: {a*100:.2f}%")

        print(f"\nValidation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {acc*100:.2f}%")

# =========================
# Helper: load one subject
# =========================
def load_subject_csv(filepath, features_to_drop=None):
    df = pd.read_csv(filepath)  
    df.columns = [col.strip() for col in df.columns]

    if features_to_drop is not None:
        cols_to_drop = [col for col in df.columns[:-1] if any(drop_feat in col for drop_feat in features_to_drop)]
        df = df.drop(columns=cols_to_drop)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    return X, y

# =========================
# Train for Deployment
# =========================
def train_for_deployment(subject_files, val_subject_key, d_hidden=32, epochs=50, batch_size=128, lr=1e-3, features_to_drop=None):
    class_names = [
        "Right biceps", "Right triceps", "Right front arm", "Left biceps", 
        "Left triceps", "Left front arm", "Rest"
    ]

    print(f"Preparing data. Validation subject will be: {val_subject_key}")
    
    X_train_list, y_train_list = [], []
    X_val, y_val = None, None

    # Load and split data based on the chosen validation subject
    for subject_name, filepath in subject_files.items():
        X_subj, y_subj = load_subject_csv(filepath, features_to_drop=features_to_drop)
        
        if subject_name == val_subject_key:
            X_val, y_val = X_subj, y_subj
            print(f"-> Set aside {subject_name} for Validation ({len(y_subj)} samples)")
        else:
            X_train_list.append(X_subj)
            y_train_list.append(y_subj)
            print(f"-> Added {subject_name} to Training ({len(y_subj)} samples)")

    X_train = torch.cat(X_train_list, dim=0)
    y_train = torch.cat(y_train_list, dim=0)

    # Convert to Numpy for Scaler
    X_train_np = X_train.numpy()
    X_val_np = X_val.numpy()

    # Apply StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_np)
    X_val_scaled = scaler.transform(X_val_np)

    # Convert back to PyTorch Tensors
    X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_val = torch.tensor(X_val_scaled, dtype=torch.float32)

    # Train Model
    print("\nStarting final model training...")
    model = train_model(
        X_train, y_train, X_val, y_val,
        d_hidden=d_hidden,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr
    )

    # Evaluate Model
    evaluate_model(model, X_val, y_val, class_names=class_names)

    # ==========================================
    # DEPLOYMENT EXPORT BLOCK
    # ==========================================
    # 1. Save PyTorch Model Weights
    model_save_path = "emg_fnn_production_model.pth"
    torch.save(model.state_dict(), model_save_path)
    
    # 2. Extract and print C++ arrays for the Scaler
    means = scaler.mean_
    stds = scaler.scale_
    
    print("\n" + "="*50)
    print("DEPLOYMENT ASSETS READY")
    print("="*50)
    print(f"Model weights saved locally to: {model_save_path}")
    print("\nCopy/Paste these Z-Score constants into your C++ real-time environment:\n")
    
    # Print C++ friendly float arrays
    print(f"const float SCALER_MEANS[{len(means)}] = {{")
    print("    " + ", ".join(f"{m:.6f}f" for m in means))
    print("};\n")
    
    print(f"const float SCALER_STDS[{len(stds)}] = {{")
    print("    " + ", ".join(f"{s:.6f}f" for s in stds))
    print("};\n")

    return model, scaler

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

    # Set which subject you want to isolate for the final validation pass
    TARGET_VALIDATION_SUBJECT = "subject7" 

    final_model, final_scaler = train_for_deployment(
        subject_files,
        val_subject_key=TARGET_VALIDATION_SUBJECT,
        d_hidden=64,
        epochs=100,
        batch_size=58,
        lr=1e-2,
        features_to_drop=[]  
    )