import torch
import torch.nn as nn
import numpy as np
import time

# ==========================================
# 1. The Blueprint (Must match training exactly)
# ==========================================
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

# ==========================================
# 2. Hardcoded Configuration & Constants
# ==========================================
# Paste the values printed from your training script here
SCALER_MEANS = np.array([53.255090, 108.761576, 5555.115375, 46.622660, 55.072045, 135.951888, 7833.922760, 58.829228, 32.433997, 104.663383, 4681.822660, 56.792775, 54.655308, 84.335304, 5406.336579, 21.514450, 57.587539, 134.921018, 7598.963806, 52.296716, 32.550909, 97.366667, 4413.976601, 36.336782])
SCALER_STDS  = np.array([119.620294, 108.751161, 9561.038613, 27.247587, 132.108750, 107.177609, 17345.592130, 22.907691, 62.107392, 91.364579, 7269.966461, 26.621610, 145.922306, 114.790928, 13598.933797, 13.209324, 122.411753, 111.914161, 13998.152592, 18.020563, 51.719171, 108.762118, 5829.053565, 18.789903])

CLASS_NAMES = [
    "Right biceps", "Right triceps", "Right front arm", 
    "Left biceps", "Left triceps", "Left front arm", "Rest"
]

# Set up device
device = torch.device("cpu") # CPU is usually faster for single-sample real-time inference

# Initialize the model blueprint
# Note: d_in must match your feature count. You have 4 features * 6 channels = 24
D_IN = len(SCALER_MEANS) 
D_HIDDEN = 64
D_OUT = 7

model = FNN(d_in=D_IN, d_hidden=D_HIDDEN, d_out=D_OUT).to(device)

# Load the trained weights
model.load_state_dict(torch.load("emg_fnn_production_model.pth", map_location=device))

# CRITICAL: Put the model in evaluation mode
model.eval()

# ==========================================
# 3. The Real-Time Prediction Function
# ==========================================
def predict_movement(raw_features: list) -> str:
    """
    Takes a raw list of features from your 6 channels, normalizes them, 
    and returns the predicted movement.
    """
    # 1. Convert incoming data to a NumPy array
    features_np = np.array(raw_features, dtype=np.float32)
    
    # 2. Apply Z-score Normalization manually
    # Formula: z = (x - mean) / std
    normalized_features = (features_np - SCALER_MEANS) / SCALER_STDS
    
    # 3. Convert to PyTorch Tensor and add batch dimension
    # Neural networks expect a 2D batch (Batch Size x Features), so [1, 18]
    tensor_input = torch.tensor(normalized_features, dtype=torch.float32).unsqueeze(0).to(device)    
    # 4. Forward Pass (Without tracking gradients to save memory/speed)
    with torch.no_grad():
        logits = model(tensor_input)
        
        # Optional: Get confidence percentages
        probs = torch.softmax(logits, dim=1)[0]
        confidence = torch.max(probs).item() * 100
        
        # Get the predicted class index
        predicted_idx = torch.argmax(logits, dim=1).item()
        
    predicted_label = CLASS_NAMES[predicted_idx]
    
    return predicted_label, confidence

# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    # Simulate receiving a single window of data from your Arduino/Serial
    # This should be a list of 18 numbers (if you dropped SSC on 6 channels)    
    live_data = [
        9.4918, 46, 2324.761, 48, 4.9961, 18, 1423.5453, 24, 
        7.2808, 58, 2368.3328, 65, 254.6053, 323, 18705.0723, 30, 
        85.5674, 264, 8915.835, 37, 85.148, 289, 10778.1318, 49
    ]    # You would put this function call inside your serial reading loop
    # 1. Start the stopwatch
    start_time = time.perf_counter()
    
    # 2. Run the prediction
    movement, conf = predict_movement(live_data)
    
    # 3. Stop the stopwatch
    end_time = time.perf_counter()
    
    # 4. Calculate the difference
    execution_time = (end_time - start_time) * 1000 # Multiply by 1000 to get milliseconds
    
    print(f"Predicted Movement: {movement} (Confidence: {conf:.1f}%)")
    print(f"Inference Time: {execution_time:.3f} milliseconds")