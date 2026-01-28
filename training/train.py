import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint

# 1. Setup Robust Paths (Based on file location, not current folder)
# Get the directory where THIS script (train.py) is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the Repo Root (one level up from training/)
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

# Add repo root to path for imports
sys.path.append(REPO_ROOT)
from models.model import create_model

# Define Paths relative to the Repo Root
DATA_PATH = os.path.join(REPO_ROOT, "data", "processed")
RESULTS_PATH = os.path.join(REPO_ROOT, "results")
CHECKPOINT_PATH = os.path.join(REPO_ROOT, "checkpoints")

# Ensure folders exist
os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

print(f"DEBUG: Paths set!")
print(f"  Data: {DATA_PATH}")
print(f"  Results: {RESULTS_PATH}")
print(f"  Checkpoints: {CHECKPOINT_PATH}")

# 2. Load Data
print("Loading data...")
try:
    X = np.load(os.path.join(DATA_PATH, "X.npy"))
    y = np.load(os.path.join(DATA_PATH, "y.npy"))
    print(f"DEBUG: Data Loaded. Shape is {X.shape}") 
except FileNotFoundError:
    print("ERROR: Could not find X.npy or y.npy. Did you run the preprocessing step?")
    sys.exit(1)

# 3. Split Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Create Model
# Dynamically get input shape
input_shape = (X.shape[1], X.shape[2]) 
num_classes = y.shape[1]

print(f"Building Model with Input Shape: {input_shape}")
model = create_model(input_shape, num_classes)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 5. Train
checkpoint = ModelCheckpoint(os.path.join(CHECKPOINT_PATH, "best_model.h5"), 
                             monitor='val_accuracy', save_best_only=True, mode='max')

print("Starting Training...")
history = model.fit(X_train, y_train, epochs=50, batch_size=16, 
                    validation_data=(X_test, y_test), callbacks=[checkpoint])

# 6. Save Graphs
plt.figure()
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy')
plt.legend()
plt.savefig(os.path.join(RESULTS_PATH, 'accuracy_curves.png'))

plt.figure()
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss')
plt.legend()
plt.savefig(os.path.join(RESULTS_PATH, 'loss_curves.png'))

print(f"Success! Model saved to {CHECKPOINT_PATH}")
print(f"Graphs saved to {RESULTS_PATH}")