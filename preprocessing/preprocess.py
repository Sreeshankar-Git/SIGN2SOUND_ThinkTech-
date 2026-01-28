import os
import numpy as np
from tensorflow.keras.utils import to_categorical

# 1. Setup Robust Paths (Matches train.py)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

RAW_DATA_PATH = "/content/data" # Your raw data
PROCESSED_PATH = os.path.join(REPO_ROOT, "data", "processed") # Target folder
MAX_FRAMES = 100
NUM_FEATURES = 126 

# Ensure save folder exists
os.makedirs(PROCESSED_PATH, exist_ok=True)
print(f"DEBUG: Saving data to {PROCESSED_PATH}")

def load_and_process():
    sequences = []
    labels = []
    label_map = {}
    
    # Handle data folder structure
    search_path = os.path.join(RAW_DATA_PATH, "data") if "data" in os.listdir(RAW_DATA_PATH) else RAW_DATA_PATH

    classes = sorted([d for d in os.listdir(search_path) if os.path.isdir(os.path.join(search_path, d))])
    
    for i, class_name in enumerate(classes):
        label_map[class_name] = i
        class_path = os.path.join(search_path, class_name)
        
        samples = os.listdir(class_path)
        for sample in samples:
            sample_path = os.path.join(class_path, sample)
            if not os.path.isdir(sample_path): continue
            
            frames = []
            frame_files = sorted(os.listdir(sample_path), key=lambda x: int(x.split('.')[0]))
            
            for f in frame_files:
                if f.endswith('.npy'):
                    d = np.load(os.path.join(sample_path, f)).flatten()
                    frames.append(d)
            
            if len(frames) == 0: continue
            
            frames = np.array(frames)
            
            # Normalization
            if np.max(np.abs(frames)) > 1.0:
                frames = frames / np.max(np.abs(frames))

            # Pad or Truncate
            if len(frames) < MAX_FRAMES:
                pad = np.zeros((MAX_FRAMES - len(frames), frames.shape[1]))
                frames = np.vstack((frames, pad))
            else:
                frames = frames[:MAX_FRAMES]
                
            sequences.append(frames)
            labels.append(i)

    return np.array(sequences), np.array(labels), label_map

print("Starting Data Processing...")
X, y, label_map = load_and_process()
y = to_categorical(y).astype(int)

np.save(os.path.join(PROCESSED_PATH, "X.npy"), X)
np.save(os.path.join(PROCESSED_PATH, "y.npy"), y)
print(f"DONE! Saved X shape: {X.shape} to {PROCESSED_PATH}")