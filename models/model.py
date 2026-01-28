
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, BatchNormalization, Input

def create_model(input_shape, num_classes):
    model = Sequential()
    
    # 1. Input Layer
    model.add(Input(shape=input_shape))
    
    # 2. Masking (Keep this! It fixes the accuracy)
    model.add(Masking(mask_value=0.0))
    
    # 3. LSTM Layer 1 
    # FIX: Added 'unroll=True' to prevent the GPU crash
    model.add(LSTM(128, return_sequences=True, activation='tanh', unroll=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    # 4. LSTM Layer 2
    # FIX: Added 'unroll=True'
    model.add(LSTM(64, return_sequences=False, activation='tanh', unroll=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    # 5. Output Layer
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    return model