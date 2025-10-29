# Script ini membuat model Keras kecil (.h5) dan sebuah file .pt dummy.
# NOTE: menjalankan script ini membutuhkan tensorflow dan torch di lingkungan lokal.
import os
os.makedirs('models', exist_ok=True)

try:
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Flatten
    from tensorflow.keras.optimizers import Adam

    model = Sequential([Flatten(input_shape=(224,224,3)), Dense(16, activation='relu'), Dense(3, activation='softmax')])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy')
    model.save('models/model.h5')
    print('Saved models/model.h5')
except Exception as e:
    print('Skipping Keras model creation — error or missing dependency:', e)

try:
    import torch
    import torch.nn as nn

    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.fc = nn.Sequential(nn.Linear(224*224*3, 64), nn.ReLU(), nn.Linear(64, 3))
        def forward(self, x):
            x = self.flatten(x)
            return self.fc(x)

    net = TinyNet()
    torch.save(net, 'models/model.pt')
    print('Saved models/model.pt')
except Exception as e:
    print('Skipping PyTorch model creation — error or missing dependency:', e)
