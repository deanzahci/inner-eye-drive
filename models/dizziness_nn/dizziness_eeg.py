import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(32*8, 5)
        
        self.fc2 = nn.Linear(5, 5)
        
        self.fc3 = nn.Linear(5, 4)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        
        x = F.leaky_relu(self.fc2(x))
        
        x = self.fc3(x)
        x = nn.Softmax(dim=0)(x)  # Output layer with softmax for multi-class classification
        return x
    
model = SimpleMLP()
model.load_state_dict(torch.load('model_weights.pth', weights_only=True))


def evaluate_dizziness(eight_channel_256_samples): # numpy array shape is (256, 8) transpose if you need to
    return model.eval(torch.from_numpy(np.abs(np.fft.fft(eight_channel_256_samples, axis = 0)[1:33]).flatten()).float()).tolist()
# probability of maximal dizziness, mid, low, none