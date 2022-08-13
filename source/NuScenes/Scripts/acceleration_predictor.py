from torch import nn
import argparse

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.input_dim = np.shape(x) 
        self.linear_tanh_stack = nn.Sequential(
            nn.BatchNorm1D(input_dim),
            nn.Linear(input_dim, 512),
            nn.Tanh(),
            nn.BatchNorm1D(input_dim),
            nn.Linear(input_dim, 512),
            nn.Tanh(),
            nn.BatchNorm1D(input_dim),
            nn.Linear(input_dim, 512),
            nn.Tanh(),
            nn.BatchNorm1D(input_dim),
            nn.Linear(input_dim, 512),
            nn.Tanh(),
            nn.BatchNorm1D(input_dim),
            nn.Linear(input_dim, 512),
            nn.Tanh(),
            nn.BatchNorm1D(input_dim),
            nn.Linear(input_dim, 512),
            nn.Tanh(),
            nn.BatchNorm1D(input_dim),
            nn.Linear(input_dim, 512),
            nn.Tanh(),
            nn.BatchNorm1D(input_dim),
            nn.Linear(input_dim, 512),
            nn.Tanh(),
            nn.BatchNorm1D(input_dim),
            nn.Linear(input_dim, 512),
            nn.Tanh(),
            nn.BatchNorm1D(input_dim),
            nn.Linear(input_dim, 512),
            nn.Tanh(),
            nn.BatchNorm1D(input_dim),
            nn.Linear(input_dim, 512),
            nn.Tanh(),
            nn.BatchNorm1D(input_dim),
            nn.Linear(input_dim, 512),
            nn.Tanh(),
            nn.BatchNorm1D(input_dim),
            nn.Linear(input_dim, 512),
            nn.Tanh(),
            nn.BatchNorm1D(input_dim),
            nn.Linear(input_dim, 512),
            nn.Tanh(),
            nn.BatchNorm1D(input_dim),
            nn.Linear(input_dim, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def main(){
    initial_positions = [] #Get from the visual node
    horizon = N_h
    model_input = initial_positions[k-N_h+1:k] #Input to the model
}   