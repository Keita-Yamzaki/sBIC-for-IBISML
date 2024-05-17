import torch 
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self, layers, activation_functions = None):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.ModuleList()
        self.activations = []

        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
            # デフォルトの活性化関数をReLUに設定
            if activation_functions is None:
                activation = nn.ReLU() if i < len(layers) - 2 else nn.Identity()
            else:
                if len(activation_functions) != len(layers) - 1:
                    raise ValueError("Activation functions list must match the number of layers minus one.")
                activation = activation_functions[i]
            self.activations.append(activation)

    #forward
    def forward(self, x):
        for layer, activation in zip(self.layers, self.activations):
            x = activation(layer(x))
            
        return x 
    
    def getActivations(self):
        return self.activations
        
