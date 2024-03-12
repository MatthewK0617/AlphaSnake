import torch
import torch.nn as nn


class DQN(nn.Module):
    '''
    Basic deep Q-network. 
    '''
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_size),
        )

    def forward(self, x):
        return self.net(x)


class CustomizableDQN(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_layers=[128, 256],
        activation_function="ReLU",
    ):
        super(CustomizableDQN, self).__init__()
        layers = []

        layers.append(nn.Linear(input_size, hidden_layers[0]))

        # Dynamically add hidden layers
        for i in range(len(hidden_layers) - 1):
            if activation_function == "ReLU":
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))

        # Activation function for last hidden layer
        if activation_function == "ReLU":
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_layers[-1], output_size))

        # Combine layers
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# example use: model = CustomizableDQN(input_size=10, output_size=4, hidden_layers=[128, 256, 128], activation_function='ReLU') 