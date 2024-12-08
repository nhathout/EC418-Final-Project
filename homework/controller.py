import pystk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ControlModel(nn.Module):
    def __init__(self):
        super(ControlModel, self).__init__()
        # Define layers
        self.fc1 = nn.Linear(4, 64)  # Input: 4 features
        self.fc2 = nn.Linear(64, 7)  # Output: 7 (5 boolean + 2 scalar outputs)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # First layer with ReLU activation
        x = self.fc2(x)  # Linear layer output
        
        # Split into boolean and scalar outputs
        boolean_output = torch.sigmoid(x[:, :5])  # Apply sigmoid to first 5 outputs
        scalar_output = x[:, 5:]  # Last 2 outputs (no activation here)

        # Normalize scalar outputs
        scalar_output_norm1 = (scalar_output - scalar_output.min(dim=0, keepdim=True)[0]) / (
            scalar_output.max(dim=0, keepdim=True)[0] - scalar_output.min(dim=0, keepdim=True)[0]
        )
        scalar_output_norm2 = 2 * scalar_output_norm1 - 1

        # Concatenate final outputs
        final_output = torch.cat([boolean_output, scalar_output_norm1, scalar_output_norm2], dim=1)
        return final_output


if __name__ == '__main__':
    from utils import PyTux
    from argparse import ArgumentParser

    alpha = 0.5
    gamma = 0.9

    def test_controller(args):
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(t, max_frames=1000, verbose=args.verbose)
            print(steps, how_far)
        pytux.close()

    parser = ArgumentParser()
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_controller(args)
