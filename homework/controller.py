import pystk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ControlModel(nn.Module):
    def __init__(self):
        super(ControlModel, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 7)

        # Optional: Initialize weights so acceleration starts slightly positive
        # This is a hack to encourage initial forward motion.
        with torch.no_grad():
            self.fc2.bias[5] = 1.0  # Make acceleration bias positive

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        boolean_output = torch.sigmoid(x[:, :5])
        scalar_output = torch.tanh(x[:, 5:]) 
        final_output = torch.cat([boolean_output, scalar_output], dim=1)
        return final_output

if __name__ == '__main__':
    from utils import PyTux
    from argparse import ArgumentParser

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
