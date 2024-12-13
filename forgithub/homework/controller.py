import pystk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import csv

ACTION_DIM = 8

class Q_Network(nn.Module):
    def __init__(self, state_dim=4, hidden_dim1=128, hidden_dim2=64, action_dim=ACTION_DIM):
        super(Q_Network, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, action_dim)  # Output layer for Q-values

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values

if __name__ == '__main__':
    from utils import PyTux
    from argparse import ArgumentParser

    RESULTS_CSV = 'rollout_results.csv'

    def get_last_epoch(file_path):
        """Reads the last epoch value from the CSV file."""
        if not os.path.exists(file_path):
            return 0

        try:
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                rows = list(reader)
                if len(rows) > 1:  # Check if there are data rows
                    return int(rows[-1][0]) + 1  # Increment last epoch value
        except Exception as e:
            print(f"Error reading CSV: {e}")

        return 0

    # Ensure the CSV file has a header if it doesn't exist
    if not os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Track', 'Frames', 'Distance'])

    def test_controller(args):
        pytux = PyTux()
        start_epoch = get_last_epoch(RESULTS_CSV)

        for i in range(start_epoch, start_epoch + 10000):
            for t in args.track:
                steps, how_far = pytux.rollout(t, max_frames=1000, verbose=args.verbose)
                print(f"Epoch: {i}, Track: {t}, Steps: {steps}, Distance: {how_far}")
                # Write to CSV after each rollout
                with open(RESULTS_CSV, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([i, t, steps, how_far])
        pytux.close()

    parser = ArgumentParser()
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_controller(args)
