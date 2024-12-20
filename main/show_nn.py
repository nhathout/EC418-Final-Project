import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# Define the Q_Network class
class Q_Network(nn.Module):
    def __init__(self, state_dim=4, hidden_dim1=128, hidden_dim2=64, action_dim=8):  # ACTION_DIM is 8 now
        super(Q_Network, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, action_dim)  # Output layer for Q-values

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values

# Create an instance of the network
state_dim = 4  # Example: state space dimension (can be adjusted)
action_dim = 8  # Updated action space dimension (8 actions)
model = Q_Network(state_dim=state_dim, action_dim=action_dim)

# Create a sample input tensor with the same shape as the input to the network
sample_input = torch.randn(1, state_dim)  # Batch size of 1, with state_dim features

# Create a SummaryWriter to log to TensorBoard
writer = SummaryWriter('runs/q_network_example')

# Log the model architecture to TensorBoard
writer.add_graph(model, sample_input)

# Close the writer after logging
writer.close()

print("TensorBoard logging is complete. Run 'tensorboard --logdir=runs' to visualize the model.")
