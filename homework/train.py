# train.py

from planner import Planner, save_model 
import torch
import numpy as np
from utils import load_data
import dense_transforms
from torchviz import make_dot  # Optional, for model visualization
import matplotlib.pyplot as plt
import os
import json

def train(args):
    from os import path
    model = Planner()

    # Set default log_dir if not provided
    log_dir = args.log_dir if args.log_dir is not None else '../logs/train'

    # Ensure the log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Initialize data containers for plotting and saving
    loss_history = []
    weight1_history = []
    weight2_history = []

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("device:", device)
    model = model.to(device)

    if args.continue_training:
        model_path = path.join(path.dirname(path.abspath(__file__)), 'planner.th')
        if path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            print(f"Loaded model from {model_path}")
        else:
            print(f"No model found at {model_path}. Starting fresh training.")

    loss_fn = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    import inspect
    transform = eval(args.transform, {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})

    # Load data from '../drive_data' instead of '../collected_data'
    train_data = load_data('../drive_data', transform=transform, num_workers=args.num_workers, batch_size=128)

    # Optional: Visualize the model graph with torchviz (only for the first batch)
    try:
        example_data, _ = next(iter(train_data))
        example_data = example_data.to(device)
        dot = make_dot(model(example_data), params=dict(model.named_parameters()))
        dot.render(path.join(log_dir, 'model_architecture'), format="png")  # Save graph as PNG
        print(f"Model architecture saved to {path.join(log_dir, 'model_architecture.png')}")
    except StopIteration:
        print("Training data is empty. Skipping model architecture visualization.")
    except Exception as e:
        print(f"An error occurred while rendering the model architecture: {e}")

    global_step = 0
    for epoch in range(args.num_epoch):
        model.train()
        losses = []
        for img, label in train_data:
            img, label = img.to(device), label.to(device)

            pred = model(img)
            loss_val = loss_fn(pred, label)

            losses.append(loss_val.detach().cpu().numpy())

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1

        # Calculate average loss for the epoch
        avg_loss = np.mean(losses)
        loss_history.append(avg_loss)

        # Track model weights
        try:
            # Assuming Planner has parameters 'weight1' and 'weight2'
            weight1 = model.normalized_weight1.item()
            weight2 = model.normalized_weight2.item()
            weight1_history.append(weight1)
            weight2_history.append(weight2)
        except AttributeError:
            print("Model does not have 'weight1' and 'weight2' attributes.")
            weight1_history.append(0)
            weight2_history.append(0)

        print(f'Epoch {epoch+1}/{args.num_epoch} \t Loss: {avg_loss:.4f}')

        # Optionally, log visualizations per epoch or at certain intervals
        # log(img, label, pred, global_step, log_dir=log_dir)

        save_model(model)

    # Final save of the model
    save_model(model)
    print("Training completed and model saved.")

    # Save training metrics
    np.save(path.join(log_dir, 'loss_history.npy'), np.array(loss_history))
    np.save(path.join(log_dir, 'weight1_history.npy'), np.array(weight1_history))
    np.save(path.join(log_dir, 'weight2_history.npy'), np.array(weight2_history))
    print(f"Training metrics saved to {log_dir}")

    # Plotting Loss Curve
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, args.num_epoch + 1), loss_history, label='Training Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    loss_plot_path = path.join(log_dir, 'loss_curve.png')
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Loss curve saved to {loss_plot_path}")

    # Plotting Weights Evolution
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, args.num_epoch + 1), weight1_history, label='Weight1', color='red')
    plt.plot(range(1, args.num_epoch + 1), weight2_history, label='Weight2', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Weight Value')
    plt.title('Weights Evolution Over Epochs')
    plt.legend()
    plt.grid(True)
    weights_plot_path = path.join(log_dir, 'weights_evolution.png')
    plt.savefig(weights_plot_path)
    plt.close()
    print(f"Weights evolution plot saved to {weights_plot_path}")

def log(img, label, pred, global_step, log_dir='plots'):
    """
    Save visualization of predictions vs ground truth.
    
    Args:
        img (torch.Tensor): Image tensor from data loader.
        label (torch.Tensor): Ground-truth aim point.
        pred (torch.Tensor): Predicted aim point.
        global_step (int): Current global step.
        log_dir (str): Directory to save visualization images.
    """
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    import os

    # Ensure the log directory exists
    os.makedirs(log_dir, exist_ok=True)

    fig, ax = plt.subplots(1, 1)
    ax.imshow(TF.to_pil_image(img[0].cpu()))
    WH2 = np.array([img.size(-1), img.size(-2)]) / 2
    # Adjust the scaling factor as needed based on your coordinate system
    ax.add_artist(plt.Circle(WH2 * (label[0].cpu().detach().numpy() + 1), 2, ec='g', fill=False, lw=1.5))
    ax.add_artist(plt.Circle(WH2 * (pred[0].cpu().detach().numpy() + 1), 2, ec='r', fill=False, lw=1.5))
    ax.set_title(f'Visualization at Step {global_step}')
    
    # Save the figure
    plt.savefig(os.path.join(log_dir, f'viz_step_{global_step}.png'))
    plt.close(fig)  # Close the figure to avoid memory leaks

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', default='../logs/train', help='Directory to save plots and model architecture')
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('-w', '--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='Learning rate for optimizer')
    parser.add_argument('-c', '--continue_training', action='store_true', help='Continue training from saved model')
    parser.add_argument('-t', '--transform', default='Compose([ColorJitter(0.2, 0.5, 0.5, 0.2), RandomHorizontalFlip(), ToTensor()])', help='Data augmentation transforms')

    args = parser.parse_args()
    train(args)
