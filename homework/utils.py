# utils.py

import numpy as np
import pystk
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import dense_transforms
from controller import control  # Changed from ControlModel to control
from PIL import Image
from glob import glob
from os import path
import os

RESCUE_TIMEOUT = 30
TRACK_OFFSET = 15
DATASET_PATH = '../drive_data'

class SuperTuxDataset(Dataset):
    """
    Dataset class for SuperTuxKart data.
    Expects PNG images and corresponding CSV labels.
    """
    def __init__(self, dataset_path=DATASET_PATH, transform=dense_transforms.ToTensor()):
        self.data = []
        self.transform = transform
        for f in glob(path.join(dataset_path, '*.csv')):
            img_path = f.replace('.csv', '.png')
            if not path.exists(img_path):
                print(f"Warning: Image file {img_path} not found for label {f}. Skipping.")
                continue
            try:
                img = Image.open(img_path).convert('RGB')
                img.load()
                label = np.loadtxt(f, dtype=np.float32, delimiter=',')
                self.data.append((img, label))
            except Exception as e:
                print(f"Error loading {f} or {img_path}: {e}. Skipping.")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img, label = self.data[idx]
        img, label = self.transform(img, label)
        return img, label

def load_data(dataset_path=DATASET_PATH, transform=dense_transforms.ToTensor(), num_workers=4, batch_size=128):
    """
    Loads the dataset and returns a DataLoader.
    
    Args:
        dataset_path (str): Path to the dataset directory.
        transform (callable): Transformations to apply to the data.
        num_workers (int): Number of subprocesses for data loading.
        batch_size (int): Number of samples per batch.
    
    Returns:
        DataLoader: PyTorch DataLoader for the dataset.
    """
    dataset = SuperTuxDataset(dataset_path, transform=transform)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)

class PyTux:
    """
    Class to handle interactions with the SuperTuxKart environment.
    """
    _singleton = None

    def __init__(self, screen_width=128, screen_height=96):
        """
        Initialize the PyTux environment.
        
        Args:
            screen_width (int): Width of the game window.
            screen_height (int): Height of the game window.
        """
        assert PyTux._singleton is None, "Cannot create more than one PyTux object"
        PyTux._singleton = self
        pystk.init(pystk.GraphicsConfig.none())
        self.k = None

    @staticmethod
    def _point_on_track(distance, track, offset=0.0):
        """
        Get a point at `distance` down the `track`. Optionally applies an offset after the track segment if found.
        Returns a 3D coordinate.
        
        Args:
            distance (float): Distance along the track.
            track (pystk.Track): Current track state.
            offset (float): Additional offset distance.
        
        Returns:
            np.array: 3D coordinate of the point on the track.
        """
        node_idx = np.searchsorted(track.path_distance[..., 1], distance % track.path_distance[-1, 1]) % len(track.path_nodes)
        d = track.path_distance[node_idx]
        x = track.path_nodes[node_idx]
        t = (distance + offset - d[0]) / (d[1] - d[0])
        return x[1] * t + x[0] * (1 - t)

    @staticmethod
    def _to_image(x, proj, view):
        """
        Convert a world coordinate to image space.
        
        Args:
            x (np.array): 3D world coordinate.
            proj (np.array): Projection matrix.
            view (np.array): View matrix.
        
        Returns:
            np.array: 2D image coordinate.
        """
        p = proj @ view @ np.array(list(x) + [1])
        return np.clip(np.array([p[0]/p[-1], -p[1]/p[-1]]), -1, 1)

    def rollout(self, track, control_fn=None, max_frames=1000, verbose=False, data_callback=None):
        """
        Perform a rollout (episode) on the specified track using the provided control function.
        
        Args:
            track (str): Name of the track.
            control_fn (callable): Function to compute actions. Should accept (aim_point, current_vel) and return pystk.Action.
            max_frames (int): Maximum number of frames per episode.
            verbose (bool): Enable verbose output with visualizations.
            data_callback (callable): Optional callback to collect data (e.g., for dataset creation).
        
        Returns:
            tuple: (steps, how_far)
        """
        # Initialize or restart the race
        if self.k is not None and self.k.config.track == track:
            self.k.restart()
            self.k.step()
        else:
            if self.k is not None:
                self.k.stop()
                del self.k
            config = pystk.RaceConfig(num_kart=1, laps=1, track=track)
            config.players[0].controller = pystk.PlayerConfig.Controller.PLAYER_CONTROL
            self.k = pystk.Race(config)
            self.k.start()
            self.k.step()

        state = pystk.WorldState()
        track_state = pystk.Track()
        last_rescue = 0

        if verbose:
            import matplotlib.pyplot as plt
            if len(self.k.render_data) > 0:
                fig, ax = plt.subplots(1, 1)
            else:
                print("Warning: No render data available for plotting.")
                fig, ax = None, None
        else:
            fig, ax = None, None

        episode_states = []
        episode_actions = []
        episode_rewards = []

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        state.update()
        track_state.update()
        kart = state.players[0].kart
        prev_distance = kart.overall_distance

        for t in range(max_frames):
            state.update()
            track_state.update()
            kart = state.players[0].kart

            # Check if the kart has completed the race
            if np.isclose(kart.overall_distance / track_state.length, 1.0, atol=2e-3):
                if verbose and fig is not None and ax is not None:
                    print(f"Finished at t={t}")
                break

            proj = np.array(state.players[0].camera.projection).T
            view = np.array(state.players[0].camera.view).T

            # Calculate aim point in world and image space
            aim_point_world = self._point_on_track(kart.distance_down_track + TRACK_OFFSET, track_state)
            aim_point_image = self._to_image(aim_point_world, proj, view)

            # Optional data collection callback
            if data_callback is not None and len(self.k.render_data) > 0:
                data_callback(t, np.array(self.k.render_data[0].image), aim_point_image)

            # Compute action using the control function
            if control_fn and len(self.k.render_data) > 0:
                image = np.array(self.k.render_data[0].image)
                current_vel = np.linalg.norm(kart.velocity)

                # Generate action using the control function
                action = control_fn(aim_point_image, current_vel)

                # Handle rescue logic
                if current_vel < 1.0 and t - last_rescue > RESCUE_TIMEOUT:
                    last_rescue = t
                    action.rescue = True

                # Verbose visualization
                if verbose and fig is not None and ax is not None:
                    ax.clear()
                    ax.imshow(image)
                    WH2 = np.array([self.config.screen_width, self.config.screen_height]) / 2
                    ax.add_artist(plt.Circle(WH2 * (1 + self._to_image(kart.location, proj, view)), 2, ec='b', fill=False, lw=1.5))
                    ax.add_artist(plt.Circle(WH2 * (1 + aim_point_image), 2, ec='r', fill=False, lw=1.5))
                    plt.pause(1e-3)

                # Execute the action
                self.k.step(action)

                # Calculate reward (for supervised learning, this might not be used)
                current_distance = kart.overall_distance
                reward = current_distance - prev_distance
                prev_distance = current_distance

                # Collect episode data if needed
                episode_states.append(torch.tensor(aim_point_image, dtype=torch.float32))
                episode_actions.append(torch.tensor([
                    action.brake,
                    action.drift,
                    action.fire,
                    action.nitro,
                    action.rescue,
                    action.acceleration,
                    action.steer
                ], dtype=torch.float32))
                episode_rewards.append(reward)

                print(f"Step: {t}, Action: {action}, Vel: {current_vel:.3f}, Reward: {reward:.3f}")

            else:
                # If no control function is provided, take no action
                action = pystk.Action()
                self.k.step(action)
                reward = 0.0

            # Optional: Implement additional logic here

        if verbose and fig is not None and ax is not None:
            plt.close(fig)

        total_return = sum(episode_rewards)
        print("Episode finished. Total Return:", total_return)

        # For supervised learning, you might not perform policy updates here
        # If you're collecting data, the model should be trained separately

        return t, kart.overall_distance / track_state.length

    def close(self):
        """
        Clean up the environment.
        """
        if self.k is not None:
            self.k.stop()
            del self.k
        pystk.clean()

if __name__ == '__main__':
    from controller import control  # Changed from ControlModel to control
    from argparse import ArgumentParser

    def noisy_control(aim_pt, vel):
        return control(aim_pt, vel)  # No noise for now

    parser = ArgumentParser("Collects a dataset for the high-level planner")
    parser.add_argument('track', nargs='+', help='Name of the track(s) to run.')
    parser.add_argument('-o', '--output', default=DATASET_PATH, help='Output directory for collected data.')
    parser.add_argument('-n', '--n_images', default=10000, type=int, help='Number of images to collect.')
    parser.add_argument('-m', '--steps_per_track', default=20000, type=int, help='Number of steps per track.')
    parser.add_argument('--aim_noise', default=0.1, type=float, help='Noise added to aim points.')
    parser.add_argument('--vel_noise', default=5, type=float, help='Noise added to velocity.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output with visualizations.')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    pytux = PyTux()
    for track in args.track:
        n, images_per_track = 0, args.n_images // len(args.track)
        aim_noise, vel_noise = args.aim_noise, args.vel_noise

        def collect(t, im, pt):
            from PIL import Image
            img_array = im
            # Optionally add noise to the aim point and velocity
            pt_noisy = pt + np.random.normal(0, aim_noise, size=pt.shape)
            img = Image.fromarray(img_array)
            fn = path.join(args.output, f"{track}_{n:05d}")
            img.save(fn + '.png')
            np.savetxt(fn + '.csv', pt_noisy, delimiter=',')
            n += 1

        while n < images_per_track:
            steps, how_far = pytux.rollout(track, control_fn=noisy_control, max_frames=args.steps_per_track, verbose=args.verbose, data_callback=collect)
            print(f"Track: {track}, Steps: {steps}, Distance Covered: {how_far}")
            aim_noise, vel_noise = args.aim_noise, args.vel_noise

    pytux.close()
