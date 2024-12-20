import os
import csv
import numpy as np
import pystk
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import dense_transforms
from controller import Q_Network, ACTION_DIM
import math

RESCUE_TIMEOUT = 30
TRACK_OFFSET = 15
DATASET_PATH = '../drive_data'
MODEL_PATH = 'q_network.pth'
PROGRESS_CSV = 'track_progress.csv'

ACTION_SPACE = {
    0: {"steer": -1.0, "acceleration": 0.0, "brake": False, "nitro": False, "drift": False},
    1: {"steer": 1.0, "acceleration": 0.0, "brake": False, "nitro": False, "drift": False},
    2: {"steer": 0.0, "acceleration": 1.0, "brake": False, "nitro": False, "drift": False},
    3: {"steer": 0.0, "acceleration": 0.0, "brake": True,  "nitro": False, "drift": False},
    4: {"steer": 0.0, "acceleration": 1.0, "brake": False, "nitro": True,  "drift": False},
    5: {"steer": 0.0, "acceleration": 0.0, "brake": False, "nitro": False, "drift": True},
    6: {"steer": -1.0, "acceleration": 1.0, "brake": False, "nitro": False, "drift": True},
    7: {"steer": 1.0, "acceleration": 1.0, "brake": False, "nitro": False, "drift": True},
}

class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path=DATASET_PATH, transform=dense_transforms.ToTensor()):
        from PIL import Image
        from glob import glob
        from os import path
        self.data = []
        for f in glob(path.join(dataset_path, '*.csv')):
            i = Image.open(f.replace('.csv', '.png'))
            i.load()
            self.data.append((i, np.loadtxt(f, dtype=np.float32, delimiter=',')))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        data = self.transform(*data)
        return data

def load_data(dataset_path=DATASET_PATH, transform=dense_transforms.ToTensor(), num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path, transform=transform)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)

class PyTux:
    _singleton = None
    def __init__(self, screen_width=128, screen_height=96):
        assert PyTux._singleton is None, "Cannot create more than one pytux object"
        PyTux._singleton = self
        self.config = pystk.GraphicsConfig.none()
        self.config.screen_width = screen_width
        self.config.screen_height = screen_height
        pystk.init(self.config)
        self.k = None

        # Initialize Q-network and optimizer
        self.q_network = Q_Network()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-3)
        self.epsilon = 1.0  # start with high exploration

        # Load existing model parameters if they exist
        if os.path.exists(MODEL_PATH):
            self.q_network.load_state_dict(torch.load(MODEL_PATH))
            print("Loaded Q-network weights from", MODEL_PATH)

        # If the CSV doesn't exist, create it and add the header
        if not os.path.exists(PROGRESS_CSV):
            with open(PROGRESS_CSV, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Track", "Step", "Percent_Covered"])

    @staticmethod
    def _point_on_track(distance, track, offset=0.0):
        node_idx = np.searchsorted(track.path_distance[..., 1], distance % track.path_distance[-1, 1]) % len(track.path_nodes)
        d = track.path_distance[node_idx]
        x = track.path_nodes[node_idx]
        t = (distance + offset - d[0]) / (d[1] - d[0])
        return x[1] * t + x[0] * (1 - t)

    @staticmethod
    def _to_image(x, proj, view):
        p = proj @ view @ np.array(list(x) + [1])
        return np.clip(np.array([p[0]/p[-1], -p[1]/p[-1]]), -1, 1)

    def select_action(self, state_tensor):
        """Select an action using epsilon-greedy policy."""
        if np.random.rand() < self.epsilon:
            action_index = np.random.choice(ACTION_DIM)
        else:
            with torch.no_grad():
                q_values = self.q_network.forward(state_tensor)
            action_index = torch.argmax(q_values, dim=1).item()
        return action_index, ACTION_SPACE[action_index]

    def rollout(self, track, planner=None, max_frames=1000, verbose=False, data_callback=None):
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

        device = next(self.q_network.parameters()).device

        state.update()
        track_state.update()
        kart = state.players[0].kart

        gamma = 0.99
        # Initial "dummy" previous state
        previous_tensor = torch.tensor([0, 0, 0, 25], dtype=torch.float32, device=device).unsqueeze(0)

        for t in range(max_frames):
            state.update()
            track_state.update()
            kart = state.players[0].kart

            # Compute percentage of track covered
            percent_covered = (kart.overall_distance / track_state.length) * 100.0

            # Log progress to CSV
            with open(PROGRESS_CSV, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([track, t, percent_covered])

            if np.isclose(kart.overall_distance / track_state.length, 1.0, atol=2e-3):
                if verbose and fig is not None and ax is not None:
                    print("Finished at t=%d" % t)
                break

            proj = np.array(state.players[0].camera.projection).T
            view = np.array(state.players[0].camera.view).T
            aim_point_world = self._point_on_track(kart.distance_down_track + TRACK_OFFSET, track_state)
            aim_point_image = self._to_image(aim_point_world, proj, view)

            if data_callback is not None and len(self.k.render_data) > 0:
                data_callback(t, np.array(self.k.render_data[0].image), aim_point_image)

            if planner and len(self.k.render_data) > 0:
                image = np.array(self.k.render_data[0].image)
                aim_point_image = planner(TF.to_tensor(image)[None]).squeeze(0).cpu().detach().numpy()

            current_vel = np.linalg.norm(kart.velocity)

            next_tensor = torch.tensor([aim_point_image[0], aim_point_image[1], current_vel, 25],
                                        dtype=torch.float32, device=device).unsqueeze(0)

            if torch.isnan(next_tensor).any() or torch.isinf(next_tensor).any():
                print(f"Invalid input tensor at t={t}. Skipping step.")
                self.k.step(pystk.Action())
                continue

            # Decay epsilon over time
            self.epsilon = max(self.epsilon * 0.95, 0.01)
            action_index, action = self.select_action(next_tensor)

            if verbose:
                print(f"Step: {t}, Action index: {action_index}, Action: {action}, Vel: {current_vel:.3f}")

            # Reward: Distance along track
            # if t != 0.99:
            #     reward = (kart.overall_distance / track_state.length) / (math.log(t + 0.01))
            # else:
            reward = (kart.overall_distance / track_state.length)

            self.optimizer.zero_grad()
            current_q_values = self.q_network.forward(previous_tensor)
            current_q_value = current_q_values[0, action_index]

            with torch.no_grad():
                next_q_values = self.q_network.forward(next_tensor)
            max_next_q_value = torch.max(next_q_values).item()

            target_q_value = reward + gamma * max_next_q_value
            target_q_value = torch.tensor(target_q_value, dtype=torch.float32, device=device)
            loss = F.mse_loss(current_q_value, target_q_value)
            loss.backward()
            self.optimizer.step()

            previous_tensor = next_tensor

            if current_vel < 1.0 and t - last_rescue > RESCUE_TIMEOUT:
                last_rescue = t
                action["rescue"] = True
            else:
                action["rescue"] = False

            action["fire"] = False
            self.k.step(pystk.Action(
                steer=action["steer"],
                acceleration=action["acceleration"],
                brake=float(action["brake"]),
                nitro=float(action["nitro"]),
                drift=action["drift"],
                fire=float(action["fire"])
            ))

        if verbose and fig is not None and ax is not None:
            plt.close(fig)

        # Save the model parameters at the end of the rollout
        torch.save(self.q_network.state_dict(), MODEL_PATH)
        print("Saved Q-network weights to", MODEL_PATH)

        return t, kart.overall_distance / track_state.length

    def close(self):
        if self.k is not None:
            self.k.stop()
            del self.k
        pystk.clean()
