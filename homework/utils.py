import numpy as np
import pystk
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import dense_transforms
from controller import ControlModel

RESCUE_TIMEOUT = 30
TRACK_OFFSET = 15
DATASET_PATH = '../drive_data'

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
        self.model = ControlModel()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

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

        episode_states = []
        episode_actions = []
        episode_rewards = []

        device = next(self.model.parameters()).device

        state.update()
        track_state.update()
        kart = state.players[0].kart
        prev_distance = kart.overall_distance

        for t in range(max_frames):
            state.update()
            track_state.update()
            kart = state.players[0].kart

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

            input_tensor = torch.tensor([aim_point_image[0], aim_point_image[1], current_vel, 25],
                                        dtype=torch.float32, device=device).unsqueeze(0)

            if torch.isnan(input_tensor).any() or torch.isinf(input_tensor).any():
                print(f"Invalid input tensor at t={t}. Skipping step.")
                self.k.step(pystk.Action())
                continue

            action = self.model(input_tensor)

            if torch.isnan(action).any() or torch.isinf(action).any():
                print(f"Invalid action output at t={t}. Skipping step.")
                self.k.step(pystk.Action())
                continue

            pystk_action = pystk.Action()
            pystk_action.brake = action[0, 0].item()
            pystk_action.drift = action[0, 1].item()
            pystk_action.fire = action[0, 2].item()
            pystk_action.nitro = action[0, 3].item()
            pystk_action.rescue = action[0, 4].item()
            pystk_action.acceleration = action[0, 5].item()
            pystk_action.steer = action[0, 6].item()

            if t < 120:
                pystk_action.acceleration = 1.0

            # Print out the chosen action to debug what's happening
            print(f"Step: {t}, Action: {pystk_action}, Vel: {current_vel:.3f}")

            if current_vel < 1.0 and t - last_rescue > RESCUE_TIMEOUT:
                last_rescue = t
                pystk_action.rescue = True

            if verbose and fig is not None and ax is not None and len(self.k.render_data) > 0:
                ax.clear()
                ax.imshow(self.k.render_data[0].image)
                WH2 = np.array([self.config.screen_width, self.config.screen_height]) / 2
                ax.add_artist(plt.Circle(WH2 * (1 + self._to_image(kart.location, proj, view)), 2, ec='b', fill=False, lw=1.5))
                ax.add_artist(plt.Circle(WH2 * (1 + self._to_image(aim_point_world, proj, view)), 2, ec='r', fill=False, lw=1.5))
                if planner:
                    ax.add_artist(plt.Circle(WH2*(1+aim_point_image), 2, ec='g', fill=False, lw=1.5))
                plt.pause(1e-3)

            self.k.step(pystk_action)

            current_distance = kart.overall_distance
            reward = current_distance - prev_distance
            prev_distance = current_distance

            episode_states.append(input_tensor.detach())
            episode_actions.append(action.detach())
            episode_rewards.append(reward)

            print(f"Step: {t}, Reward: {reward:.3f}")

        if verbose and fig is not None and ax is not None:
            plt.close(fig)

        total_return = sum(episode_rewards)
        print("Episode finished. Total Return:", total_return)

        loss = torch.tensor(-total_return, dtype=torch.float32, device=device, requires_grad=True)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        print("Policy updated with loss =", loss.item())

        return t, kart.overall_distance / track_state.length

    def close(self):
        if self.k is not None:
            self.k.stop()
            del self.k
        pystk.clean()

if __name__ == '__main__':
    from controller import control
    from argparse import ArgumentParser
    from os import makedirs

    def noisy_control(aim_pt, vel):
        return control(aim_pt, vel) # no noise for now

    parser = ArgumentParser("Collects a dataset for the high-level planner")
    parser.add_argument('track', nargs='+')
    parser.add_argument('-o', '--output', default=DATASET_PATH)
    parser.add_argument('-n', '--n_images', default=10000, type=int)
    parser.add_argument('-m', '--steps_per_track', default=20000, type=int)
    parser.add_argument('--aim_noise', default=0.1, type=float)
    parser.add_argument('--vel_noise', default=5, type=float)
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    try:
        makedirs(args.output)
    except OSError:
        pass

    pytux = PyTux()
    for track in args.track:
        n, images_per_track = 0, args.n_images // len(args.track)
        aim_noise, vel_noise = 0, 0

        def collect(_, im, pt):
            from PIL import Image
            from os import path
            id = n if n < images_per_track else np.random.randint(0, n + 1)
            if id < images_per_track:
                fn = path.join(args.output, track + '_%05d' % id)
                Image.fromarray(im).save(fn + '.png')
                with open(fn + '.csv', 'w') as f:
                    f.write('%0.1f,%0.1f' % tuple(pt))
            n += 1

        while n < args.steps_per_track:
            steps, how_far = pytux.rollout(track, noisy_control, max_frames=1000, verbose=args.verbose, data_callback=collect)
            print(steps, how_far)
            aim_noise, vel_noise = args.aim_noise, args.vel_noise
    pytux.close()
