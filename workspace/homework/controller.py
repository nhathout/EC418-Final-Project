import pystk

def control(aim_point, current_vel, steer_gain=6, skid_thresh=0.2, target_vel=25):
    import numpy as np
    # Initialize an action object
    action = pystk.Action()

    # Steering control
    action.steer = np.clip(steer_gain * aim_point[0], -1, 1)
    action.drift = abs(aim_point[0]) > skid_thresh

    # Acceleration and braking control
    if current_vel < target_vel:
        action.acceleration = 1.0  # Go!
        action.brake = False
    else:
        action.acceleration = 0.0
        action.brake = True  # Stop!

    # Nitro control
    if action.acceleration == 1.0 and abs(action.steer) < 0.1:
        action.nitro = True
    else:
        action.nitro = False

    return action

def test_controller(pytux, track, verbose=False):
    import numpy as np
    track = [track] if isinstance(track, str) else track
    for t in track:
        steps, how_far = pytux.rollout(t, control, max_frames=1000, verbose=verbose)
        print(steps, how_far)
    pytux.close()

if __name__ == '__main__':
    from utils import PyTux
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    pytux = PyTux()
    test_controller(pytux, args.track, verbose=args.verbose)
    pytux.close()
