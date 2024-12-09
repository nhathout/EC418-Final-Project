import pystk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PID:
    """ Proportional-Integral-Derivative (PID) Controller. """
    def __init__(self, Kp, Ki, Kd, setpoint=0.0, output_limits=(None, None)):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.output_limits = output_limits
        self.reset()
        logger.info(f"Initialized PID with Kp={Kp}, Ki={Ki}, Kd={Kd}, setpoint={setpoint}")

    def reset(self):
        """ Reset the PID controller's integral and previous error. """
        self._integral = 0.0
        self._prev_error = None
        logger.info("PID controller reset.")

    def __call__(self, measurement, dt=1.0):
        error = self.setpoint - measurement
        self._integral += error * dt
        derivative = 0.0 if self._prev_error is None else (error - self._prev_error) / dt
        self._prev_error = error

        output = self.Kp * error + self.Ki * self._integral + self.Kd * derivative
        
        # Apply output limits
        if self.output_limits[0] is not None:
            output = max(self.output_limits[0], output)
        if self.output_limits[1] is not None:
            output = min(self.output_limits[1], output)

        logger.debug(f"PID output: {output} (Error: {error}, Integral: {self._integral}, Derivative: {derivative})")
        return output


# Instantiate a global PID controller for speed
speed_pid = PID(Kp=1.0, Ki=0.1, Kd=0.05, setpoint=25.0, output_limits=(-1.0, 1.0))


def control(aim_point, current_vel, dt=1.0):
    logger.info(f"Running control - Aim Point: {aim_point}, Current Velocity: {current_vel}, dt: {dt}")

    # Calculate acceleration using the PID controller
    acceleration_signal = speed_pid(current_vel, dt)
    logger.info(f"PID Output (Acceleration): {acceleration_signal}")

    action = pystk.Action()

    if acceleration_signal > 0:
        action.acceleration = np.clip(acceleration_signal, 0.0, 1.0)
        action.brake = False
    else:
        action.acceleration = 0.0
        action.brake = np.clip(-acceleration_signal, 0.0, 1.0)
    
    # Steering logic
    steer_gain = 6.0
    skid_thresh = 0.2
    
    action.steer = np.clip(steer_gain * aim_point[0], -1.0, 1.0)
    action.drift = abs(aim_point[0]) > skid_thresh
    
    action.nitro = action.acceleration > 0.0 and abs(action.steer) < 0.1
    
    logger.info(f"Generated Action: Accel={action.acceleration}, Brake={action.brake}, Steer={action.steer}, Drift={action.drift}, Nitro={action.nitro}")
    return action


def test_controller(pytux, track, verbose=False):
    track = [track] if isinstance(track, str) else track
    for t in track:
        logger.info(f"Starting track: {t}")
        steps, how_far = pytux.rollout(t, control, max_frames=1000, verbose=verbose)
        logger.info(f"Finished track: {t} | Steps: {steps} | Distance Covered: {how_far}")
    pytux.close()


if __name__ == '__main__':
    from .utils import PyTux
    from argparse import ArgumentParser

    parser = ArgumentParser("PID-Based Controller for SuperTuxKart")
    parser.add_argument('track', nargs='+', help='Name of the track(s) to run.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output with visualizations.')
    args = parser.parse_args()

    pytux = PyTux()
    logger.info("Starting SuperTuxKart PID controller.")
    test_controller(pytux, args.track, verbose=args.verbose)
    logger.info("Testing completed.")
    pytux.close()
