import pystk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PID:
    """
    Proportional-Integral-Derivative (PID) Controller.
    """
    def __init__(self, Kp, Ki, Kd, setpoint=0.0, output_limits=(None, None)):
        """
        Initialize the PID controller with gains and setpoint.

        Args:
            Kp (float): Proportional gain.
            Ki (float): Integral gain.
            Kd (float): Derivative gain.
            setpoint (float): Desired target value.
            output_limits (tuple): Min and max limits for the output.
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.output_limits = output_limits
        self.reset()
    
    def reset(self):
        """
        Reset the PID controller's integral and previous error.
        """
        self._integral = 0.0
        self._prev_error = None
    
    def __call__(self, measurement, dt=1.0):
        """
        Calculate the PID output based on the current measurement.

        Args:
            measurement (float): Current measured value.
            dt (float): Time difference since the last call.

        Returns:
            float: Control signal after applying PID logic.
        """
        error = self.setpoint - measurement
        self._integral += error * dt
        derivative = 0.0
        if self._prev_error is not None:
            derivative = (error - self._prev_error) / dt
        self._prev_error = error
        output = self.Kp * error + self.Ki * self._integral + self.Kd * derivative
        
        # Apply output limits
        if self.output_limits[0] is not None:
            output = max(self.output_limits[0], output)
        if self.output_limits[1] is not None:
            output = min(self.output_limits[1], output)
        
        return output

# Instantiate a global PID controller for speed
# Adjust Kp, Ki, Kd based on tuning requirements
speed_pid = PID(Kp=1.0, Ki=0.1, Kd=0.05, setpoint=25.0, output_limits=(-1.0, 1.0))  # Target velocity is 25 units

def control(aim_point, current_vel, dt=1.0):
    """
    PID-based control function to adjust kart actions based on speed.

    Args:
        aim_point (np.array): Target aim point (e.g., [x, y]).
        current_vel (float): Current velocity of the kart.
        dt (float): Time delta since the last control update.

    Returns:
        pystk.Action: Action vector to control the kart.
    """
    # Calculate acceleration using the PID controller based on current velocity
    acceleration_signal = speed_pid(current_vel, dt)
    
    # Initialize the action object
    action = pystk.Action()
    
    # Map the PID output to acceleration and brake actions
    if acceleration_signal > 0:
        # Accelerate forward
        action.acceleration = np.clip(acceleration_signal, 0.0, 1.0)
        action.brake = False
    else:
        # Apply braking (invert the signal for braking)
        action.acceleration = 0.0
        action.brake = np.clip(-acceleration_signal, 0.0, 1.0)
    
    # Steering logic based on aim_point
    steer_gain = 6.0  # Adjust gain as needed
    skid_thresh = 0.2  # Threshold for initiating a drift
    
    # Assuming aim_point[0] represents lateral deviation from the track center
    action.steer = np.clip(steer_gain * aim_point[0], -1.0, 1.0)
    action.drift = abs(aim_point[0]) > skid_thresh
    
    # Nitro activation based on acceleration and steering
    if action.acceleration > 0.0 and abs(action.steer) < 0.1:
        action.nitro = True
    else:
        action.nitro = False
    
    # Set other action fields to default values
    action.fire = False
    action.rescue = False
    
    return action

def test_controller(pytux, track, verbose=False):
    """
    Test the controller on specified track(s).

    Args:
        pytux (PyTux): Instance of the PyTux environment.
        track (str or list): Track name(s).
        verbose (bool): Enable verbose output.
    """
    import numpy as np
    track = [track] if isinstance(track, str) else track
    for t in track:
        steps, how_far = pytux.rollout(t, control, max_frames=1000, verbose=verbose)
        print(f"Track: {t}, Steps: {steps}, Distance Covered: {how_far}")
    pytux.close()

if __name__ == '__main__':
    from utils import PyTux
    from argparse import ArgumentParser
    parser = ArgumentParser("PID-Based Controller for SuperTuxKart")
    parser.add_argument('track', nargs='+', help='Name of the track(s) to run.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output with visualizations.')
    args = parser.parse_args()
    pytux = PyTux()
    test_controller(pytux, args.track, verbose=args.verbose)
    pytux.close()