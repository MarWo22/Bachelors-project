import numpy as np
import random


def add_gaussian_noise(signal, sd, n_channels, low: int = 0, high: int = 308):
    signal_copy = signal.copy()
    signal_channels = signal_copy.shape[0]

    channels_to_modify = random.sample(range(0, signal_channels), n_channels)

    for channel in channels_to_modify:
        noise = np.random.normal(0, sd, high - low)
        signal_copy[channel][low:high]+= noise

    return signal_copy
    

def zero_signal(signal, n_channels, low: int = 0, high: int = 308, percentage: int = 10):
    signal_copy = signal.copy()
    signal_channels = signal_copy.shape[0]

    channels_to_modify = random.sample(range(0, signal_channels), n_channels)

    for channel in channels_to_modify:
        zero_idxs = random.sample(range(low, high), int(((high - low) / 100) * percentage))
        signal_copy[channel][zero_idxs] = 0

    return signal_copy