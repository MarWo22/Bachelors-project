import numpy as np
import random


def add_gaussian_noise(signal, sd, n_channels, low: int = 0, high: int = 308):
    signal_channels = signal.shape[0]

    channels_to_modify = random.sample(range(0, signal_channels), n_channels)

    for channel in channels_to_modify:
        noise = np.random.normal(0, sd, high - low)
        signal[channel]+= noise

    return signal
    

def zero_signal(signal, n_channels, low: int = 0, high: int = 0, percentage: int = 10):
    pass