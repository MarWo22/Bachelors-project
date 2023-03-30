import pandas as pd
import numpy as np
from scipy.signal import butter, sosfilt
import time
import os
import sys
from typing import Tuple
import random



def butterworth_bandpass(data: np.ndarray, low_cut: int = 1, high_cut: int = 10, fs: int = 512, order: int = 2) -> np.ndarray:
    nyq = 0.5 * fs
    low = low_cut / nyq
    high = high_cut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sosfilt(sos, data)

def create_windows(timeseries: np.ndarray, events_df: pd.DataFrame, window_size: int = 256) -> Tuple[list]:
    timestamps = events_df.index.to_list()
    events = events_df.to_numpy()

    x = list()
    y = list()

    for timestamp, label in zip(timestamps, events):
        window = timeseries[timestamp : min(timestamp + window_size, timeseries.shape[0])]
        x.append(window)
        y.append(label)

    return (x, y)

def create_dataset(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    file_names = get_raw_file_paths(data_dir)
    train_files, test_files = split_train_test(file_names)

    x_train = list()
    y_train = list()

    for file in train_files:
        x, y = process_recording(file)
        x_train += x
        y_train += y

    x_test = list()
    y_test = list()

    for file in test_files:
        x, y = process_recording(file)
        x_test += x
        y_test += y

    return (np.asarray(x_train), np.asarray(y_train), np.asarray(x_test), np.asarray(y_test))

def load_file(path: str) -> pd.DataFrame:
    file = pd.read_pickle(path)
    return file
    

def process_recording(path: str):
    df = load_file(path)
    events = df['events'].dropna()
    timeseries = df.drop(columns='events')
    timeseries_np = timeseries.to_numpy()
    filtered = butterworth_bandpass(timeseries_np)

    return create_windows(filtered, events)


def get_raw_file_paths(path) -> list[str]:
    file_name = os.listdir(path)
    paths = list()

    for file in file_name:
        paths.append('./pickle_df/' + file)

    return paths

def split_train_test(filenames: list[str]) -> Tuple[list[str]]:
    train = list()
    test = list()
    selected_file = random.choice(filenames)
    subject = selected_file.split('/')[-1][7:9]
    for file in filenames:
        if file.split('/')[-1][7:9] == subject:
            test.append(file)
        else:
            train.append(file)

    return (train, test)

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = create_dataset('./pickle_df')
    print(x_train)
    print(x_train.shape)
    print(y_train.shape)
    print(np.count_nonzero(y_train == 0))
    print(np.count_nonzero(y_train == 1))
        