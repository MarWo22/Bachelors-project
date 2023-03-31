import pandas as pd
import numpy as np
from scipy.signal import butter, sosfilt
import time
import os
import sys
from typing import Tuple
import random



def butterworth_bandpass(data: np.ndarray, low_cut: int = 1, high_cut: int = 10, fs: int = 512, order: int = 2) -> np.ndarray:
    """Applies a butterworth bandpass filter to the timeseries data

    Args:
        data (np.ndarray): The timeseries data
        low_cut (int, optional): Low cut for the butterworth filter. Defaults to 1.
        high_cut (int, optional): High cut for the butterworth filter. Defaults to 10.
        fs (int, optional): Sampling rate. Defaults to 512.
        order (int, optional): Order for the butterworth filter. Defaults to 2.

    Returns:
        np.ndarray: Filtered timeseries
    """

    nyq = 0.5 * fs
    low = low_cut / nyq
    high = high_cut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sosfilt(sos, data)

def create_windows(timeseries: np.ndarray, events_df: pd.DataFrame, window_size: int = 308) -> Tuple[list, list]:
    """Creates fixed-sized windows from the series and events

    Args:
        timeseries (np.ndarray): Timeseries of the data
        events_df (pd.DataFrame): Labels of the data
        window_size (int, optional): The epochs per window size. Defaults to 308 (which is ~600ms at 512hz fs).

    Returns:
        Tuple[list, list]: List with the windows and list with the labels
    """

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
    """Creates the dataset from the given directory. It reads all pickle files, splits them into a train/test split,
    creates windows, applies preprocessing and balances the data

    Args:
        data_dir (str): The directory in which the pickle datafiles are located

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: in order: Training timeseries (train_x), Training lables (train_y), Testing timeseries (test_x), Testing labels (text_y)
    """

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


    x_train_balanced, y_train_balanced = balance_data(np.asarray(x_train), np.asarray(y_train))

    return (x_train_balanced, y_train_balanced, np.asarray(x_test), np.asarray(y_test))
    

def process_recording(path: str) -> Tuple[list, list]:
    """Processes one recording. It applies filtering and creates windows of 600ms

    Args:
        path (str): The path to pickle datafile

    Returns:
        Tuple[list, list]: List of the timeseries windows (x) and a list of the labels (y)
    """

    df = pd.read_pickle(path)
    events = df['events'].dropna()
    timeseries = df.drop(columns='events')
    timeseries_np = timeseries.to_numpy()
    filtered = butterworth_bandpass(timeseries_np)
    return create_windows(filtered, events)

def get_raw_file_paths(path) -> list[str]:
    """Extracts the pickle file filenames from a directory

    Args:
        path (_type_): Path to the folder

    Returns:
        list[str]: List with filenames
    """

    file_name = os.listdir(path)
    paths = list()

    for file in file_name:
        paths.append('./pickle_df/' + file)

    return paths

def split_train_test(filenames: list[str]) -> Tuple[list[str], list[str]]:
    """Randomly selects one subject for the test split and splist the files accordingly.

    Args:
        filenames (list[str]): List with the filenames of the data

    Returns:
        Tuple[list[str], list[str]]]: List of filenames for the training data and list for the testing data
    """

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

def balance_data(timeseries: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Balances the data to equal distributions

    Args:
        timeseries (np.ndarray): The timeseries (x) of the dataset
        labels (np.ndarray): The labels (y) of the dataset

    Returns:
        Tuple[np.ndarray, np.ndarray]: The balanced timeseries (x) and labels (y)
    """

    label_count = (np.count_nonzero(labels == 0), np.count_nonzero(labels == 1))

    class_to_sample = label_count.index(min(label_count))
    sample_count = label_count[(class_to_sample + 1) % 2] - label_count[class_to_sample]

    indexes = np.where(labels == class_to_sample)[0]
    samples = np.random.choice(indexes, sample_count, replace=True)

    sampled_labels = np.full(sample_count, class_to_sample)
    sampled_timeseries = np.empty((sample_count, timeseries.shape[1], timeseries.shape[2]))
    for i, sample in enumerate(samples):
        sampled_timeseries[i] = timeseries[sample]
    # Not the most efficient method, but it works     
    timeseries_balanced = np.concatenate([timeseries, sampled_timeseries])
    labels_balanced = np.concatenate([labels, sampled_labels])

    randomize = np.arange(len(timeseries_balanced))
    np.random.shuffle(randomize)
    timeseries_shuffled = timeseries_balanced[randomize]
    labels_shuffled = labels_balanced[randomize]

    return (timeseries_shuffled, labels_shuffled)


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = create_dataset('./pickle_df')
    print(x_train)
    print(x_train.shape)
    print(y_train.shape)
    print(np.count_nonzero(y_train == 0))
    print(np.count_nonzero(y_train == 1))
        