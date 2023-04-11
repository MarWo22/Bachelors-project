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

def create_windows(timeseries: np.ndarray, events_df: pd.DataFrame, window_size: int = 308) -> list:
    """Creates fixed-sized windows from the series and events

    Args:
        timeseries (np.ndarray): Timeseries of the data
        events_df (pd.DataFrame): Labels of the data
        window_size (int, optional): The epochs per window size. Defaults to 308 (which is ~600ms at 512hz fs).

    Returns:
        Tuple[list, list]: List with the windows and list with the labels
    """

    timestamps = events_df.index.to_numpy()
    events = events_df.to_numpy()

    data = list()

    for timestamp, label in zip(timestamps, events):
        window = timeseries[timestamp : min(timestamp + window_size, timeseries.shape[0])].astype(np.float32)
        window = np.swapaxes(window,0,1)
        data.append([window, int(label)])

    return data

def create_dataset(data_dir: str) -> Tuple[list, list, list]:
    """Creates the dataset from the given directory. It reads all pickle files, splits them into a train/test split,
    creates windows, applies preprocessing and balances the data

    Args:
        data_dir (str): The directory in which the pickle datafiles are located

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: in order: Training timeseries (train_x), Training lables (train_y), Testing timeseries (test_x), Testing labels (text_y)
    """

    file_names = get_raw_file_paths(data_dir)
    train_files, val_files, test_files = split_train_val_test(file_names)
    
    train = list()

    for file in train_files:
        data = process_recording(file)
        train += data

    val = list()

    for file in val_files:
        data = process_recording(file)
        val += data

    test = list()

    for file in test_files:
        data = process_recording(file)
        test += data


    train_balanced = balance_data(train)
    return (train_balanced, val, test)
    

def process_recording(path: str) -> list:
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
        paths.append(path + '/' + file)

    return paths

def split_train_val_test(filenames: list[str]) -> Tuple[list[str], list[str], list[str]]:
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

    # Grabs 20 for validation set
    random.shuffle(train)
    val = train[:20]
    train = train[20:]
    
    return (train, val, test)

def balance_data(data: list) -> list:
    """Balances the data to equal distributions

    Args:
        timeseries (np.ndarray): The timeseries (x) of the dataset
        labels (np.ndarray): The labels (y) of the dataset

    Returns:
        Tuple[np.ndarray, np.ndarray]: The balanced timeseries (x) and labels (y)
    """


    class_a = sum(1 for i in data if i[1] == 0)
    class_b = len(data) - class_a

    class_to_sample = 0 if class_a < class_b else 1
    sample_count = max(class_a, class_b) - min(class_a, class_b)

    minority_class = [i for i in data if i[1] == class_to_sample]

    oversamples = random.choices(minority_class, k=sample_count)
    
    data += oversamples

    return data
        

if __name__ == '__main__':
    file_names = get_raw_file_paths('./pickle_df')
    train_files, val_files, test_files = split_train_val_test(file_names)
    data = process_recording(train_files[0])