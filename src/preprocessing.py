import pandas as pd
import time
import os
import sys



def butterworth_bandpass(data):
    pass


def load_file(path):
    file = pd.read_pickle(path)
    return file
    


if __name__ == '__main__':
    for file in os.listdir('./pickle_df'):
        load_file('./pickle_df/' + file)