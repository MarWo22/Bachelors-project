import scipy
import numpy as np
import pandas as pd
import os


#Dataset: https://lampz.tugraz.at/~bci/database/013-2015/description.pdf

def main():

    if not os.path.exists('C:/Users/marti/Desktop/Bachelors project/src/data'):
        print("Please add a 'data' folder and download datasets into this folder")
    else:
        if not os.path.exists('pickle_df'):
            os.makedirs('pickle_df')

        for i in range(1, 7):
            for j in range(1, 3):
                convert_mat_file(i, j)

def convert_mat_file(subject_id, trial):
    """Loads the .mat files from the 'data' folder. It extracts the EEG channels and the events,
    and converts them to a pandas dataframe. The dataframes are then pickled and saved to the 'pickle_df' folder

    Args:
        subject_id (_type_): The id of the subject's file to load
        trial (_type_): the trial to load
    """
    # Load the file for the given subject and trial
    mat = scipy.io.loadmat('C:/Users/marti/Desktop/Bachelors project/src/data/Subject0{id}_s{trial}.mat'.format(id=subject_id, trial=trial))

    for run in range(10):
        # Extract columns (Drop last 'status' column)
        columns = mat['run'][0][run]['header'][0][0]['Label'][0][0][:64]
        # Remove nested np.arrays
        columns = np.array([i[0][0] for i in columns])
        # Extract raw EEG data
        data = mat['run'][0][run]['eeg'][0][0]
        # Convert to a pandas DataFrame
        df = pd.DataFrame(data, columns = columns)

        # Extract the event timestamps
        event_timestamps = mat['run'][0][run]['header'][0][0]['EVENT'][0][0]['POS'][0][0]
        event_timestamps = np.array([i[0] for i in event_timestamps])

        #Extract the event codes
        event_codes = mat['run'][0][run]['header'][0][0]['EVENT'][0][0]['TYP'][0][0]
        event_codes = np.array([i[0] for i in event_codes])

        # Create a numpy array filled with NaN for the events
        events = np.empty((data.shape[0], 1))
        events[:] = np.nan

        # Add the events to the numpy array. A normal event is encoded as a 0, whereas a ErrP event is encoded with a 1
        for timestamp, code in zip(event_timestamps, event_codes):
            if code in [5, 10]: # Normal event
                events[timestamp] = 0
            elif code in [6, 9]: # ErrP Event
                events[timestamp] = 1
            elif code in [4, 8]: # Unkown Event
                events[timestamp] = 2
        
        # Add the events to the dataframe
        df["events"] = events  

        # Write to a pickle file
        df.to_pickle('pickle_df/Subject0{id}_s{trial}_r{run}'.format(id=subject_id, trial=trial, run=run))

if __name__ == '__main__':
    main()