import numpy as np
from torch.utils.data import Dataset, DataLoader

class RunningWindowDataset(Dataset):
    def __init__(self, data:np.array, window_size:int):
        """
        Initialize the RunningWindowDataset.

        Parameters:
            data (numpy.ndarray): The input data.
            window_size (int): The size of the sliding window.
        """
        self.data = sliding_window_array(data, window_size)
        self.window_size = window_size
        self.shape = self.data.shape

    def __len__(self):
        """
        Return the length of the dataset after considering the sliding window.
        """
        return self.data.shape[0]

    def __getitem__(self, index):
        """
        Retrieve a window of data from the dataset.

        Parameters:
            index (int, slice): The index or slice to retrieve.

        Returns:
            numpy.ndarray: The window of data.
        """
        if isinstance(index, int):
            # If index is an integer, get the window of data at that index
            window_data = self.data[index]
        elif isinstance(index, slice):
            # If index is a slice, handle slice indexing
            start, stop, step = index.indices(len(self))
            # Stack the window data along a new axis to create a single array
            window_data = np.stack([self[i] for i in range(start, stop, step)], axis=0)
        else:
            raise TypeError("Unsupported index type. Expected int or slice.")

        return window_data

def sliding_window_array(input_data: np.array, window_size: int) -> np.ndarray:
    """
    Converts a NumPy array into a NumPy ndarray with specified sliding window size.

    Parameters:
        input_data (np.array): An array with samples as rows and features as columns.
        window_size (int): Number of samples in each window.

    Returns:
        np.ndarray: NumPy ndarray with the shape of [Batch, Sample #, Feature #].
    """
    # Convert input data to a NumPy ndarray
    ndarray_copy = np.array(input_data, copy=True)
    
    # Ensure input_data is at least 2D
    if ndarray_copy.ndim == 1:
        ndarray_copy = ndarray_copy.reshape(-1, 1)

    # Calculate the number of windows
    total_samples = ndarray_copy.shape[0]
    num_windows = total_samples - window_size + 1

    # Create an empty array to store the windowed data
    windowed_array = np.zeros((num_windows, window_size, ndarray_copy.shape[1]))

    # Iterate over the input data and create windows
    for i in range(num_windows):
        windowed_array[i] = ndarray_copy[i:i+window_size]

    return windowed_array

class DateTimeSlidingWindowDataset(Dataset):
    def __init__(self, datetimes, window_size):
        """
        Initialize the DateTimeSlidingWindowDataset.

        Parameters:
            datetimes (list of datetime.datetime): The input datetime data.
            window_size (int): The size of the sliding window.
        """
        self.data = self.create_sliding_windows(datetimes, window_size)
        self.window_size = window_size

    def create_sliding_windows(self, datetimes, window_size):
        """
        Creates sliding windows from a list of datetime objects.

        Parameters:
            datetimes (list of datetime.datetime): List of datetime objects.
            window_size (int): The sliding window size.

        Returns:
            list of lists: Each sublist contains datetime objects in a window.
        """
        if len(datetimes) < window_size:
            raise ValueError("The total number of datetime entries must be greater than the window size.")
        
        windows = []
        for i in range(len(datetimes) - window_size + 1):
            windows.append(datetimes[i:i + window_size])
        return windows

    def __len__(self):
        """
        Return the length of the dataset after considering the sliding window.
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Retrieve a window of data from the dataset.

        Parameters:
            index (int): The index to retrieve.

        Returns:
            list of datetime.datetime: The window of data.
        """
        if not isinstance(index, int):
            raise TypeError("Index must be an integer")
        return self.data[index]