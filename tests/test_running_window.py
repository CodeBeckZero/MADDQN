import numpy as np
import pytest
from utilities.data import RunningWindowDataset, sliding_window_array

# Generate some sample data for testing
sample_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
window_size = 2

# PyTest for the sliding_window_array function
def test_sliding_window_array():
    # Test if sliding_window_array returns the correct shape
    windowed_array = sliding_window_array(sample_data, window_size)
    assert windowed_array.shape == (len(sample_data) - window_size + 1, window_size, sample_data.shape[1])

    # Test if sliding_window_array returns the correct windowed data
    assert np.array_equal(windowed_array[0], sample_data[:window_size])
    assert np.array_equal(windowed_array[-1], sample_data[-window_size:])

# PyTest for the RunningWindowDataset class
def test_running_window_dataset():
    # Initialize the RunningWindowDataset
    dataset = RunningWindowDataset(sample_data, window_size)

    # Test if __len__ method returns the correct length
    assert len(dataset) == len(sample_data) - window_size + 1

    # Test if __getitem__ method returns the correct windowed data for integer indexing
    assert np.array_equal(dataset[0], sample_data[:window_size])
    assert np.array_equal(dataset[-1], sample_data[-window_size:])

    # Test if __getitem__ method returns the correct windowed data for slice indexing
    assert np.array_equal(dataset[0:3], np.array([sample_data[0:2],sample_data[1:3], sample_data[2:4]]))

    # Test if __getitem__ method raises TypeError for unsupported index type
    with pytest.raises(TypeError):
        dataset['string_index']