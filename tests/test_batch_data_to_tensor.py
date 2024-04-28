import numpy as np
import torch
import pytest
from utilities.cleandata import batch_data_to_tensor

@pytest.fixture
def random_data():
    """Generate random data for testing."""
    random_array = np.random.randint(0, 100, size=(100, 5))
    random_window_size = np.random.randint(3, 25)
    additional_elements = random_window_size - (random_array.shape[0] % random_window_size)
    return random_array, random_window_size, additional_elements

def test_correct_n_batches(random_data):
    """Test if the number of batches of the output tensor is correct."""
    random_array, random_window_size, _ = random_data
    b_tensor = batch_data_to_tensor(random_array, random_window_size)
    n_batches = (random_array.shape[0] // random_window_size) + 1
    
    assert b_tensor.size(0) == n_batches

def test_correct_padding(random_data):
    """Test if the correct padding is used."""
    random_array, random_window_size, additional_elements = random_data
    
    assert ((random_array.shape[0] %  random_window_size) + additional_elements) == random_window_size
    
def test_batch_data_to_tensor_size(random_data):
    """Test if the window size of the output tensor is correct."""
    random_array, random_window_size, _ = random_data
    b_tensor = batch_data_to_tensor(random_array, random_window_size)
    
    assert b_tensor.size(1) == random_window_size

def test_batch_data_to_tensor_sum(random_data):
    """Test if the sum of the last window in the first series matches."""
    random_array, random_window_size, additional_elements = random_data
    b_tensor = batch_data_to_tensor(random_array, random_window_size)
    narray_last_valid_idx = random_window_size - additional_elements   
    last_window_sum_tensor = torch.sum(b_tensor[-1, :, 0]).item()
    last_window_sum_array = np.sum(random_array[-narray_last_valid_idx:, 0])
    
    assert last_window_sum_tensor == last_window_sum_array