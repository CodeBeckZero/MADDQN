import numpy as np
import pytest
from cleandata import OHLC_to_Candle_img

def test_basic_correctness():
    # Define a simple OHLC window with known characteristics
    OHLC_window_narray = np.array([[10, 15, 5, 15],
                                   [12, 20, 8, 10],
                                   [8, 10, 6, 9]])

    # Call the OHLC_to_Candle_img function
    candlestick_img = OHLC_to_Candle_img(OHLC_window_narray)
      
    # Check specific characteristics of the resulting candlestick image
    assert candlestick_img.shape == (9, 9)  # Since window_size is not defined in the function
    assert np.all(candlestick_img >= 0) 
    assert np.all(candlestick_img <= 255)
    ## First Candlestick   
    expected_pattern_c1 = np.array([[255,128,255],
                                    [255,128,255],
                                    [255,128,255],
                                    [128,128,128],
                                    [128,128,128],
                                    [128,128,128],
                                    [128,128,128],
                                    [255,255,255],
                                    [255,255,255]])
    
    assert np.array_equal(candlestick_img[:, 0:3], expected_pattern_c1)                                     
    ## Second Candlestick
    expected_pattern_c2 = np.array([[255,255,255],
                                    [255,255,255],
                                    [255,0,255],
                                    [0,0,0],
                                    [0,0,0],
                                    [255,0,255],
                                    [255,0,255],
                                    [255,0,255],
                                    [255,0,255],])
    assert np.array_equal(candlestick_img[:, 3:6], expected_pattern_c2)
        
    ## Third Candlestick
    expected_pattern_c3 = np.array([[255,255,255],
                                    [255,128,255],
                                    [128,128,128],
                                    [255,128,255],
                                    [255,255,255],
                                    [255,255,255],
                                    [255,255,255],
                                    [255,255,255],
                                    [255,255,255]])
    assert np.array_equal(candlestick_img[:, 6:9], expected_pattern_c3)
 