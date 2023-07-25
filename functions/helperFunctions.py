# imports
import numpy as np


# Find nearest value in numpy array
# input:
#   array: array to search in.
#   value: 
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

# interpolation{‘linear’, ‘lower’, ‘higher’, ‘midpoint’, ‘nearest’} This optional parameter specifies the interpolation method to use when the desired quantile lies between two data points i < j.
# MATLAB apparently uses midpoint interpolation by default. NumPy and R use linear interpolation by default.
# var_th = np.quantile(good_channels_var,0.95,interpolation='midpoint')
# UGH!!!!!!!
# https://stackoverflow.com/questions/24764966/numpy-percentile-function-different-from-matlabs-percentile-function


def quantile(x, q):
    n = len(x)
    y = np.sort(x)
    return(np.interp(q, np.linspace(1 / (2 * n), (2 * n - 1) / (2 * n), n), y))


# https://gist.github.com/fasiha/eff0763ca25777ec849ffead370dc907
# In Python and Matplotlib, an image like this is a little harder to obtain, because by default, Matplotlib’s imshow forces square pixels.
def extents(f):
    delta = f[1] - f[0]
    return [f[0] - delta / 2, f[-1] + delta / 2]
