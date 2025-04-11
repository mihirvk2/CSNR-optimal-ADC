import numpy as np

def int_to_bits(x, nbits=8,flip=False):
    # Converts signed integer vector x to bits matrix (N x nbits)
    # Rightmost bit is LSB and leftmost is MSB if flip is False
    # Leftmost bit is LSB and rightmost is MSB if flip is True
        nrows = len(x)
        ncols = nbits
        y = np.zeros(shape=(nrows,ncols))
        for i in range(nrows):
            x_curr_int = int(x[i])
            temp = np.binary_repr(x_curr_int ,nbits)
            if(flip): y[i,:] = np.flip(np.array([int(char) for char in temp]))
            else: y[i,:] = np.array([int(char) for char in temp])
        return y

def quantize_to_signed_int(x,nbits,input_range = 1):
    # Quantizes a real floating point input scalar to real signed int scalar
    # Input range assumed to be always symmetric around 0
    scale = 2 * input_range / (2**nbits) # Calculate LSB
    x_clipped = np.clip(x, -input_range, input_range - scale)
    level = np.round(x_clipped / scale)
    return level, scale