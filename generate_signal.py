import numpy as np
from numpy.fft import *


def move_band(fs,nb):
    return ifft(np.roll(fft(fs), nb))  # move baseband (-)nb frequency units higher (lower)


def raisedcosine(roloff_bins, n_symbols, sps):
    beta = roloff_bins / n_symbols
    shape = np.vectorize(lambda x: raisedcosine_shape(x, beta))
    fs = shape(fftfreq(n_symbols * sps, 1))
    return ifft(fs)


def squareroot_raisedcosine(roloff_bins, n_symbols, sps):
    beta = roloff_bins / n_symbols
    shape = np.vectorize(lambda x: raisedcosine_shape(x, beta))
    fs = np.sqrt(shape(fftfreq(n_symbols * sps, 1)))
    fs = ifft(fs)
    fsq = ifft(np.multiply(fft(fs),fft(fs)))
    return np.divide(fs , np.sqrt(fsq(1)))


# fT=n/Ns
def raisedcosine_shape(fT, beta):
    if np.abs(fT) <= (1 - beta) / 2:
        x = 1
    elif (1 + beta) / 2 > abs(fT) & abs(fT) > (1-beta) / 2:
        x = (1 + np.cos(np.pi / beta * (abs(fT) - (1 - beta)/2))) / 2
    else:
        x = 0
    return x


def upsample(vec, rate):
    mat = np.zeros((len(vec),rate))
    mat[0:len(vec),0] = vec
    return np.ravel(mat)


def analog_signal(symbols_vec, pulse_vec):
    up_sample_rate = len(pulse_vec) / len(symbols_vec)
    symbols_vec = upsample(symbols_vec, up_sample_rate)
    return ifft(np.multiply(fft(symbols_vec), fft(pulse_vec)))