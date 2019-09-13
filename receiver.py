import numpy as np
import optical_line
import line_functions as lf
from numpy.fft import *
import qam_gray_coding as qam
import generate_signal
import qam_gray_coding
import functools
import my_functools


class ReceiverSim:
    """Class for managing receiver parameters"""
    def __init__(self, mod_order, n_symbol_batch, n_discarded_symbols, discarding_factor = 2, up_sample = 2^4, receiver_oversampling_rate = 4):
        self.bits_per_symbol = np.log2(mod_order)
        self.n_symbol_batch = n_symbol_batch
        self.n_discarded_symbols = n_discarded_symbols
        self.discarding_factor = discarding_factor
        self.n_symbol_band=(n_symbol_batch-3*n_discarded_symbols)/2 # number of symbols in each "band". ie number of symbols we analyze each time
        self.n_symbol_gen=n_symbol_batch+2*n_discarded_symbols # number of symbols to generate to avoid periodicity
        # explanation: dfSim_Hz = (B_one_channel_Hz * (1 + roloff) + dfSim_Hz) / N_symbol_gen;
        self.df_sim_hz = self.width_one_channel_hz * (1 + self.roloff) / (self.n_symbol_gen - 1) # this sets the scale of the dispersion etc
        self.receiver_oversampling_rate = receiver_oversampling_rate
        self.roloff_bins_rec = self.roloff_bins_sim
        self.receiver_df_Hz = self.df_sim_hz
        self.receiver_vector_length = self.receiver_oversampling_rate * self.n_symbol_gen
        self.width_rec_hz = self.df_sim_hz * self.receiver_oversampling_rate * (self.n_symbol_gen - 1)  # = R_upsmpl*B_one_channel_Hz*(1+roloff)
        self.receiver_dt_sec = 1 / self.width_rec_hz
        # batch vectors
        self.batch_df_Hz = self.B_one_channel_Hz * (1 + self.roloff) / (self.n_symbol_batch - 1)


class Signal:
    """Class for managing signal parameters"""
    def __init__(self, mod_order, n_symbol_gen, roloff_bins, width_one_channel_hz, guard_band_hz, df_hz):
        self.mod_order = mod_order
        self.n_symbol_gen = n_symbol_gen
        self.symbols_bin = np.random.randint(low=0, high=mod_order-1, size=n_symbol_gen)
        self.roloff_bins = roloff_bins # roll-off on both sides
        self.width_one_channel_hz = width_one_channel_hz
        self.roloff = roloff_bins / n_symbol_gen # square root raised cosine pulse roloff parameter
        self.Minimum_phase_shift_bins = n_symbol_gen / 2 + roloff_bins / 2 + 1
        self.guard_band_hz = guard_band_hz
        self.guard_band_bins = round(self.guard_band_hz / df_hz)


class OptimizationEnv:
    """Class for managing EKK optimization"""
    def __init__(self, iterations = 50, gamma_lb = 0.7):
        self.iterations = iterations
        self.gamma_lb = gamma_lb


class EKKSim:
    """Class for managing EKK simulation"""
    def __init__(self,  optline, receiver):
        self.optline = optline
        self.receiver = receiver