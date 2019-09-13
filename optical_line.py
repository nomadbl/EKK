import numpy as np
import line_functions as lf
from numpy.fft import *
import qam_gray_coding as qam
import generate_signal
import qam_gray_coding
import functools
import my_functools


class OpticalLine:
    """Class for managing optical parameters"""
    def __init__(self, mod_order, n_symbol_batch, n_discarded_symbols, n_channels, roloff_bins = 2**5, up_sample = 2 ** 4, relative_dispersion_line1 = 1, relative_dispersion_line2 = 1.2, non_linear = 1.3e-3,
                 a1 = 0.22 / (10 * np.log10(np.exp(1))), span_length_km = 100, beta_2_pico_sec_sq_per_km = 21, coupling_loss_db = 0,
                 noise_figure_db = 5, gain_db = 22, n_spans = 1, width_one_channel_hz = 32 * 10e9, osnr_bandwidth_hz = 12.5 * 10e9, guard_band_hz=16 * 10e9):
        # signal params
        self.mod_order = mod_order
        self.n_symbol_batch = n_symbol_batch
        self.n_discarded_symbols = n_discarded_symbols
        self.n_channels = n_channels
        self.roloff_bins = roloff_bins
        self.guard_band_hz = guard_band_hz
        self.n_symbol_band = (n_symbol_batch - 3 * n_discarded_symbols) / 2  # number of symbols in each "band". ie number of symbols we analyze each time
        self.n_symbol_gen = n_symbol_batch + 2 * n_discarded_symbols  # number of symbols to generate to avoid periodicity
        self.width_one_channel_hz = width_one_channel_hz

        # optical params
        self.non_linear = non_linear # 1/mw/km
        self.a1 = a1
        self.beta_2_sec_sq_per_km = beta_2_pico_sec_sq_per_km * 1e-24
        self.coupling_loss_db = coupling_loss_db # Added noise due to coupling coefficients at connection
        self.noise_figure_db = noise_figure_db
        self.gain_db = gain_db
        self.osnr_bandwidth_hz = osnr_bandwidth_hz
        self.n_spans = n_spans
        self.power_noise_osnr_width_dbm = -58 + self.noise_figure_db + self.gain_db + 10 * np.log10(self.n_spans) + self.coupling_loss_db
        self.power_noise_dbm = self.power_noise_osnr_width_dbm + 10 * np.log10(width_one_channel_hz / self.osnr_bandwidth_hz)
        self.power_noise_mwatt = 10 ** (self.power_noise_dbm / 10)  # this is in mWatt
        self.relative_dispersion_line1 = relative_dispersion_line1
        self.relative_dispersion_line2 = relative_dispersion_line2
        self.span_length_km = span_length_km # propagation length in km
        self.line2_length_km = self.span_length_km * (self.relative_dispersion_line2 - self.relative_dispersion_line1)
        self.up_sample = up_sample
        #self.width_sim_hz = self.df_sim_Hz * up_sample * (self.n_symbol_gen - 1)
        self.width_sim_hz = up_sample * width_one_channel_hz * (1 + self.signal.roloff)
        self.dt_sec = 1 / self.width_sim_hz
        self.df_hz = self.width_sim_hz / self.n_symbol_gen

        g_sim_base_band = generate_signal.raisedcosine(roloff_bins, self.n_symbol_gen, up_sample)
        g_sim_base_band = np.divide(g_sim_base_band, g_sim_base_band[1])
        self.g_sim_base_band = fftshift(g_sim_base_band)

        self.signal = Signal(mod_order, self.n_symbol_gen, roloff_bins, width_one_channel_hz, guard_band_hz, self.df_hz)
        channel_shift = lambda channel_vec, n: generate_signal.move_band(channel_vec,
                                                                         n * (self.n_symbol_gen + self.roloff_bins
                                                                              + self.signal.guard_band_bins))
        code, decode = qam_gray_coding.qam_gray_coding(self.mod_order)

        side_channel_list = np.setdiff1d(np.arange(0,n_channels)-np.floor(n_channels/2), np.zeros(1))
        self.signal_vec = np.zeros(len(self.g_sim_base_band))
        gi = lambda x: channel_shift(g_sim_base_band, int(x))
        # build up the signal of all channels
        self.signal_vec = np.add(generate_signal.analog_signal(code(self.signal.symbols_bin), self.g_sim_base_band),
                                 functools.reduce(np.add,
                                                  my_functools.zipwith(generate_signal.analog_signal,
                                                                       map(lambda x: code(np.random.random(
                                                                           self.signal.n_symbol_gen)), side_channel_list),
                                                                       map(np.vectorize(gi),side_channel_list))))


    def dispersion_line1(self, signal):
        return lf.dispersion(signal, self.beta_2_sec_sq_per_km, self.span_length_km, self.dt_sec)


    def dispersion_line2(self, signal):
        return lf.dispersion(signal, self.beta_2_sec_sq_per_km, self.line2_length_km, self.dt_sec)


    def propagation(self):
        self.signal_vec = lf.splitstep(self.signal_vec, self.dt_sec, self.span_length_km, 1, self.beta_2_sec_sq_per_km, self.non_linear, self.a1)





