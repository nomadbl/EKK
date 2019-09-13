import functools
import numpy as np
import matplotlib.pyplot as plt
from qam_gray_coding import to_binary_string, qam_gray_coding


# plot data points along with constellation
# binary = True also adds in the binary representation of constellation points
def qam_plot(p_arr, m, binary=False, fignum = None):
    if fignum != None:
        plt.figure(fignum)
    else:
        plt.figure()
    code, decode = qam_gray_coding(m)
    vdecode = np.vectorize(decode)
    qam_arr = vdecode(np.arange(m))
    plt.scatter(np.real(p_arr), np.imag(p_arr), s=4, c='b', alpha=0.5)
    plt.scatter(np.real(qam_arr), np.imag(qam_arr), s=4, c='r')
    if binary:
        for i in range(m):
            plt.text(np.real(qam_arr[i]), np.imag(qam_arr[i]), to_binary_string(i), va='top', ha='center')
    plt.show()
