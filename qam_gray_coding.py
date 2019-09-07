import functools
from numpy import multiply
import numpy as np


def count_set_bits(n):
    # base case
    if n == 0:
        return 0
    else:
        return (n & 1) + count_set_bits(n >> 1)


def gray_code(n_bits):
    return np.fromiter(map(lambda i: i ^ i >> 1,
                           range(0, 2 ** n_bits)),
                       int)


def is_gray_neighbors(x, y):
    dif = np.absolute(np.subtract(x, y))
    return count_set_bits(dif) == 1


def neighborhood(mat, index):
    hood = []
    mat_x, mat_y = mat.shape
    if index[0] < mat_x-1:
        if mat[(index[0] + 1, index[1])] > -1:
            hood = hood + [mat[(index[0] + 1, index[1])]]
    if index[0] > 0:
        if mat[(index[0] - 1, index[1])] > -1:
            hood = hood + [mat[(index[0] - 1, index[1])]]
    if index[1] < mat_y-1:
        if mat[(index[0], index[1] + 1)] > -1:
            hood = hood + [mat[(index[0], index[1] + 1)]]
    if index[1] > 0:
        if mat[(index[0], index[1] - 1)] > -1:
            hood = hood + [mat[(index[0], index[1] - 1)]]

    return hood


def populate_gray_mat(mat, m, available_list, index):
    mat_length = np.sqrt(m) - 1  # index is in [0,line_length]
    if mat[index] != -1:  # already populated, lets find unpopulated slot
        if index[1] < mat_length:  # we can go "down"
            new_index = (index[0], index[1] + 1)
            return populate_gray_mat(mat, m, available_list, new_index)
        elif index[0] < mat_length:  # we can go "right"
            new_index = (index[0] + 1, 0)
            return populate_gray_mat(mat, m, available_list, new_index)
        else:  # everything populated! done!
            return mat
    else:  # populate slot!
        hood = neighborhood(mat, index)
        candidate = available_list[0]
        neighborhood_check = list(map(lambda i: is_gray_neighbors(candidate, i),
                                      hood))
        neighborhood_check = functools.reduce(lambda a, b: a and b,
                                              neighborhood_check, True)
        if neighborhood_check:
            # populate and proceed
            mat[index] = candidate
            available_list = np.setdiff1d(available_list, np.array(candidate), True)
            return populate_gray_mat(mat, m, available_list, index)
        else:
            # run again with a different candidate
            return populate_gray_mat(mat, m, np.roll(available_list, 1), index)


def qam_center(z_list):
    return np.subtract(z_list, np.mean(z_list) )


def qam_normalize(z_list_in):
    z_list = qam_center(z_list_in)
    p = sum(np.square(np.absolute(z_list)))
    return np.divide(z_list, np.sqrt(p))


# returns a matrix mapping indexes to qam
def n_qam_ind_to_qam(n):
    mat_length = int(np.sqrt(n))
    x = np.arange(mat_length)
    y = np.arange(mat_length)
    xv, yv = np.meshgrid(x, y)
    xv = xv.ravel()
    yv = yv.ravel()
    # make index list
    yv = np.multiply(yv, 1j)
    z_list = np.add(xv, yv)
    z_list = qam_normalize(z_list)
    return z_list.reshape((mat_length, mat_length))


# returns a function mapping indexes to binary
def qam_gray_mat(m):
    line_length = int(np.sqrt(m))
    gray_x = gray_code(int(np.log2(line_length)))
    gray_y = multiply(gray_x, 2 ** (line_length-1))
    fill_list = np.setdiff1d(np.arange(m), gray_x, True)
    fill_list = np.setdiff1d(fill_list, gray_y, True)
    mat = np.full((line_length, line_length), -1)
    mat[0, :] = gray_x
    mat[:, 0] = gray_y
    return populate_gray_mat(mat, m, fill_list, (0, 0))


# output two functions
# decode: binary -> QAM
# code: QAM -> binary
# QAM is a complex number
# binary is an integer
def qam_gray_coding(n):
    # mat can be seen as a function index -> binary
    bin_mat = qam_gray_mat(n)
    # function index -> qam
    qam_mat = n_qam_ind_to_qam(n)
    # because the two matrices have corresponding elements in the same indices,
    # we can make them into arrays with the same property
    bin_arr = bin_mat.ravel()
    qam_arr = qam_mat.ravel()
    # build dictionaries to form the functions
    code_dict = dict(zip(qam_arr, bin_arr))
    decode_dict = dict(zip(bin_arr, qam_arr))
    # convert to functions
    code_func = lambda qam: code_dict[qam]
    decode_func = lambda binary: decode_dict[binary]
    # return functions
    return code_func, decode_func
