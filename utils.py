import scipy.sparse as sp
import numpy as np
from typing import List, Tuple
from copy import deepcopy
import time
from tqdm import tqdm
from joblib import Parallel, delayed



# A wrapper for calculating time taken by a function.
# To be used as @time_func before a function definition.
def time_func(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        print("Time taken for function: ", end-start)
        return res
    return wrapper

# A function for column reducing a matrix according to Persistence Algorithm.
def col_redn(Mat_given: sp.csc_matrix):
    
    Mat = Mat_given.copy()
    operations = {}
    rows, cols = Mat.shape
    
    if rows == 0 or cols == 0: return {}, {}, []
    
    for i in range(cols):
        operations[i] = []
    low_indices = {}
    pivot_rows_corresponding_cols = {}
    for i in range(rows):
        pivot_rows_corresponding_cols[i] = -10
    
    col_wise_rows_non_zero = np.split(Mat.indices, Mat.indptr[1:-1])
    d = dict(enumerate(col_wise_rows_non_zero,0))
    for idcs in d:
        if len(d[idcs]) > 0:
            d[idcs] = sorted(list(d[idcs]))
            low_c = d[idcs][-1]
        else:
            low_indices[idcs]= -10
            continue
        while True:
            if pivot_rows_corresponding_cols[low_c] != -10:
                
                d[idcs] = sorted(list(set(d[idcs]).symmetric_difference(set(d[pivot_rows_corresponding_cols[low_c]]))))
                operations[idcs].append(pivot_rows_corresponding_cols[low_c])
                
                if len(d[idcs]) > 0:
                    low_c = d[idcs][-1]
                else:
                    low_indices[idcs]= -10
                    break
            else:
                low_indices[idcs]=low_c
                pivot_rows_corresponding_cols[low_c] = idcs
                break
    
    return d, operations, list(low_indices.values())


''' A sparse matrix can be stored as a dictionary where each key is a column index 
    and the value is a list of row indices where the column has non-zero entries.
    Since we are working with Z_2 coefficients, the non-zero entries are always 1.'''
    
# This function converts a dictionary to a sparse matrix of the given shape.
def get_sparse_matrix_from_dict(d:dict, shape=None):
    row_ind_col_ind = []
    for col in d:
        for rows in d[col]:
            row_ind_col_ind.append(np.array([rows,col]))
    row_ind_col_ind = np.array(row_ind_col_ind)
    if len(row_ind_col_ind) > 0:
        data = np.ones(len(row_ind_col_ind))
        return sp.csc_matrix((data, (row_ind_col_ind[:,0], row_ind_col_ind[:,1])), shape=shape)
    else:
        return sp.csc_array(shape)

# This function converts a dictionary to a sparse matrix of the given shape. 
# However, the dictionary is row-wise instead of column-wise.
def get_sparse_matrix_from_dict_row(d:dict, shape=None):
    row_ind_col_ind = []
    for row in d:
        for cols in d[row]:
            row_ind_col_ind.append(np.array([row,cols]))
    row_ind_col_ind = np.array(row_ind_col_ind)
    if len(row_ind_col_ind)>0:
        data = np.ones(len(row_ind_col_ind))
        return sp.csc_matrix((data, (row_ind_col_ind[:,0], row_ind_col_ind[:,1])), shape=shape)
    else:
        return sp.csc_array(shape)
    
# This function gives the dictionary representation of the
# given matrix.
def get_dict_from_sparse_mat(mat: sp.csc_matrix):
    if mat.shape[0] == 0 or mat.shape[1] == 0: return {}
    col_wise_rows_non_zero = np.split(mat.indices, mat.indptr[1:-1])
    mat_d = dict(enumerate(col_wise_rows_non_zero,0))
    return mat_d


# This function sorts the columns of the given matrix according to the birth-times of the columns.
def sort_matrix_cols_acc_to_birth_times(mat, birth_times):
    sorted_cols = np.argsort(birth_times)
    return mat[:, sorted_cols], sorted_cols


def transform_mat_given_ops_col_map(mat_d:dict, ops:dict, col_map:dict, rows=False, shape=None):
    if rows:
        mat = get_sparse_matrix_from_dict(mat_d, shape=shape)
        mat_r = mat.tocsr()
        row_wise_cols_non_zero = np.split(mat_r.indices, mat_r.indptr[1:-1])
        mat_d = dict(enumerate(row_wise_cols_non_zero,0))
        for col in ops:
            if len(ops[col]) > 0:
                for cols_to_be_added in ops[col]:
                    mat_d[col_map[cols_to_be_added]] = list(set(mat_d[col_map[cols_to_be_added]]).symmetric_difference(set(mat_d[col_map[col]])))
        
        mat = get_sparse_matrix_from_dict_row(mat_d, shape=shape)
        col_wise_rows_non_zero = np.split(mat.indices, mat.indptr[1:-1])
        mat_d = dict(enumerate(col_wise_rows_non_zero,0))
    
    else:
        for col in ops:
            if len(ops[col]) > 0:
                for cols_to_be_added in ops[col]:
                    mat_d[col_map[col]] = list(set(mat_d[col_map[col]]).symmetric_difference(set(mat_d[col_map[cols_to_be_added]])))
    
    return mat_d

# This function gives the indices of the rows that do not 
# have a pivot element (unstabbed rows). It also gives the 
# corresponding column index of the pivot element for each
# stabbed row. This information is extracted from the list
# pivot rows.
def get_unstabbed_row_idcs(n_rows:int, pivot_rows):
    stabbed_row_corresponding_col = {}
    pivot_rows = np.array(pivot_rows)
    pivot_row_col_idcs = np.where(pivot_rows != -10)[0]
    pivot_rows = pivot_rows[pivot_rows != -10]
    stabbed_row_idcs = pivot_rows
    for i in range(len(pivot_row_col_idcs)): stabbed_row_corresponding_col[pivot_rows[i]] = pivot_row_col_idcs[i]
    unstabbed_row_idcs = list(set(range(n_rows)) - set(stabbed_row_idcs))
    
    return unstabbed_row_idcs, stabbed_row_corresponding_col

# This function computes the indices of the columns that form 
# the null space of the given matrix (stored as a dictionary).
def get_ker_col_idcs(mat_d:dict):
    ker_cols = []
    for col in mat_d:
        if len(mat_d[col]) == 0: ker_cols.append(col)
    return ker_cols



class MatrixWithBirthDeathIdcs:
    def __init__(self, mat: sp.csc_matrix, row_births: dict, row_deaths: dict, col_births: dict, col_deaths: dict):
        self.mat = mat
        self.row_births = row_births
        self.col_births = col_births
        self.row_deaths = row_deaths
        self.col_deaths = col_deaths

def read_off_bars(mat_r_d, col_degs, row_degs, pivot_rows_cols):
    bars = []
    pivot_rows = []
    for i in range(len(pivot_rows_cols)):
        if pivot_rows_cols[i] != -10:
            pivot_rows.append(pivot_rows_cols[i])
            if row_degs[pivot_rows_cols[i]] < col_degs[i]:        # To avoid unimportant bars such as [1,1)
                bars.append((row_degs[pivot_rows_cols[i]], col_degs[i]))
    non_pivot_rows = list(set(list(range(len(mat_r_d)))) - set(pivot_rows))
    for i in non_pivot_rows:
        bars.append((row_degs[i], np.inf))
    
    return bars
