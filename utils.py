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

def get_A_reduced_B_reduced_f0_trans(f0, A, B, col_map_A_to_f0, col_map_B_to_f0, step_num_filt):
    A_copy = deepcopy(A)
    B_copy = deepcopy(B)
    f0_copy = deepcopy(f0.mat)
    f0_copy_dict = get_dict_from_sparse_mat(f0_copy)
    
    A_reduced_dict, ops_A, pivot_rows_A = col_redn(A_copy)
    B_reduced_dict, ops_B, pivot_rows_B = col_redn(B_copy)
    
    f0_trans_dict = transform_mat_given_ops_col_map(f0_copy_dict, ops_A, col_map_A_to_f0, shape=(f0.mat.shape[0], f0.mat.shape[1]))
    f0_trans_dict = transform_mat_given_ops_col_map(f0_trans_dict, ops_B, col_map_B_to_f0, rows=True, shape=(f0.mat.shape[0], f0.mat.shape[1]))
    f0.mat = get_sparse_matrix_from_dict(f0_trans_dict, shape=(f0.mat.shape[0], f0.mat.shape[1]))
    
    ker_idcs_A = get_ker_col_idcs(A_reduced_dict)
    ker_idcs_B = get_ker_col_idcs(B_reduced_dict)
    
    for idx in ker_idcs_A: f0.col_deaths[col_map_A_to_f0[idx]] = step_num_filt + 1
    for idx in ker_idcs_B: f0.row_deaths[col_map_B_to_f0[idx]] = step_num_filt + 1
    
    return f0, A_reduced_dict, B_reduced_dict, pivot_rows_A, pivot_rows_B


def get_next_f0_col_maps_A_B_to_f_0(A_reduced_dict, B_reduced_dict, C_next, f0, step_num_filt,
                                    pivot_rows_A, pivot_rows_B, col_map_A_to_f_0, col_map_B_to_f_0):
    
    C_next_copy = deepcopy(C_next)
    C_next_dict = get_dict_from_sparse_mat(C_next)
    C_next_copy_dict = get_dict_from_sparse_mat(C_next_copy)
    
    for col in A_reduced_dict:
        if len(A_reduced_dict[col]) > 0:
            non_zero_rows = A_reduced_dict[col]
            for row in non_zero_rows[:-1]:
                C_next_copy_dict[non_zero_rows[-1]] = list(set(C_next_copy_dict[non_zero_rows[-1]]).symmetric_difference(set(C_next_dict[row])))

    C_next_copy = get_sparse_matrix_from_dict(C_next_copy_dict, shape=(C_next.shape[0], C_next.shape[1]))
    C_next_copy_r = C_next_copy.tocsr()   
    C_next_copy_dict_r = get_dict_from_sparse_mat(C_next_copy_r)     
    for col in B_reduced_dict:
        if len(B_reduced_dict[col]) > 0:
            non_zero_rows = B_reduced_dict[col]
            for row in non_zero_rows[:-1]:
                C_next_copy_dict_r[row] = list(set(C_next_copy_dict_r[row]).symmetric_difference(set(C_next_copy_dict_r[non_zero_rows[-1]])))
    
    C_next_copy = get_sparse_matrix_from_dict_row(C_next_copy_dict_r, shape=(C_next.shape[0], C_next.shape[1]))
    C_next_copy_dict = get_dict_from_sparse_mat(C_next_copy)
    
    unstabbed_row_idcs_A, stabbed_row_corresponding_col_A = get_unstabbed_row_idcs(C_next_copy.shape[1], pivot_rows_A)
    unstabbed_row_idcs_B, stabbed_row_corresponding_col_B = get_unstabbed_row_idcs(C_next_copy.shape[0], pivot_rows_B)
       
    col_map_A_to_f_0_new, col_map_B_to_f_0_new = {}, {}
    A_next_birth_times, B_next_birth_times = [], []
    for i in range(C_next_copy.shape[1]):
        if i in unstabbed_row_idcs_A:
            col_map_A_to_f_0_new[i] = len(f0.col_births)
            f0.col_births[len(f0.col_births)] = step_num_filt+1
            f0.col_deaths[len(f0.col_deaths)] = -10
        elif i in stabbed_row_corresponding_col_A:
            col_map_A_to_f_0_new[i] = col_map_A_to_f_0[stabbed_row_corresponding_col_A[i]]
    
    for i in range(C_next_copy.shape[1]):      
        A_next_birth_times.append(f0.col_births[col_map_A_to_f_0_new[i]])
           
    for i in range(C_next_copy.shape[0]):
        if i in unstabbed_row_idcs_B:
            col_map_B_to_f_0_new[i] = len(f0.row_births)
            f0.row_births[len(f0.row_births)] = step_num_filt+1
            f0.row_deaths[len(f0.row_deaths)] = -10
        elif i in stabbed_row_corresponding_col_B:
            col_map_B_to_f_0_new[i] = col_map_B_to_f_0[stabbed_row_corresponding_col_B[i]]
    
    for i in range(C_next_copy.shape[0]):
        B_next_birth_times.append(f0.row_births[col_map_B_to_f_0_new[i]])    
    
    A_next_col_idcs_sorted = np.argsort(A_next_birth_times)
    B_next_col_idcs_sorted = np.argsort(B_next_birth_times)
    
    C_next_copy_dict_sorted = {A_next_col_idcs_sorted[k]: v for k, v in C_next_copy_dict.items()}
    
    col_map_A_to_f_0_sorted = {A_next_col_idcs_sorted[k]: v for k, v in col_map_A_to_f_0_new.items()}
    col_map_B_to_f_0_sorted = {B_next_col_idcs_sorted[k]: v for k, v in col_map_B_to_f_0_new.items()}
    
    col_map_A_to_f_0 = col_map_A_to_f_0_sorted
    col_map_B_to_f_0 = col_map_B_to_f_0_sorted
    
    C_next_copy = get_sparse_matrix_from_dict(C_next_copy_dict_sorted, shape=(C_next.shape[0], C_next.shape[1]))
    C_next_copy_r = C_next_copy.tocsr()
    C_next_copy_dict_r = get_dict_from_sparse_mat(C_next_copy_r)
    C_next_copy_dict_r_sorted = {B_next_col_idcs_sorted[k]: v for k, v in C_next_copy_dict_r.items()}
    C_next_copy = get_sparse_matrix_from_dict_row(C_next_copy_dict_r_sorted, shape=(C_next.shape[0], C_next.shape[1]))
    C_next_copy_dict_sorted = get_dict_from_sparse_mat(C_next_copy)
    
    C_next_copy_dict = C_next_copy_dict_sorted  
    
    if len(unstabbed_row_idcs_A) == 0 and len(unstabbed_row_idcs_B) == 0:
        f_0_mat_lil = f0.mat.tolil()
    elif len(unstabbed_row_idcs_A) != 0 and len(unstabbed_row_idcs_B) == 0:
        n_cols = f0.mat.shape[1]
        f0.mat = sp.hstack([f0.mat, sp.csc_array((f0.mat.shape[0], len(unstabbed_row_idcs_A)))])
        f_0_mat_lil = f0.mat.tolil()
        n_cols_new = f0.mat.shape[1]
        added_col_idcs = list(range(n_cols, n_cols_new))
        for ii,idx in enumerate(unstabbed_row_idcs_A):
            non_zero_idcs = C_next_copy[:,[idx]].nonzero()[0].tolist()
            for j in non_zero_idcs:
                f_0_mat_lil[col_map_B_to_f_0[j],added_col_idcs[ii]] = 1
    
    elif len(unstabbed_row_idcs_A) == 0 and len(unstabbed_row_idcs_B) != 0:
        n_rows = f0.mat.shape[0]
        f0.mat = sp.vstack([f0.mat, sp.csc_array((len(unstabbed_row_idcs_B), f0.mat.shape[1]))])
        f_0_mat_lil = f0.mat.tolil()
        n_rows_new = f0.mat.shape[0]
        added_row_idcs = list(range(n_rows, n_rows_new))
        for ii,idx in enumerate(unstabbed_row_idcs_B):
            non_zero_idcs = C_next_copy[idx,:].nonzero()[1].tolist()
            for j in non_zero_idcs:
                f_0_mat_lil[added_row_idcs[ii],col_map_A_to_f_0[j]] = 1
                
    else:
        n_cols = f0.mat.shape[1]
        n_rows = f0.mat.shape[0]
        f0.mat = sp.hstack([f0.mat, sp.csc_array((f0.mat.shape[0], len(unstabbed_row_idcs_A)))])
        f0.mat = sp.vstack([f0.mat, sp.csc_array((len(unstabbed_row_idcs_B), f0.mat.shape[1]))])
        f_0_mat_lil = f0.mat.tolil()
        
        n_cols_new = f0.mat.shape[1]
        added_col_idcs = list(range(n_cols, n_cols_new))
        for ii,idx in enumerate(unstabbed_row_idcs_A):
            non_zero_idcs = C_next_copy[:,idx].nonzero()[0].tolist()
            for j in non_zero_idcs:
                f_0_mat_lil[col_map_B_to_f_0[j],added_col_idcs[ii]] = 1
    
        n_rows_new = f0.mat.shape[0]
        added_row_idcs = list(range(n_rows, n_rows_new))
        for ii,idx in enumerate(unstabbed_row_idcs_B):
            non_zero_idcs = C_next_copy[idx,:].nonzero()[1].tolist()
            for j in non_zero_idcs:
                f_0_mat_lil[added_row_idcs[ii],col_map_A_to_f_0[j]] = 1
    
    f0.mat = f_0_mat_lil.tocsc()
    
    return f0

def get_next_f0_col_maps_A_B_to_f_0_trans_A_next_B_next(A_reduced_dict, B_reduced_dict, C_next, f0, step_num_filt, 
                                                        A_next, B_next, pivot_rows_A, pivot_rows_B, col_map_A_to_f_0, col_map_B_to_f_0):
    
    A_next_copy = deepcopy(A_next)
    B_next_copy = deepcopy(B_next)
    C_next_copy = deepcopy(C_next)
    
    A_next_dict = get_dict_from_sparse_mat(A_next)
    B_next_dict = get_dict_from_sparse_mat(B_next)
    C_next_dict = get_dict_from_sparse_mat(C_next)
    
    A_next_copy_dict = get_dict_from_sparse_mat(A_next_copy)
    B_next_copy_dict = get_dict_from_sparse_mat(B_next_copy)
    C_next_copy_dict = get_dict_from_sparse_mat(C_next_copy)
    
    # start = time.time()
    for col in A_reduced_dict:
        if len(A_reduced_dict[col]) > 0:
            non_zero_rows = A_reduced_dict[col]
            for row in non_zero_rows[:-1]:
                A_next_copy_dict[non_zero_rows[-1]] = list(set(A_next_copy_dict[non_zero_rows[-1]]).symmetric_difference(set(A_next_dict[row])))
                C_next_copy_dict[non_zero_rows[-1]] = list(set(C_next_copy_dict[non_zero_rows[-1]]).symmetric_difference(set(C_next_dict[row])))
    
    C_next_copy = get_sparse_matrix_from_dict(C_next_copy_dict, shape=(C_next.shape[0], C_next.shape[1]))
    C_next_copy_r = C_next_copy.tocsr()   
    C_next_copy_dict_r = get_dict_from_sparse_mat(C_next_copy_r)     
    for col in B_reduced_dict:
        if len(B_reduced_dict[col]) > 0:
            non_zero_rows = B_reduced_dict[col]
            for row in non_zero_rows[:-1]:
                B_next_copy_dict[non_zero_rows[-1]] = list(set(B_next_copy_dict[non_zero_rows[-1]]).symmetric_difference(set(B_next_dict[row])))
                # if C_next_copy_dict_r:
                C_next_copy_dict_r[row] = list(set(C_next_copy_dict_r[row]).symmetric_difference(set(C_next_copy_dict_r[non_zero_rows[-1]])))
    
    C_next_copy = get_sparse_matrix_from_dict_row(C_next_copy_dict_r, shape=(C_next.shape[0], C_next.shape[1]))
    C_next_copy_dict = get_dict_from_sparse_mat(C_next_copy)
    # end = time.time()
    # print("Time taken for transformation: ", end-start)
    
    unstabbed_row_idcs_A, stabbed_row_corresponding_col_A = get_unstabbed_row_idcs(A_next_copy.shape[1], pivot_rows_A)
    unstabbed_row_idcs_B, stabbed_row_corresponding_col_B = get_unstabbed_row_idcs(B_next_copy.shape[1], pivot_rows_B)
    
    col_map_A_to_f_0_new, col_map_B_to_f_0_new = {}, {}
    A_next_birth_times, B_next_birth_times = [], []
    for i in range(A_next_copy.shape[1]):
        if i in unstabbed_row_idcs_A:
            col_map_A_to_f_0_new[i] = len(f0.col_births)
            f0.col_births[len(f0.col_births)] = step_num_filt+1
            f0.col_deaths[len(f0.col_deaths)] = -10
        elif i in stabbed_row_corresponding_col_A:
            col_map_A_to_f_0_new[i] = col_map_A_to_f_0[stabbed_row_corresponding_col_A[i]]
            # A_next_birth_times.append(stabbed_row_corresponding_col_A[i])
    for i in range(A_next_copy.shape[1]):      
        A_next_birth_times.append(f0.col_births[col_map_A_to_f_0_new[i]])
    
    for i in range(B_next_copy.shape[1]):
        if i in unstabbed_row_idcs_B:
            col_map_B_to_f_0_new[i] = len(f0.row_births)
            f0.row_births[len(f0.row_births)] = step_num_filt+1
            f0.row_deaths[len(f0.row_deaths)] = -10
        elif i in stabbed_row_corresponding_col_B:
            col_map_B_to_f_0_new[i] = col_map_B_to_f_0[stabbed_row_corresponding_col_B[i]]
            
    for i in range(B_next_copy.shape[1]):
        B_next_birth_times.append(f0.row_births[col_map_B_to_f_0_new[i]])

    A_next_col_idcs_sorted = np.argsort(A_next_birth_times)
    B_next_col_idcs_sorted = np.argsort(B_next_birth_times)
    
    A_next_copy_dict_sorted = {A_next_col_idcs_sorted[k]: v for k, v in A_next_copy_dict.items()}
    B_next_copy_dict_sorted = {B_next_col_idcs_sorted[k]: v for k, v in B_next_copy_dict.items()}
    C_next_copy_dict_sorted = {A_next_col_idcs_sorted[k]: v for k, v in C_next_copy_dict.items()}
    
    col_map_A_to_f_0_sorted = {A_next_col_idcs_sorted[k]: v for k, v in col_map_A_to_f_0_new.items()}
    col_map_B_to_f_0_sorted = {B_next_col_idcs_sorted[k]: v for k, v in col_map_B_to_f_0_new.items()}
    
    col_map_A_to_f_0 = col_map_A_to_f_0_sorted
    col_map_B_to_f_0 = col_map_B_to_f_0_sorted
    
    C_next_copy = get_sparse_matrix_from_dict(C_next_copy_dict_sorted, shape=(C_next.shape[0], C_next.shape[1]))
    C_next_copy_r = C_next_copy.tocsr()
    C_next_copy_dict_r = get_dict_from_sparse_mat(C_next_copy_r)
    C_next_copy_dict_r_sorted = {B_next_col_idcs_sorted[k]: v for k, v in C_next_copy_dict_r.items()}
    C_next_copy = get_sparse_matrix_from_dict_row(C_next_copy_dict_r_sorted, shape=(C_next.shape[0], C_next.shape[1]))
    C_next_copy_dict_sorted = get_dict_from_sparse_mat(C_next_copy)
    
    A_next_copy_dict = A_next_copy_dict_sorted
    B_next_copy_dict = B_next_copy_dict_sorted
    C_next_copy_dict = C_next_copy_dict_sorted       
    C_next_copy = get_sparse_matrix_from_dict(C_next_copy_dict, shape=(C_next.shape[0], C_next.shape[1]))
    
    if len(unstabbed_row_idcs_A) == 0 and len(unstabbed_row_idcs_B) == 0:
        f_0_mat_lil = f0.mat.tolil()
    elif len(unstabbed_row_idcs_A) != 0 and len(unstabbed_row_idcs_B) == 0:
        n_cols = f0.mat.shape[1]
        f0.mat = sp.hstack([f0.mat, sp.csc_array((f0.mat.shape[0], len(unstabbed_row_idcs_A)))])
        f_0_mat_lil = f0.mat.tolil()
        n_cols_new = f0.mat.shape[1]
        added_col_idcs = list(range(n_cols, n_cols_new))
        for ii,idx in enumerate(unstabbed_row_idcs_A):
            non_zero_idcs = C_next_copy[:,[A_next_col_idcs_sorted[idx]]].nonzero()[0].tolist()
            for j in non_zero_idcs:
                f_0_mat_lil[col_map_B_to_f_0[j],added_col_idcs[ii]] = 1
    elif len(unstabbed_row_idcs_A) == 0 and len(unstabbed_row_idcs_B) != 0:
        n_rows = f0.mat.shape[0]
        f0.mat = sp.vstack([f0.mat, sp.csc_array((len(unstabbed_row_idcs_B), f0.mat.shape[1]))])
        f_0_mat_lil = f0.mat.tolil()
        n_rows_new = f0.mat.shape[0]
        added_row_idcs = list(range(n_rows, n_rows_new))
        for ii,idx in enumerate(unstabbed_row_idcs_B):
            non_zero_idcs = C_next_copy[[B_next_col_idcs_sorted[idx]],:].nonzero()[1].tolist()
            for j in non_zero_idcs:
                f_0_mat_lil[added_row_idcs[ii],col_map_A_to_f_0[j]] = 1
                
    else:
        n_cols = f0.mat.shape[1]
        n_rows = f0.mat.shape[0]
        f0.mat = sp.hstack([f0.mat, sp.csc_array((f0.mat.shape[0], len(unstabbed_row_idcs_A)))])
        f0.mat = sp.vstack([f0.mat, sp.csc_array((len(unstabbed_row_idcs_B), f0.mat.shape[1]))])
        f_0_mat_lil = f0.mat.tolil()
        
        n_cols_new = f0.mat.shape[1]
        added_col_idcs = list(range(n_cols, n_cols_new))
        for ii,idx in enumerate(unstabbed_row_idcs_A):
            non_zero_idcs = C_next_copy[:,[A_next_col_idcs_sorted[idx]]].nonzero()[0].tolist()
            for j in non_zero_idcs:
                f_0_mat_lil[col_map_B_to_f_0[j], added_col_idcs[ii]] = 1
    
        n_rows_new = f0.mat.shape[0]
        added_row_idcs = list(range(n_rows, n_rows_new))
        for ii,idx in enumerate(unstabbed_row_idcs_B):
            non_zero_idcs = C_next_copy[[B_next_col_idcs_sorted[idx]],:].nonzero()[1].tolist()
            for j in non_zero_idcs:
                f_0_mat_lil[added_row_idcs[ii],col_map_A_to_f_0[j]] = 1
    
    f0.mat = f_0_mat_lil.tocsc()
    
    return col_map_A_to_f_0, col_map_B_to_f_0, f0, A_next_copy_dict, B_next_copy_dict

def compute_maps_betn_bars(list_of_A_matrices, list_of_B_matrices, list_of_C_matrices):
    col_map_A_to_f_0, col_map_B_to_f_0 = {}, {}
    # assert len(list_of_A_matrices) == len(list_of_B_matrices) == len(list_of_C_matrices) - 1
    
    C = list_of_C_matrices[0]
    f0 = MatrixWithBirthDeathIdcs(mat=C, row_births = {k:0 for k in range(C.shape[0])}, 
                               row_deaths = {k:-10 for k in range(C.shape[0])}, 
                               col_births = {k:0 for k in range(C.shape[1])}, col_deaths = {k:-10 for k in range(C.shape[1])})
    for i in range(C.shape[0]): col_map_B_to_f_0[i] = i
    for i in range(C.shape[1]): col_map_A_to_f_0[i] = i
    
    A_curr = list_of_A_matrices[0]
    B_curr = list_of_B_matrices[0]
    
    for i in range(len(list_of_A_matrices)):
        f0, A_reduced, B_reduced, pivot_rows_A, pivot_rows_B = get_A_reduced_B_reduced_f0_trans(f0, A_curr, B_curr, 
                                                                      col_map_A_to_f_0, col_map_B_to_f_0, i)
        
        if i == len(list_of_A_matrices)-1:
            f0 = get_next_f0_col_maps_A_B_to_f_0(A_reduced, B_reduced, list_of_C_matrices[i+1], f0, i+1, pivot_rows_A, pivot_rows_B,
                                                 col_map_A_to_f_0, col_map_B_to_f_0)
            
            return f0
        
        col_map_A_to_f_0, col_map_B_to_f_0, f0, A_curr_dict, B_curr_dict = get_next_f0_col_maps_A_B_to_f_0_trans_A_next_B_next(A_reduced, B_reduced,
                                                                                list_of_C_matrices[i+1], f0, i, 
                                                                                list_of_A_matrices[i+1],
                                                                                list_of_B_matrices[i+1], pivot_rows_A, pivot_rows_B,
                                                                                col_map_A_to_f_0, col_map_B_to_f_0)
        
        A_curr = get_sparse_matrix_from_dict(A_curr_dict, shape=(list_of_A_matrices[i+1].shape[0], list_of_A_matrices[i+1].shape[1]))
        B_curr = get_sparse_matrix_from_dict(B_curr_dict, shape=(list_of_B_matrices[i+1].shape[0], list_of_B_matrices[i+1].shape[1]))