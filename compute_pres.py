import scipy.sparse as sp
import numpy as np
from copy import deepcopy
import time
from tqdm import tqdm
from joblib import Parallel, delayed
from utils import *

'''Given three sequences of matrices A_i, B_i, C_i (as mentioned in the paper), 
this file has the functions required to compute the presentations of the barcodes 
the maps. This algorithm maintains a matrix f0 which encodes this information as follows:
Each column corresponds to a bar running in the sequence of matrices A_i, each row to a bar
running in the sequence B_i and the entries in f0 indicate the maps between the bars. Thus,
each row and each column have birth times and death times corresponding to the births and 
deaths of the bars as encoded in the sequences of A_i's and B_i's.'''


# This performs the same row operations on f0 that correspond to the column operations required 
# to reduce B and the same columnn operations on f0 that correspond to the column opereations 
# required to reduce A. 
def get_A_reduced_B_reduced_f0_trans(f0: MatrixWithBirthDeathIdcs, A: sp.csc_matrix, B: sp.csc_matrix, 
                                     col_map_A_to_f0: dict, col_map_B_to_f0: dict, step_num_filt: int):
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


# This function maintains and updates dictionaries col_map_A_to_f0 and col_map_B_to_f0
# which have the columns of A and B mapped out to the columns and rows of f0 respectively.
# 
def get_next_f0_col_maps_A_B_to_f_0_trans_A_next_B_next(A_reduced_dict: dict, B_reduced_dict: dict, 
                                                        C_next: sp.csc_matrix, f0: MatrixWithBirthDeathIdcs, 
                                                        step_num_filt: int, A_next: sp.csc_matrix, B_next: sp.csc_matrix, 
                                                        pivot_rows_A: list, pivot_rows_B: list, 
                                                        col_map_A_to_f_0: dict, col_map_B_to_f_0: dict):
    
    A_next_copy = deepcopy(A_next)
    B_next_copy = deepcopy(B_next)
    C_next_copy = deepcopy(C_next)
    
    A_next_dict = get_dict_from_sparse_mat(A_next)
    B_next_dict = get_dict_from_sparse_mat(B_next)
    C_next_dict = get_dict_from_sparse_mat(C_next)
    
    A_next_copy_dict = get_dict_from_sparse_mat(A_next_copy)
    B_next_copy_dict = get_dict_from_sparse_mat(B_next_copy)
    C_next_copy_dict = get_dict_from_sparse_mat(C_next_copy)
    
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
                C_next_copy_dict_r[row] = list(set(C_next_copy_dict_r[row]).symmetric_difference(set(C_next_copy_dict_r[non_zero_rows[-1]])))
    
    C_next_copy = get_sparse_matrix_from_dict_row(C_next_copy_dict_r, shape=(C_next.shape[0], C_next.shape[1]))
    C_next_copy_dict = get_dict_from_sparse_mat(C_next_copy)
    
    
    # Unstabbed row indices start a new bar and stabbed rows continue the existing bar
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
    
    # We add a column to f0 for every unstabbed row index in A and a row to f0 for every unstabbed row index in B.
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



def get_next_f0_col_maps_A_B_to_f_0(A_reduced_dict: dict, B_reduced_dict: dict, C_next: sp.csc_matrix, 
                                    f0: MatrixWithBirthDeathIdcs, step_num_filt: int,
                                    pivot_rows_A: list, pivot_rows_B: list, 
                                    col_map_A_to_f_0: dict, col_map_B_to_f_0: dict):
    
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


# Given three lists of matrices A_i's, B_i's, C_i's, this function computes the barcodes of 
# A_i's and B_i's and the maps between the barcodes.
def compute_maps_betn_bars(list_of_A_matrices, list_of_B_matrices, list_of_C_matrices):
    col_map_A_to_f_0, col_map_B_to_f_0 = {}, {}
    assert len(list_of_A_matrices) == len(list_of_B_matrices) == len(list_of_C_matrices) - 1
    
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



      
