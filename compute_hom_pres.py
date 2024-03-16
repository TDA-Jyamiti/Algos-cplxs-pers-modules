import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
import time
from utils import *

'''Given a complex of presentations, these functions compute the homology of the complex.
We encode the complex as two matrices, f0 and g0, where each row and each column has a 
birth time and a death time. The notations used for the matrices are consistent with the
ones used in the paper.'''

def get_q_r(f0: MatrixWithBirthDeathIdcs, g0: MatrixWithBirthDeathIdcs):
    r = sp.identity(g0.mat.shape[0], format='csc')
    q = sp.identity(f0.mat.shape[0], format='csc')
    
    return q, r

# It need not be necessary that a complex of graded modules is a complex of presentations.
# To fix this, we need to add zero bars as mentioned in the paper.
def fix_zero_cols_g0_f0(f0: MatrixWithBirthDeathIdcs,
                    g0: MatrixWithBirthDeathIdcs):
    g0f0 = sp.csc_matrix((g0.mat @ f0.mat).A % 2)
    g0f0_d = get_dict_from_sparse_mat(g0f0)
    ker_cols = get_ker_col_idcs(g0f0_d)
    non_ker_cols = list(set(np.arange(g0f0.shape[1])) - set(ker_cols))
    g0_d = get_dict_from_sparse_mat(g0.mat)
    f0_r_d = get_dict_from_sparse_mat(f0.mat.tocsr())
    for j in tqdm(non_ker_cols):
        g0_d[len(g0_d)] = g0f0_d[j]        
        f0_r_d[len(f0_r_d)] = [j]
        f0.row_births[len(f0.row_births)] = f0.col_births[j]
        f0.row_deaths[len(f0.row_deaths)] = f0.col_deaths[j]
        g0.col_births[len(g0.col_births)] = f0.col_births[j]
        g0.col_deaths[len(g0.col_deaths)] = f0.col_deaths[j]
    f0.mat = get_sparse_matrix_from_dict_row(f0_r_d, shape=(len(f0.row_births), f0.mat.shape[1]))
    g0.mat = get_sparse_matrix_from_dict(g0_d, shape=(g0.mat.shape[0], len(g0.col_births)))
    f0.mat = sp.csc_matrix(f0.mat.A % 2)
    g0.mat = sp.csc_matrix(g0.mat.A % 2)
    return f0, g0


def get_bars_hom_cplx_pres(f0: MatrixWithBirthDeathIdcs, g0: MatrixWithBirthDeathIdcs):
    f0, g0 = fix_zero_cols_g0_f0(f0, g0)
    f0.col_births = np.array(list(f0.col_births.values()))
    f0.col_deaths = np.array(list(f0.col_deaths.values()))
    f0.row_births = np.array(list(f0.row_births.values()))
    f0.row_deaths = np.array(list(f0.row_deaths.values()))
    g0.col_births = np.array(list(g0.col_births.values()))
    g0.col_deaths = np.array(list(g0.col_deaths.values()))
    g0.row_births = np.array(list(g0.row_births.values()))
    g0.row_deaths = np.array(list(g0.row_deaths.values()))
    
    
    q, r = get_q_r(f0,g0)
    g1 = g0.mat
    g0_plus_r = sp.hstack([g0.mat, r])
    
    col_degs_g0_plus_r = np.hstack([f0.row_births, g0.row_deaths])
    row_degs_g0_plus_r = g0.row_births
    g0_plus_r, sorted_cols = sort_matrix_cols_acc_to_birth_times(g0_plus_r, col_degs_g0_plus_r)
    
    col_degs_g0_plus_r = col_degs_g0_plus_r[sorted_cols]
    
    q_concat_g1 = sp.vstack([q, g1])
    
    f0_concat_zero = sp.vstack([f0.mat, sp.csc_matrix((q_concat_g1.shape[0]-f0.mat.shape[0], f0.mat.shape[1]))])
    
    f0_concat_zero_plus_q_concat_g1 = sp.hstack([f0_concat_zero, q_concat_g1])
    col_degs_f0_concat_zero_plus_q_concat_g1 = np.hstack([f0.col_births, g0.col_deaths])
    row_degs_f0_concat_zero_plus_q_concat_g1 = np.hstack([f0.row_births, g0.row_deaths])
    
    f0_concat_zero_plus_q_concat_g1, sorted_cols = sort_matrix_cols_acc_to_birth_times(f0_concat_zero_plus_q_concat_g1, col_degs_f0_concat_zero_plus_q_concat_g1)    
    col_degs_f0_concat_zero_plus_q_concat_g1 = col_degs_f0_concat_zero_plus_q_concat_g1[sorted_cols]
    
    g0_plus_r_reduced_d, ops, pivot_rows = col_redn(g0_plus_r)
    g0_plus_r_ker_cols = get_ker_col_idcs(g0_plus_r_reduced_d)
    
    f0_concat_zero_plus_q_concat_g1_restricted = f0_concat_zero_plus_q_concat_g1[g0_plus_r_ker_cols, :]
      
    f0_concat_zero_plus_q_concat_g1_restricted_reduced_d, __, pivot_rows_cols = col_redn(f0_concat_zero_plus_q_concat_g1_restricted)
    row_degs_f0_concat_zero_plus_q_concat_g1_restricted_reduced = row_degs_f0_concat_zero_plus_q_concat_g1[g0_plus_r_ker_cols]
    
    f0_concat_zero_plus_q_concat_g1_restricted_reduced_r_d = get_dict_from_sparse_mat(get_sparse_matrix_from_dict(
                                                            f0_concat_zero_plus_q_concat_g1_restricted_reduced_d, 
                                                            shape=(len(g0_plus_r_ker_cols), f0_concat_zero_plus_q_concat_g1.shape[1])).tocsr())
    
    bars = read_off_bars(f0_concat_zero_plus_q_concat_g1_restricted_reduced_r_d, 
                         col_degs_f0_concat_zero_plus_q_concat_g1, 
                         row_degs_f0_concat_zero_plus_q_concat_g1_restricted_reduced,
                         pivot_rows_cols)
    
    return bars
