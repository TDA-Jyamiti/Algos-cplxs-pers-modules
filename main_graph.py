import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed
import networkx as nx
from utils import *
from compute_pres import *


def get_graph(num_nodes):
    """Get a random graph with num_nodes nodes as edge_list."""
    
    graph = nx.gnp_random_graph(num_nodes, 0.4)
    edge_list = list(graph.edges)
    return edge_list


def compute_bars_birth_rows(num_rows, filt_len=100):
    row_birth_times = np.random.randint(0, filt_len//2, size=num_rows-1)
    row_birth_times = np.sort(np.append(row_birth_times, 0))
    
    return row_birth_times

def compute_bars_birth_cols(row_birth_times, Mat, filt_len=100):
    col_birth_times = []
    col_wise_rows_non_zero = np.split(Mat.indices, Mat.indptr[1:-1])
    d = dict(enumerate(col_wise_rows_non_zero,0))
    for col in d:
        if len(d[col]>0):
           col_birth_times.append(np.random.randint(row_birth_times[d[col][-1]], filt_len)) 
        else:
            col_birth_times.append(np.random.randint(0, filt_len))
    
    return np.array(col_birth_times)

def generate_filt_matrices(f0, col_birth_times, row_birth_times, filt_len=100):
    list_of_A_matrices = []
    list_of_B_matrices = []
    list_of_C_matrices = []
    cond_row = (row_birth_times == 0)
    cond_col = (col_birth_times == 0)
    cond_true_row = np.where(cond_row)[0]
    cond_true_col = np.where(cond_col)[0]
    C_prev = f0[cond_true_row, :][:, cond_true_col]
    list_of_C_matrices.append(C_prev)
    for i in range(1, filt_len):
        C_prev = list_of_C_matrices[-1]
        cond_row = cond_row | (row_birth_times == i)
        cond_col = cond_col | (col_birth_times == i)
        cond_true_row = np.where(cond_row)[0]
        cond_true_col = np.where(cond_col)[0]
        C_curr = f0[cond_true_row, :][:, cond_true_col]
        A = sp.identity(C_prev.shape[1], format='csc')
        A = sp.vstack((A, sp.csc_array((C_curr.shape[1] - C_prev.shape[1], C_prev.shape[1]))))
        B = sp.identity(C_prev.shape[0], format='csc')
        B = sp.vstack((B, sp.csc_array((C_curr.shape[0] - C_prev.shape[0], C_prev.shape[0]))))
        list_of_C_matrices.append(C_curr)
        list_of_A_matrices.append(A)
        list_of_B_matrices.append(B)
        
    return list_of_A_matrices, list_of_B_matrices, list_of_C_matrices


def compute_bars_death_rows(num_rows, filt_len=100):
    row_death_times = np.sort(np.random.randint(filt_len//2, filt_len, size=num_rows))
    
    return row_death_times

def compute_bars_death_cols(row_death_times, filt_len=100):
    col_death_times = []
    for i in range(len(row_death_times)):
        col_death_times.append(np.random.randint(row_death_times[i], filt_len))
    col_death_times = np.sort(np.array(col_death_times))
    
    return col_death_times

def generate_filt_matrices2(f0, col_birth_times, row_birth_times, col_death_times, row_death_times, filt_len=100):
    list_of_A_matrices = []
    list_of_B_matrices = []
    list_of_C_matrices = []
    col_death_times = col_birth_times + 2
    row_death_times = row_birth_times + 2
    cond_row = (row_birth_times == 0)
    cond_row_death = ~(row_death_times == 0)
    cond_col = (col_birth_times == 0)
    cond_col_death = ~(col_death_times == 0)
    cond_row = cond_row & cond_row_death
    cond_col = cond_col & cond_col_death
    cond_true_row = np.where(cond_row)[0]
    cond_true_col = np.where(cond_col)[0]
    C_prev = f0[cond_true_row, :][:, cond_true_col]
    list_of_C_matrices.append(C_prev)
    for i in range(1, filt_len):
        C_prev = list_of_C_matrices[-1]
        cond_row = cond_row | (row_birth_times == i)
        cond_col = cond_col | (col_birth_times == i)
        cond_row_death = cond_row_death & ~(row_death_times == i)
        cond_col_death = cond_col_death & ~(col_death_times == i)
        cond_row = cond_row & cond_row_death
        cond_col = cond_col & cond_col_death
        cond_true_row = np.where(cond_row)[0]
        cond_true_col = np.where(cond_col)[0]
        C_curr = f0[cond_true_row, :][:, cond_true_col]
        A = sp.identity(C_prev.shape[1], format='lil')
        A[:, np.nonzero(col_death_times == i)[0]] = 0
        A = A.tocsc()
        A = sp.vstack((A, sp.csc_array((len(np.nonzero(col_birth_times == i)[0]), C_prev.shape[1]))))
        B = sp.identity(C_prev.shape[0], format='lil')
        B[:, np.nonzero(row_death_times == i)[0]] = 0
        B = B.tocsc()
        B = sp.vstack((B, sp.csc_array((len(np.nonzero(row_birth_times == i)[0]), C_prev.shape[0]))))
        list_of_C_matrices.append(C_curr)
        list_of_A_matrices.append(A)
        list_of_B_matrices.append(B)
        
    return list_of_A_matrices, list_of_B_matrices, list_of_C_matrices

class MatrixWithBirthDeathIdcs:
    def __init__(self, mat: sp.csc_matrix, row_births, row_deaths, col_births, col_deaths):
        self.mat = mat
        self.row_births = row_births
        self.col_births = col_births
        self.row_deaths = row_deaths
        self.col_deaths = col_deaths
        
def create_f0s(edge_list, filt_len = 100, dim_mat = 7):
    """Create f0s for each vertex-edge incidence."""
    f0_list_u = []
    col_births_u = []
    row_births_u = []
    f0_list_v = []
    col_births_v = []
    row_births_v = []
    for idx, (u, v) in enumerate(edge_list):
        f0_u = sp.random(dim_mat, dim_mat, density=0.4, format='csc')
        f0_v = sp.random(dim_mat, dim_mat, density=0.4, format='csc')
        row_births = compute_bars_birth_rows(dim_mat, filt_len)
        u_col_births = compute_bars_birth_cols(row_births, f0_u, filt_len)
        v_col_births = compute_bars_birth_cols(row_births, f0_v, filt_len)
        col_sorted_u_idcs = np.argsort(u_col_births)
        col_sorted_v_idcs = np.argsort(v_col_births)
        f0_u = f0_u[:, col_sorted_u_idcs]
        f0_v = f0_v[:, col_sorted_v_idcs]
        col_births_u.append(u_col_births[col_sorted_u_idcs])
        col_births_v.append(v_col_births[col_sorted_v_idcs])
        row_births_u.append(row_births)
        row_births_v.append(row_births)
        f0_list_u.append(f0_u)
        f0_list_v.append(f0_v)
    
    return f0_list_u, f0_list_v, col_births_u, row_births_u, col_births_v, row_births_v
    

if __name__ == '__main__':
    np.random.seed(42)

    num_nodes = 20
    max_dim_stalk = 7
    graph_edge_list = get_graph(num_nodes)
    filt_len = 500
    f0_list_u, f0_list_v, col_births_u, row_births_u, col_births_v, row_births_v = create_f0s(graph_edge_list, filt_len=filt_len, dim_mat=max_dim_stalk)
    
    
    obj_u = Parallel(n_jobs=16, verbose=4)(delayed(generate_filt_matrices)(f0_list_u[i], col_births_u[i], row_births_u[i], filt_len=filt_len) for i in range(len(f0_list_u)))
    obj_v = Parallel(n_jobs=16, verbose=4)(delayed(generate_filt_matrices)(f0_list_v[i], col_births_v[i], row_births_v[i], filt_len=filt_len) for i in range(len(f0_list_v)))
    
    # n = 0
    # for i in range(len(obj_u[0][0])):
    #     n += obj_u[0][0][i].shape[0]
    # print(f'Summation n_i is {n}')
    
    # start = time.time()
    obj2_u = Parallel(n_jobs=16, verbose=4)(delayed(compute_maps_betn_bars)(obj_u[j][0], obj_u[j][1], obj_u[j][2]) for j in range(len(obj_u)))
    obj2_v = Parallel(n_jobs=16, verbose=4)(delayed(compute_maps_betn_bars)(obj_v[j][0], obj_v[j][1], obj_v[j][2]) for j in range(len(obj_v)))

    big_f0_matrix = sp.lil_array((len(graph_edge_list)*max_dim_stalk, num_nodes*max_dim_stalk))
    big_f0 = MatrixWithBirthDeathIdcs(big_f0_matrix, {},
                                      {}, 
                                      {},
                                      {})
    
    big_f0_row_births = np.array([0 for i in range(len(graph_edge_list)*max_dim_stalk)])
    big_f0_row_deaths = np.array([-10 for i in range(len(graph_edge_list)*max_dim_stalk)])
    big_f0_col_births = np.array([0 for i in range(num_nodes*max_dim_stalk)])
    big_f0_col_deaths = np.array([-10 for i in range(num_nodes*max_dim_stalk)])
    for idx, (u,v) in enumerate(graph_edge_list):
        edge_row_idcs = np.arange(idx * max_dim_stalk, (idx+1) * max_dim_stalk)
        vertex_u_col_idcs = np.arange(u * max_dim_stalk, (u+1) * max_dim_stalk)
        vertex_v_col_idcs = np.arange(v*max_dim_stalk, (v+1)*max_dim_stalk)
        big_f0_matrix[edge_row_idcs[:,None], vertex_u_col_idcs] = obj2_u[idx].mat
        big_f0_matrix[edge_row_idcs[:,None], vertex_v_col_idcs] = obj2_v[idx].mat
        big_f0_row_births[edge_row_idcs] = list(obj2_u[idx].row_births.values())
        big_f0_row_deaths[edge_row_idcs] = list(obj2_u[idx].row_deaths.values())
        big_f0_col_births[vertex_u_col_idcs] = list(obj2_u[idx].col_births.values())
        big_f0_col_births[vertex_v_col_idcs] = list(obj2_v[idx].col_births.values())
        big_f0_col_deaths[vertex_u_col_idcs] = list(obj2_u[idx].col_deaths.values())
        big_f0_col_deaths[vertex_v_col_idcs] = list(obj2_v[idx].col_deaths.values())
    
    
    big_f0.mat = big_f0_matrix.tocsc()
    rows_sorted = np.argsort(big_f0_row_births)
    cols_sorted = np.argsort(big_f0_col_births)
    big_f0.mat = big_f0.mat[rows_sorted, :]
    big_f0.mat = big_f0.mat[:, cols_sorted]
    big_f0_row_births = big_f0_row_births[rows_sorted]
    big_f0_row_deaths = big_f0_row_deaths[rows_sorted]
    big_f0_col_births = big_f0_col_births[cols_sorted]
    big_f0_col_deaths = big_f0_col_deaths[cols_sorted]
    
    big_f0.col_births = dict(enumerate(list(big_f0_col_births)))
    big_f0.col_deaths = dict(enumerate(list(big_f0_col_deaths)))
    big_f0.row_births = dict(enumerate(list(big_f0_row_births)))
    big_f0.row_deaths = dict(enumerate(list(big_f0_row_deaths)))

    f0_reduced_d, _, pivot_rows_cols = col_redn(big_f0.mat)
    
    '''Since g0, r, are all zero, we can directly get the bars from f0_reduced_d and q'''
    
    bars = []
    pivot_rows = []
    for i in range(len(pivot_rows_cols)):
        if pivot_rows_cols[i] != -10:
            pivot_rows.append(pivot_rows_cols[i])
            if big_f0.row_births[pivot_rows_cols[i]] < big_f0.col_births[i]: # This is to remove unimportant bars such as [1,1). 
                bars.append((big_f0.row_births[pivot_rows_cols[i]], big_f0.col_births[i]))
    
    non_pivot_rows = list(set(list(range(big_f0.mat.shape[0]))) - set(pivot_rows))
    for i in non_pivot_rows:
        bars.append((big_f0.row_births[i], np.inf))
    # end = time.time()
    # print(f'Time taken is {end-start}')
    
    # print(bars)
    
    
