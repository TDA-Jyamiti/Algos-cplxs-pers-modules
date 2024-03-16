import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
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

def compute_bars_birth_cols(row_birth_times, row_death_times, Mat, filt_len=100):
    col_birth_times = []
    col_wise_rows_non_zero = np.split(Mat.indices, Mat.indptr[1:-1])
    d = dict(enumerate(col_wise_rows_non_zero,0))
    for col in d:
        if len(d[col]>0):
            if np.max(row_birth_times[d[col]]) >= np.min(row_death_times[d[col]]):
                col_birth_times.append(np.max(row_birth_times[d[col]]))
            else:
                col_birth_times.append(np.random.randint(np.max(row_birth_times[d[col]]), np.min(row_death_times[d[col]]))) 
        else:
            col_birth_times.append(np.random.randint(0, filt_len-1))
    
    return np.array(col_birth_times)

def compute_bars_death_rows(num_rows, filt_len=100):
    row_death_times = np.random.randint(filt_len//2, filt_len-1, size=num_rows)
    
    return row_death_times

def compute_bars_death_cols(row_death_times, col_birth_times, Mat, filt_len=100):
    col_death_times = []
    col_wise_rows_non_zero = np.split(Mat.indices, Mat.indptr[1:-1])
    d = dict(enumerate(col_wise_rows_non_zero,0))
    for col in d:
        if len(d[col]>0):
           col_death_times.append(np.random.randint(np.max(row_death_times[d[col]]), filt_len+1))
        else:
            col_death_times.append(np.random.randint(col_birth_times[col]+1, filt_len+1))

    return np.array(col_death_times)

def generate_filt_matrices(f0, col_birth_times, row_birth_times, row_death_times, col_death_times, filt_len=100):
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
        C_prev_col_map = dict(enumerate(cond_true_col))
        C_prev_row_map = dict(enumerate(cond_true_row))
        cond_row = (cond_row | (row_birth_times == i)) & (row_death_times > i)
        cond_col = (cond_col | (col_birth_times == i)) & (col_death_times > i)
        cond_true_row = np.where(cond_row)[0]
        cond_true_col = np.where(cond_col)[0]
        C_curr = f0[cond_true_row, :][:, cond_true_col]
        C_curr_col_map = dict(enumerate(cond_true_col))
        C_curr_row_map = dict(enumerate(cond_true_row))
        if C_curr.shape[1] == 0 and C_prev.shape[1] != 0:
            A = sp.lil_array((C_prev.shape[1], C_prev.shape[1]))
        else:
            A = sp.lil_array((C_curr.shape[1], C_prev.shape[1]))
        for prev_col in C_prev_col_map:
            for curr_col in C_curr_col_map:
                if C_prev_col_map[prev_col] == C_curr_col_map[curr_col]:
                    A[curr_col, prev_col] = 1
        
        if C_curr.shape[0] == 0 and C_prev.shape[0] != 0:
            B = sp.lil_array((C_prev.shape[0], C_prev.shape[0]))
        else:    
            B = sp.lil_array((C_curr.shape[0], C_prev.shape[0]))
        for prev_row in C_prev_row_map:
            for curr_row in C_curr_row_map:
                if C_prev_row_map[prev_row] == C_curr_row_map[curr_row]:
                    B[curr_row, prev_row] = 1   
        
        list_of_C_matrices.append(C_curr.tocsc())
        list_of_A_matrices.append(A.tocsc())
        list_of_B_matrices.append(B.tocsc())
        
    return list_of_A_matrices, list_of_B_matrices, list_of_C_matrices


class MatrixWithBirthDeathIdcs:
    def __init__(self, mat: sp.csc_matrix, row_births, row_deaths, col_births, col_deaths):
        self.mat = mat
        self.row_births = row_births
        self.col_births = col_births
        self.row_deaths = row_deaths
        self.col_deaths = col_deaths

def compute_bars_births_given_cols(col_births, f0, filt_len):
    Mat = f0.tocsr()
    row_births = []
    row_wise_non_zero = np.split(Mat.indices, Mat.indptr[1:-1])
    d = dict(enumerate(row_wise_non_zero,0))
    for row in d:
        if len(d[row]>0):
            if np.min(col_births[d[row]]) > 0:
                row_births.append(np.random.randint(0, np.min(col_births[d[row]])))
            else:
                row_births.append(0)
        else:
            row_births.append(np.random.randint(0, filt_len//2))
    return np.array(row_births)

def compute_bars_death_rows_given_cols(row_births, col_births, col_deaths, f0, filt_len):
    Mat = f0.tocsr()
    f0=f0.tolil()
    row_death_times = []
    row_wise_non_zero = np.split(Mat.indices, Mat.indptr[1:-1])
    d = dict(enumerate(row_wise_non_zero,0))
    for row in d:
        if len(d[row]>0):
            if max(row_births[row], np.max(col_births[d[row]])) +1 >= np.min(col_deaths[d[row]]):
                # f0[row, np.argmax(col_births[d[row]])] = 0
                f0[row, :] = 0
                row_death_times.append(max(row_births[row], np.max(col_births[d[row]]))+2)
            else:
                row_death_times.append(np.random.randint(max(row_births[row], np.max(col_births[d[row]]))+1, np.min(col_deaths[d[row]])))
        else:
            row_death_times.append(np.random.randint(row_births[row], filt_len))
    f0 = f0.tocsc()
    return np.array(row_death_times), f0

def create_f0s(edge_list, filt_len = 10, dim_mat = 7):
    f0_list_u = []
    vertex_birth_times, vertex_death_times = {}, {}
    col_births_u = []
    col_deaths_u = []
    row_births_u = []
    row_deaths_u = []
    f0_list_v = []
    col_births_v = []
    col_deaths_v = []
    row_births_v = []
    row_deaths_v = []
    for idx, (u, v) in enumerate(edge_list):
        f0_u = sp.random(dim_mat, dim_mat, density=0.4, format='csc')
        f0_v = sp.random(dim_mat, dim_mat, density=0.4, format='csc')
        if u in vertex_birth_times:
            u_col_births = vertex_birth_times[u]
            u_col_deaths = vertex_death_times[u]
            row_births = compute_bars_births_given_cols(u_col_births, f0_u, filt_len)
            row_deaths, f0_u = compute_bars_death_rows_given_cols(row_births, u_col_births, u_col_deaths, f0_u, filt_len)
            v_col_births = compute_bars_birth_cols(row_births, row_deaths, f0_v, filt_len)
            v_col_deaths = compute_bars_death_cols(row_deaths, v_col_births, f0_v, filt_len)
        elif v in vertex_birth_times:
            v_col_births = vertex_birth_times[v]
            v_col_deaths = vertex_death_times[v]
            row_births = compute_bars_births_given_cols(v_col_births, f0_v, filt_len)
            row_deaths, f0_v = compute_bars_death_rows_given_cols(row_births, v_col_births, v_col_deaths, f0_v, filt_len)
            u_col_births = compute_bars_birth_cols(row_births, row_deaths, f0_u, filt_len)
            u_col_deaths = compute_bars_death_cols(row_deaths, u_col_births, f0_u, filt_len)
        else:
            row_births = compute_bars_birth_rows(dim_mat, filt_len)
            row_deaths = compute_bars_death_rows(dim_mat, filt_len)
            u_col_births = compute_bars_birth_cols(row_births, row_deaths, f0_u, filt_len)
            u_col_deaths = compute_bars_death_cols(row_deaths, u_col_births, f0_u, filt_len)
            v_col_births = compute_bars_birth_cols(row_births, row_deaths, f0_v, filt_len)
            v_col_deaths = compute_bars_death_cols(row_deaths, v_col_births, f0_v, filt_len)
            vertex_birth_times[u] = u_col_births
            vertex_death_times[u] = u_col_deaths
            vertex_birth_times[v] = v_col_births
            vertex_death_times[v] = v_col_deaths
        col_sorted_u_idcs = np.argsort(u_col_births)
        col_sorted_v_idcs = np.argsort(v_col_births)
        f0_u = f0_u[:, col_sorted_u_idcs]
        f0_v = f0_v[:, col_sorted_v_idcs]
        col_births_u.append(u_col_births[col_sorted_u_idcs])
        col_births_v.append(v_col_births[col_sorted_v_idcs])
        col_deaths_u.append(u_col_deaths[col_sorted_u_idcs])
        col_deaths_v.append(v_col_deaths[col_sorted_v_idcs])
        row_births_u.append(row_births)
        row_births_v.append(row_births)
        row_deaths[row_deaths == row_births] +=1
        row_deaths_u.append(row_deaths)
        row_deaths_v.append(row_deaths)
        f0_list_u.append(f0_u)
        f0_list_v.append(f0_v)
    
    return f0_list_u, f0_list_v, col_births_u, row_births_u, col_births_v, row_births_v, col_deaths_u, col_deaths_v, row_deaths_u, row_deaths_v


if __name__ == '__main__':
    # np.random.seed(42)

    num_nodes = 20
    max_dim_stalk = 7
    graph_edge_list = get_graph(num_nodes)
    filt_len = 500
    f0_list_u, f0_list_v, col_births_u, row_births_u, col_births_v, row_births_v, col_deaths_u, col_deaths_v, row_deaths_u, row_deaths_v = create_f0s(graph_edge_list, filt_len=filt_len, dim_mat=max_dim_stalk)
    
    print('Generating filtration matrices...')
    
    obj_u = Parallel(n_jobs=16, verbose=4)(delayed(generate_filt_matrices)(f0_list_u[i], col_births_u[i], row_births_u[i], row_deaths_u[i], col_deaths_u[i], filt_len=filt_len) for i in range(len(f0_list_u)))
    obj_v = Parallel(n_jobs=16, verbose=4)(delayed(generate_filt_matrices)(f0_list_v[i], col_births_v[i], row_births_v[i], row_deaths_v[i], col_deaths_v[i], filt_len=filt_len) for i in range(len(f0_list_v)))
    
    n = 0
    for i in range(len(obj_u[0][0])):
        n += obj_u[0][0][i].shape[0]
    print(f'Input size (summation n_i) is {n*len(graph_edge_list)}')

    
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
        vertex_v_col_idcs = np.arange(v * max_dim_stalk, (v+1) * max_dim_stalk)
        
        big_f0_matrix[edge_row_idcs[:,None], vertex_u_col_idcs] = obj2_u[idx].mat
        big_f0_matrix[edge_row_idcs[:,None], vertex_v_col_idcs] = obj2_v[idx].mat
        big_f0_row_births[edge_row_idcs] = list(obj2_u[idx].row_births.values())
        big_f0_row_deaths[edge_row_idcs] = list(obj2_u[idx].row_deaths.values())
        big_f0_col_births[vertex_u_col_idcs] = list(obj2_u[idx].col_births.values())
        big_f0_col_births[vertex_v_col_idcs] = list(obj2_v[idx].col_births.values())
        big_f0_col_deaths[vertex_u_col_idcs] = list(obj2_u[idx].col_deaths.values())
        big_f0_col_deaths[vertex_v_col_idcs] = list(obj2_v[idx].col_deaths.values())
        
    # concatenating q to f0
    big_f0_matrix = sp.hstack([big_f0_matrix, sp.identity(big_f0_matrix.shape[0], format='lil')])
    big_f0_col_births = np.array(big_f0_col_births.tolist() + big_f0_row_deaths.tolist())
    big_f0.mat = big_f0_matrix.tocsc()
    cols_sorted = np.argsort(big_f0_col_births)
    big_f0.mat = big_f0.mat[:, cols_sorted]
    big_f0_col_births = big_f0_col_births[cols_sorted]
    
    big_f0.col_births = dict(enumerate(list(big_f0_col_births),0))
    big_f0.col_deaths = dict(enumerate(list(big_f0_col_deaths),0))
    big_f0.row_births = dict(enumerate(list(big_f0_row_births),0))
    big_f0.row_deaths = dict(enumerate(list(big_f0_row_deaths),0))
    
    f0_reduced_d, _, pivot_rows_cols = col_redn(big_f0.mat)
    
    bars = read_off_bars(big_f0.col_births, big_f0.row_births, pivot_rows_cols)
    
    print(bars)
    
    
