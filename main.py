import numpy as np
import scipy.sparse as sp
from utils import *
from compute_hom_pres import *



if __name__ == '__main__':
    f0 = MatrixWithBirthDeathIdcs(sp.csc_matrix(np.array([[1,1,0],[0,1,1],[1,0,1]])), dict(enumerate([0,0,1])), dict(enumerate([2,1,1])), dict(enumerate([2,1,1])), dict(enumerate([6,5,7])))
    g0 = MatrixWithBirthDeathIdcs(sp.csc_matrix(np.array([[1,1,1]])), dict(enumerate([0])), dict(enumerate([3])), dict(enumerate([0,0,1])), dict(enumerate([5,5,6])))
    bars = get_bars_hom_cplx_pres(f0, g0)
    print(bars)