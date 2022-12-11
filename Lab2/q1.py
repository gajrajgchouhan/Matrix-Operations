import numpy as np
from math import sqrt, sin, cos, pi

def l1_norm(mat):
    r, c = mat.shape
    mx = -1
    for i in range(c):
        s = 0
        for j in range(r):
            s += abs(mat[j][i])
        mx = max(mx, s)
    return mx

def linf_norm(mat):
    return l1_norm(mat.T)

def l2_norm(mat):
    s = 0
    r, c = mat.shape
    for i in range(r):
        for j in range(c):
            s += abs(mat[i][j])**2
    return sqrt(s)

def spectral_norm(mat):
    mat = mat.T
    r, c = mat.shape
    for i in range(r):
        for j in range(c):
            com = complex(mat[i][j])
            mat[i][j] = complex(com.real,-(com.imag))
    v, _ = np.linalg.eig(mat)
    return max([abs(i) for i in v])

def condition_no(mat, norm):
    return norm(mat) * norm(np.linalg.inv(mat))

norms = [l1_norm, linf_norm, l2_norm, spectral_norm]

if __name__ == '__main__':
    theta = pi/6
    matA = np.array([
        [cos(theta), -sin(theta)],
        [sin(theta), cos(theta)]
    ]).astype(complex)

    matB = np.array([
        [1, 3, 4],
        [4,5,6],
        [-15,6,9]
    ]).astype(complex)

    matC = np.array([
        [2,1,0,0,0],
        [3,3,12,0,0],
        [0,4,-33,21,0],
        [0,0,12,0,23],
        [5,0,0,14,67]
    ]).astype(complex)
    
    print(f"condition_no for matA for l1_norm = {condition_no(matA, l1_norm)}")
    print(f"condition_no for matA for l2_norm = {condition_no(matA, l2_norm)}")
    print(f"condition_no for matA for linf_norm = {condition_no(matA, linf_norm)}")
    print(f"condition_no for matA for spectral_norm = {condition_no(matA, spectral_norm)}")
    print()
    print(f"condition_no for matB for l1_norm = {condition_no(matB, l1_norm)}")
    print(f"condition_no for matB for l2_norm = {condition_no(matB, l2_norm)}")
    print(f"condition_no for matB for linf_norm = {condition_no(matB, linf_norm)}")
    print(f"condition_no for matB for spectral_norm = {condition_no(matB, spectral_norm)}")
    print()
    print(f"condition_no for matC for l1_norm = {condition_no(matC, l1_norm)}")
    print(f"condition_no for matC for l2_norm = {condition_no(matC, l2_norm)}")
    print(f"condition_no for matC for linf_norm = {condition_no(matC, linf_norm)}")
    print(f"condition_no for matC for spectral_norm = {condition_no(matC, spectral_norm)}")
