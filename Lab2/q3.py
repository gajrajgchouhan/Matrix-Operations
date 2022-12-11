import numpy as np
from math import sqrt, sin, cos, pi
from q1 import l1_norm, linf_norm, l2_norm, spectral_norm, condition_no
from methods import GaussianElimination, LU, QR, Cholesky

A = np.zeros((60,60))

n = 60

for row in range(n):
    for col in range(n):
        if col == n - 1:
            A[row,col] = 1
        if row == col:
            A[row,col] = 1
        if row > col:
            A[row,col] = -1

print(f"condition_no = {condition_no(A, l2_norm)}")

x = np.random.random((60, 1))

B = A @ x

x0 = GaussianElimination(A, B)
x0.solve()

def rel_diff(x, y):
    return l2_norm(x - y) / l2_norm(x)

print(f"1. GaussianElimination {rel_diff(x, x0.X)}")

x1 = LU(A, B)
x1.solve()

print(f"2. LU {rel_diff(x, x1.X)}")

x2 = QR(A, B)
x2.solve()

print(f"3. QR {rel_diff(x, x2.X)}")
