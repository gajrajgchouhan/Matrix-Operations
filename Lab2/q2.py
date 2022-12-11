import numpy as np
from math import sqrt, sin, cos, pi
from q1 import l1_norm, linf_norm, l2_norm, spectral_norm, condition_no
from methods import GaussianElimination, LU, QR, Cholesky

n = 5

mat = []

for i in range(n):
    r = []
    for j in range(n):
        r.append(1/(i+j+1))
    mat.append(r)

mat = np.array(mat)

B = mat.sum(axis=0)

G = GaussianElimination(mat, B)
G.solve()

print(G.X)

LUMethod = LU(mat, B)
LUMethod.solve()

print(LUMethod.X)


QRMethod = QR(mat, B)
QRMethod.solve()

print(QRMethod.X)

CholeskyMethod = Cholesky(mat, B)
CholeskyMethod.solve()

print(CholeskyMethod.X)

print(f"condition_no = {condition_no(mat, l2_norm)}")