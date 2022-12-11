import numpy as np
from math import pow, sqrt

class GaussianElimination(object):
    """docstring for GaussianElimination"""

    def __init__(self, A, B):
        self.A = np.copy(np.array(A, dtype="float"))
        self.B = np.copy(np.array(B, dtype="float"))
        self.n = A.shape[0]
        self.X = np.zeros_like(B, dtype="float")

    def solve(self):
        for col in range(self.n):

            # pivoting
            mx = self.A[col][col]
            mx_row = col
            for row in range(col + 1, self.n):
                if self.A[row][col] > mx:
                    mx = self.A[row][col]
                    mx_row = row

            tmp1 = self.B[col]
            tmp2 = self.B[mx_row]
            self.B[mx_row] = tmp1
            self.B[col] = tmp2

            for col2 in range(self.n):
                tmp1 = self.A[col][col2]
                tmp2 = self.A[mx_row][col2]

                self.A[col][col2] = tmp2
                self.A[mx_row][col2] = tmp1
            # end pivoting

            for row in range(col + 1, self.n):
                m = self.A[row][col] / self.A[col][col]
                for i in range(col, self.n):
                    self.A[row][i] -= self.A[col][i] * m
                self.B[row] -= self.B[col] * m

        for i in range(self.n - 1, -1, -1):
            if i == self.n - 1:
                self.X[i] = self.B[i] / self.A[i][i]
            else:
                s = 0
                for j in range(i + 1, self.n):
                    s += self.A[i][j] * self.X[j]
                self.X[i] = (self.B[i] - s) / self.A[i][i]


class LU(object):
    """
    L is lower triangular matrix
    U is upper triangulat matrix"""

    def __init__(self, A, B):
        self.A = np.copy(np.array(A, dtype="float"))
        self.B = np.copy(np.array(B, dtype="float"))
        self.n = self.A.shape[0]
        self.L = np.zeros_like(A, dtype="float")
        self.U = np.zeros_like(A, dtype="float")
        self.X = np.zeros_like(B, dtype="float")

    def solve(self):
        # A = LU
        # LUX = B
        # UX = y => Ly = B
        self._calcLU()
        G = GaussianElimination(self.L, self.B)
        G.solve()
        Y = np.copy(G.X)
        G = GaussianElimination(self.U, Y)
        G.solve()
        self.X = np.copy(G.X)

    def _calcLU(self):
        for i in range(self.n):
            for j in range(i):
                total = 0
                for k in range(j):
                    total += self.L[i][k] * self.U[k][j]
                self.L[i][j] = (self.A[i][j] - total) / self.U[j][j]
            self.L[i][i] = 1
            for j in range(i, self.n):
                total = 0
                for k in range(i):
                    total += self.L[i][k] * self.U[k][j]
                self.U[i][j] = self.A[i][j] - total

class QR(object):
    """
    Q is orthogonal
    R is upper triangular matrix
    can be used to find eigenvalues"""

    def __init__(self, A, B):
        self.A = np.copy(np.array(A, dtype="float"))
        self.B = np.copy(np.array(B, dtype="float"))
        self.n, self.k = A.shape
        self.Q = np.zeros_like(A, dtype="float")
        self.R = np.zeros((self.k, self.k), dtype="float")
        self.X = np.zeros_like(B, dtype="float")

    def solve(self):
        # AX = B
        # RX = Qt*B
        self._calcQ()
        C = self.Q.T @ self.B
        GE = GaussianElimination(self.R, C)
        GE.solve()
        self.X = GE.X

    def _calcQ(self):
        for i in range(self.k):
            s = np.zeros((self.n,), dtype="float")
            for j in range(i):
                self.R[j, i] = self.Q[:, j].T @ self.A[:, i]
                # print(f"R{j+1}{i+1}=>{self.R[j,i]}; A{i+1}=>{list(self.A[:,i].flatten())}; Q{j+1}=>{list(self.Q[:, j].flatten())}; {i=}; toadd={self.R[j, i] * self.Q[:, j]}")
                s += (self.R[j, i] * self.Q[:, j])
            
            norm = np.linalg.norm(self.A[:, i] - s)
            self.Q[:, i] = (self.A[:, i] - s) / norm
            # print(f"R{i+1}{i+1}=>{np.linalg.norm(self.A[:, i] - s)}; {s=}")
            self.R[i, i] = norm

class Cholesky(object):
    """
    only if A is symmetric
    A should be POSITIVE DEFINITE
    A is L.Lt
    """

    def __init__(self, A, B):
        self.A = np.copy(np.array(A, dtype="float"))  
        self.B = np.copy(np.array(B, dtype="float"))
        self.X = np.zeros_like(B, dtype="float")
        self.n = A.shape[0]
        self.L = np.zeros_like(A, dtype="float")

    def solve(self):
        # A = L Lt
        # L Lt x = B
        # Ly = B => Lt x = y
        self._calcL()
        Y = np.zeros_like(self.X)
        for i in range(self.n):
            s = 0 
            for j in range(i):
                s += self.L[i,j] * Y[j]
            Y[i] = (self.B[i] - s) / self.L[i,i]
        # Lt
        for i in range(self.n - 1, -1, -1):
            s = 0 
            for j in range(i+1, self.n):
                s += self.L[j,i] * self.X[j]
            self.X[i] = (Y[i] - s) / self.L[i,i]

    def _calcL(self):
        for i in range(self.n):
            # print(f"{i=} {self.A[i,i]=}")
            s = self.A[i, i]
            for j in range(i - 1, -1, -1):
                # print(f"{i=} {j=} {self.L[i,j]=}")
                s -= pow(self.L[i, j], 2)
            self.L[i, i] = sqrt(s)
            # print(f"{self.L[i,i]=}")
            for j in range(i + 1, self.n):
                s = self.A[j,i]
                for k in range(j - 1):
                    s -= (self.L[i,k] * self.L[j,k])
                s /= self.L[i,i]
                self.L[j,i] = s 

def make_diagonally_dominant(A, B):
    r, c = A.shape

    flag = False
    
    toswap = []
    options = []

    for i in range(r):
        s = 0
        for j in range(c):
            s += abs(A[i][j])
        if 2*abs(A[i][i]) >= s:
            if 2*abs(A[i][i]) > s:
                flag = True
        
        toswap.append(i)
        o = []
        for k in range(c):
            if 2*abs(A[i][k]) >= s:
                if 2*abs(A[i][k]) > s:
                    o.append((k, 1))
                else: o.append((k, 0))
        options.append(o)

    # print(toswap)
    # print(options)

    sol = np.zeros_like(A, dtype="float")
    solB = np.zeros_like(B, dtype="float")
    flag = False
    for index, o in zip(toswap, options):
        for row, f in o:
            flag = flag or f   
            sol[row] = A[index]
            solB[row] = B[index]

    return sol, solB
        
class Jacobi(object):
    def __init__(self, A, B):
        self.A = np.copy(np.array(A, dtype="float"))
        self.B = np.copy(np.array(B, dtype="float"))
        self.X = np.zeros_like(B, dtype="float")
        self.r, self.c = A.shape
        self.iterations = 0
        self.thresold = 1e-5
        self.errors = []

    def solve(self):
        self.A, self.B = make_diagonally_dominant(self.A, self.B)
        self.iterations = 0 
        self.errors = []

        while 1:
            new = np.zeros_like(self.X)
            for i in range(self.r):
                s = 0
                for j in range(self.c):
                    if i != j:
                        s += self.X[j] * self.A[i][j]
                new[i] = (self.B[i] - s) / self.A[i][i]
            self.iterations += 1
            self.errors.append(np.linalg.norm(new - self.X))
            if np.all(abs(new - self.X) < self.thresold):
                self.X = np.copy(new)
                break
            self.X = np.copy(new)


class GSiedel(object):
    def __init__(self, A, B):
        self.A = np.copy(np.array(A, dtype="float"))
        self.B = np.copy(np.array(B, dtype="float"))
        self.X = np.zeros_like(B, dtype="float")
        self.r, self.c = A.shape
        self.iterations = 0
        self.thresold = 1e-5
        self.errors = []

    def solve(self):
        self.A, self.B = make_diagonally_dominant(self.A, self.B)
        self.iterations = 0 
        self.errors = []

        while 1:
            new = np.copy(self.X)
            for i in range(self.r):
                s = 0
                for j in range(self.c):
                    if i != j:
                        s += new[j] * self.A[i][j]
                new[i] = (self.B[i] - s) / self.A[i][i]
            self.iterations += 1
            self.errors.append(np.linalg.norm(new - self.X))
            if np.all(abs(new - self.X) < self.thresold):
                self.X = np.copy(new)
                break
            self.X = np.copy(new)

class SOR(object):
    def __init__(self, A, B):
        self.A = np.copy(np.array(A, dtype="float"))
        self.B = np.copy(np.array(B, dtype="float"))
        self.X = np.zeros_like(B, dtype="float")
        self.r, self.c = A.shape
        self.iterations = 0
        self.thresold = 1e-5
        self.omega = 1.1
        self.errors = []

    def solve(self):
        self.A, self.B = make_diagonally_dominant(self.A, self.B)
        self.iterations = 0 
        self.errors = []

        while 1:
            new = np.copy(self.X)
            for i in range(self.r):
                s = 0
                for j in range(self.c):
                    if i != j:
                        s += new[j] * self.A[i][j]
                    else:
                        s += self.X[j] * self.A[i][j]
                new[i] = self.X[i] + (self.omega * ((self.B[i] - s) / self.A[i][i]))
            self.iterations += 1
            self.errors.append(np.linalg.norm(new - self.X))
            if np.all(abs(new - self.X) < self.thresold):
                self.X = np.copy(new)
                break
            self.X = np.copy(new)

class CG(object):
    def __init__(self, A, B):
        self.A = np.copy(np.array(A, dtype="float"))
        self.B = np.copy(np.array(B, dtype="float"))
        self.X = np.zeros_like(B, dtype="float")
        self.n = A.shape[0]
        self.iterations = 0
        self.thresold = 1e-5
        self.errors = []
        self.verbose = False
        self.max_iter = None

    def solve(self):
        self.iterations = 0 
        self.errors = []
        self.X = np.zeros_like(self.B, dtype="float")
        r = self.B - (self.A @ self.X)
        p = np.copy(r)

        while 1:
            if self.verbose:
                print(f"{self.iterations=}")
                print(f"{r=}")
                print(f"{p=}")
                print(f"{self.X=}")
                print()
            old_r = np.copy(r)
            a = ( old_r.T @ old_r ) / ( p.T @ self.A @ p )
            self.X += a * p 
            r -= a * (self.A @ p)

            self.iterations += 1
            self.errors.append(r)
            if np.all(r < self.thresold):
                if self.max_iter is None:
                    break
                else:
                    if self.iterations >= self.max_iter:
                        break

            b = (r.T @ r )/ (old_r.T @ old_r)
            p = r + (b * p)

if __name__ == '__main__':
    x = np.array([[2,1], [1,2]])
    y = np.array([4,5])
    G = CG(x, y)
    G.verbose = True
    G.solve()
