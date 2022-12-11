import matplotlib.pyplot as plt 
import numpy as np 
from methods import QR

def powerMethod(A):
    A = A.astype("float"); thresold = 1e-3
    r = A.shape[0]
    guess = np.ones((r, 1), dtype="float")
    old_guess = np.copy(guess)
    while 1:
        guess = A @ guess
        guess /= guess.max()
        error = np.linalg.norm(guess - old_guess)
        if error < thresold:
            break
        else:
            old_guess = np.copy(guess)
    eigval = (A @ guess) / guess
    eigenvector, eigval = guess, np.mean(eigval)
    eigval = np.linalg.norm(eigenvector) * eigval
    eigenvector = eigenvector / np.linalg.norm(eigenvector)
    return eigenvector, eigval

def eigfromQR(A, max_iter=20):
    A = A.astype("float")
    eigenvectors = np.diagflat(np.ones((A.shape[0], 1)))
    temp = np.ones((A.shape[0], 1))
    for _ in range(max_iter): 
        qr = QR(A, temp)
        qr.solve()
        q, r = qr.Q, qr.R
        A = r @ q 
        eigenvectors = eigenvectors @ q
    return eigenvectors, np.diag(A)

def error_eigvals(e1,e2):
    return abs(e1-e2)

from math import isclose
from numpy.linalg import norm
def error_eigvecs(e1,e2):
    if not isclose(norm(e1),norm(e2),rel_tol=1e-4):
        raise Exception("Input must be unit vectors and equal to each other.")

    # Check if e1 = -e2 ?
    return min(norm(e1+e2), norm(e1-e2))

if __name__ == '__main__':
    A = np.array([[8, -6, 2], [-6, 7, -4], [2, -4, 3]], dtype="float")
    print("Q1----")
    print("Largest eigenval from A")
    eigvector, eigval = powerMethod(A)
    print("eigvector", eigvector.ravel())
    print("eigvalue", eigval)
    print()

    n = 10
    rnd = np.random.random((n, n))
    print("Largest eigenval from random matrix")
    eigvector, eigval = powerMethod(rnd)
    print("eigvector", eigvector.ravel())
    print("eigvalue", eigval)
    print()

    print("Q3----")
    A = np.array([[2, 1, 0], [1, 3, 1], [0, 1, 4]])
    eigvector, eigval = powerMethod(rnd)
    print("eigvector", eigvector.ravel())
    print("eigvalue", eigval)
    print()
    eigvec, eigval = eigfromQR(A)
    # Sorted eigenvals and eigenvectors to keep them equal
    order = eigval.argsort()[::-1]
    eigval = eigval[order]
    eigvec = eigvec[:, order]
    np_eigval, np_eigvec = np.linalg.eig(A)
    order = np_eigval.argsort()[::-1]
    np_eigval = np_eigval[order]
    np_eigvec = np_eigvec[:, order]

    print("eigenval and eigenvectors from A using QR")
    for i in range(eigval.size):
        print(f"eigvalue #{i} : ", eigval[i])
        print(f"eigvector #{i} : ", eigvec[:,i])
        print("error with numpy eigenval", error_eigvals(eigval[i], np_eigval[i]))
        print("error with numpy eigenvec", error_eigvecs(eigvec[:,i], np_eigvec[:,i]))
        print()
