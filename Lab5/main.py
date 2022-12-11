import numpy as np
from methods import QR
import warnings

def error(true,pred):
    return np.linalg.norm((pred-true)/true)

def eigvaluesfromQR(A, max_iter=20):
    A = A.astype("float")
    temp = np.ones((A.shape[0], 1))
    for _ in range(max_iter): 
        qr = QR(A, temp)
        qr.solve()
        q, r = qr.Q, qr.R
        A = r @ q 
    return np.diag(A)

def householderMethod(A):
    A = A.astype("float")
    h = A.shape[0]
    col = 0
    row = 1
    for _ in range(h - 2):
        x = A[:, col].reshape(h, 1)
        k = np.linalg.norm(x[row:])
        y = np.zeros((h, 1))
        y[:row] = x[:row]
        y[row] = k
        w = (x - y) / np.linalg.norm(x - y)
        p = np.identity(h) - 2 * (w @ w.T)
        A = p @ A @ p
        col += 1
        row += 1
    return A

def sturmSequence(k,A,lamb):
    if k==-1: return 1 
    elif k==0: return A[0,0]-lamb
    else:
        return ((A[k,k]-lamb)*sturmSequence(k-1,A,lamb)) - \
               ((A[k-1,k]**2)*sturmSequence(k-2,A,lamb))

def findIntervals(A,start=0,maxiter=20,stepsize=1):
    A = A.astype("float")
    n = A.shape[0]
    lamb = start
    V = []
    it = 0
    eigvalsFoundYet=0
    while it < maxiter and eigvalsFoundYet < n:
        noofChanges=0
        signs = [sturmSequence(k,A,lamb) for k in range(-1,n)]
        signs = np.sign(signs).astype(int)
        signs[signs == 0] = 1
        for i in range(0,len(signs)-1):
            noofChanges += int(signs[i] != signs[i+1])
        if it==0:
            V.append((lamb, noofChanges))
        elif noofChanges != V[-1][1]:
            eigvalsFoundYet += abs(noofChanges - V[-1][1])
            V.append((lamb, noofChanges))
        lamb += stepsize
        it += 1
    if eigvalsFoundYet < n:
        warnings.warn('\033[93m'+"\n".join((f"Current start={start}",
                              "Couldnt find all eigenvalues, increase the start of range using start=",
                              "or increase the maximum iterations"))+'\033[0m')
    return V

def bisectfromInterval(intervals,A):
    eigenvals = []
    for i in range(len(intervals) - 1):
        curr = intervals[i][0]
        nxt = intervals[i+1][0]
        eigenvals.append(bisect(curr,nxt,A))
    if len(eigenvals) < A.shape[0]:
        warnings.warn('\033[93m'+"\n".join(("Couldnt find all eigenvalues",
                                "Make sure there isn't more than one root in a interval "
                                "Maybe decrease the step size of the intervals in 'findIntervals'"
            ))+'\033[0m')
    return eigenvals

def charactPoly(lamb,A):
    # np.eye is basically identity matrix
    return np.linalg.det(A - lamb*np.eye(A.shape[0],dtype="float"))

def bisect(start,end,A,thres=1e-4):
    while start <= end:
        mid = (start+end)/2
        s = charactPoly(start,A)
        e = charactPoly(end,A)
        m = charactPoly(mid,A)
        # print(start,end,mid,s,e,m)
        if m == 0 or abs(m) < thres or abs(start-mid) < thres:
            return mid
        elif np.sign(m) == np.sign(s):
            start = mid
        else:
            end = mid

if __name__ == "__main__":
    print("-------------------Part 1-------------------")
    A = np.array([[9,13,3,6],[13,11,7,6,],[3,7,4,7,],[6,6,7,10]])
    print("\nEigenvalues from numpy")
    np_eigenvalues = sorted(np.linalg.eig(A)[0])
    print(np_eigenvalues)
    print("\nEigenvalues from our QR method")
    qr_eigenvalues = np.array(sorted(eigvaluesfromQR(A)))
    print(qr_eigenvalues)
    trid = householderMethod(A)
    print("\nTridiagonal matrix by householder method")
    print(trid)
    intervals = findIntervals(trid,start=-10,maxiter=50)
    print("\nIntervals of eigenvalues in form of (lambda, V(lambda))")
    print(intervals)
    print("\nBisected interval to find the eigenvalues")
    our_eigenvalues = np.array(sorted(bisectfromInterval(intervals,trid)))
    print(our_eigenvalues)
    print("\n"+'\033[93m'+"If no of eigenvalues in interval is >1, then decrease the step size.")
    print("As bisection only finds one root at a time."+'\033[0m')
    print("error b/w numpy and our method", error(our_eigenvalues,np_eigenvalues))
    print("error b/w qr and our method", error(our_eigenvalues,qr_eigenvalues))
    print()

    print("-------------------Part 2-------------------")
    n = int(input("enter width of n x n random symmetric matrix (default 5)") or 5)
    print("Random symmetric matrix of n = ", n)
    A = np.random.random((n,n))
    A = (A+A.T)/2 # symmetric
    print("\nEigenvalues from numpy")
    np_eigenvalues = sorted(np.linalg.eig(A)[0])
    print(np_eigenvalues)
    print("\nEigenvalues from our QR method")
    qr_eigenvalues = np.array(sorted(eigvaluesfromQR(A,max_iter=100)))
    print(qr_eigenvalues)
    trid = householderMethod(A)
    print("\nTridiagonal matrix by householder method")
    print(trid)
    intervals = findIntervals(trid,start=-10,maxiter=5000,stepsize=0.1)
    print("\nIntervals of eigenvalues in form of (lambda, V(lambda))")
    print(intervals)
    print("\nBisected interval to find the eigenvalues")
    our_eigenvalues = np.array(sorted(bisectfromInterval(intervals,trid)))
    print(our_eigenvalues)
    print("\n"+'\033[93m'+"If no of eigenvalues in interval is >1, then decrease the step size.")
    print("As bisection only finds one root at a time."+'\033[0m')
    print("error b/w numpy and our method", error(our_eigenvalues,np_eigenvalues))
    print("error b/w qr and our method", error(our_eigenvalues,qr_eigenvalues))
    print()