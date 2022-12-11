import os
import cv2
import numpy as np 
from matplotlib import pyplot as plt
from math import ceil

def reconstruct_PCA(A,k):
    A = A.astype("float") 
    mean = np.mean(A, axis=0)
    A -= mean
    cov = A@A.T / (A.shape[0] - 1) 
    eigval, eigvec = np.linalg.eigh(cov)
    order = eigval.argsort()[::-1]
    eigval = eigval[order]
    eigvec = eigvec[:,order]
    eigvec = eigvec[:,:k] 
    # A At v = l v
    # v vt A At = v(l v)t
    return (eigvec @ eigvec.T @ A) + mean

def SVD(A):
    A = A.astype("float")
    m = A.shape[0]
    n = A.shape[1]
    r = min(m,n)
    S = np.zeros((m,n))

    helper = A.T@A
    eigenvalues1, eigenvectors = np.linalg.eigh(helper)
    index = eigenvalues1.argsort()[::-1]
    eigenvalues1 = eigenvalues1[index]
    eigenvectors = eigenvectors[:, index]

    V = np.copy(eigenvectors)
    j = 0
    for i in eigenvalues1:
        if j == r:
            break
        else:
            S[j,j] = np.sqrt(i)
            j += 1

    helper = A@A.T
    eigenvalues2, eigenvectors = np.linalg.eigh(helper)
    index = eigenvalues2.argsort()[::-1]
    eigenvalues2 = eigenvalues2[index]
    eigenvectors = eigenvectors[:, index]

    U = np.copy(eigenvectors)

    S = U.T@A@V # sign of S might be differ so 

    # U = m,m
    # S = m,n
    # V = n,n
    # A = m,n

    # for taking top k!
    # U = m,k
    # S = k,k
    # V = k,n

    return U, S, V

def reconstruct_SVD(A,k):
    U,S,V = SVD(A)
    reconstructed = U[:,:k]@ S[:k,:k]@((V[:,:k]).T)
    return reconstructed

def reconstruct(file, K):
    img = cv2.imread(file,0).astype(float)/255
    nplots=len(K)
    #find number of columns, rows, and empty plots
    nc=int(nplots**0.5)
    nr=int(ceil(nplots/nc))
    empty=nr*nc-nplots
    #make the plot grid
    f,ax=plt.subplots(nr,nc)

    #force ax to have two axes so we can index it properly
    if nplots==1:
        ax=array([ax])
    if nc==1:
        ax=ax.reshape(nr,1)
    if nr==1:
        ax=ax.reshape(1,nc)

    #hide the unused subplots
    for i in range(empty): ax[-(1+i),-1].axis('off')

    #loop through subplots and make output
    for i in range(nplots):
        ic=i//nr #find which row we're on. If the definitions of ir and ic are switched, the indecies for empty (above) should be switched, too.
        ir=i%nr #find which column we're on
        axx=ax[ir,ic] #get a pointer to the subplot we're working with
        concat_img = []
        norm = []
        for func in (reconstruct_PCA, reconstruct_SVD):
            k=K[i]
            reconstructed_img = func(img,k)
            # reconstructed_img = 255 * (reconstructed_img - reconstructed_img.min()) / (reconstructed_img.max() - reconstructed_img.min())
            # reconstructed_img = reconstructed_img.astype("uint8")
            concat_img.append(reconstructed_img)
            norm.append(np.linalg.norm(reconstructed_img-img))
        concat_img = np.hstack(concat_img)
        axx.imshow(concat_img, cmap="gray")
        axx.set_title(f"K = {k} | PCA:{norm[0]:.5f} | SVD:{norm[1]:.5f}")
        axx.axis("off")
    plt.show()

if __name__ == '__main__':
    file = "The-Lena-face-Test-Image.ppm"
    reconstruct(file,[10,20,50,70,100,200,500])
