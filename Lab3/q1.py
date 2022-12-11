from methods import QR
import numpy as np 
import matplotlib.pyplot as plt

Y = np.array([0, 0.0040, 0.0160, 0.0360, 0.0640, 0.1])

X = np.array([10.0000, 10.2, 10.4, 10.6, 10.8, 11])

A = np.vstack((X**2, X, np.ones(X.shape))).T
C = np.copy(Y)


QRMethod = QR(A, C)
QRMethod.solve()

a, b, c = QRMethod.X

print("Q:", QRMethod.Q, "\n")
print("R:", QRMethod.R, "\n")
print("coeeficents", QRMethod.X)

print("error : ", Y - (A @ QRMethod.X))

x = np.linspace(9, 12)
plt.plot(x, a*x*x + b*x + c, "r")
plt.scatter(X, A @ QRMethod.X)
plt.show()
