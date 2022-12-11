from methods import Jacobi, GSiedel, SOR, make_diagonally_dominant
import numpy as np 
import matplotlib.pyplot as plt

A = np.array([[2.412, 9.879, 1.564], [1.876, 2.985, -11.62], [12.214, 2.367, 3.672]])
B = np.array([4.89, -0.972, 7.814])

jacobi = Jacobi(A, B)
jacobi.solve()

print("Jacobi : iterations=>", jacobi.iterations, " solution=>", jacobi.X)

gs = GSiedel(A, B)
gs.solve()

print("Gauss Siedal : iterations=>", gs.iterations, " solution=>", gs.X)

sor = SOR(A, B)
sor.solve()
print("SOR : iterations=>", sor.iterations, " solution=>", sor.X)

fig, ax = plt.subplots(ncols=3, sharex=True)

ax[0].plot(jacobi.errors)
ax[1].plot(gs.errors)
ax[2].plot(sor.errors)

ax[0].set_title("jacobi")
ax[1].set_title("gs")
ax[2].set_title("sor")

plt.show()
