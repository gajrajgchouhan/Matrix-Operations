import matplotlib.pyplot as plt 
import numpy as np  
from mpl_toolkits.mplot3d import Axes3D


m = np.array([[1,2,3],[2,4,6],[3,6,5]])

ax = plt.figure().add_subplot(projection='3d')

origin = [0,0,0]
ax.quiver(origin, origin, origin, m[0,:], m[1,:], m[2,:], color=["red","green","blue"], length=0.1, normalize=False)

plt.show()

