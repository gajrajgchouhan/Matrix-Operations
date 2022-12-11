import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, pi

topx = np.array([1, 0.98, 0.8, 0.83, 0.8, 0.77, 0.5, 0.25, 0, -0.25, -0.5, -0.8, -0.8, -0.98, -1])
topy = np.array([0, 0.15, 0.7, 1.15, 1.03, 1.15, 0.85, 0.95, 1, 0.95, 0.85, 1.2, 0.7, 0.15, 0])
botx = -np.cos(np.pi * np.arange(1, 10) / 10)
boty = -np.sin(np.pi * np.arange(1, 10) / 10)
wiskx = np.array([0.2, 1.3, 0.2, 1.4, 0.2, 1.4, 0.2, 1.3, 0.2, 0.17, 0.13, 0.08, 0.03, 0])
wisky = np.array([0, 0.3, 0, 0.1, 0, -0.1, 0, -0.3, 0, 0.1, -0.1, 0.1, -0.1, 0]) - 0.2
xeye = np.array([0, 0.2, 0.3, 0.4, 0.43, 0.45, 0.43, 0.4, 0.37, 0.35, 0.37, 0.4, 0.5, 0.6, 0.5, 0.4, 0.3, 0.2, 0])
yeye = (
    np.array([0.5, 0.5, 0.43, 0.4, 0.42, 0.5, 0.58, 0.6, 0.58, 0.5, 0.42, 0.4, 0.43, 0.5, 0.57, 0.6, 0.57, 0.5, 0.5])
    - 0.2
)
x = np.concatenate((topx, botx, wiskx, xeye, -xeye, -wiskx[::-1]))
y = np.concatenate((topy, boty, wisky, yeye, yeye, wisky[::-1]))
cat = np.array([x, y]).T


def plotcat(a, title=""):
    """
    Plotting the cat.
    """
    plt.plot(a[:, 0], a[:, 1], "b")
    plt.title(title)
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.show()


plotcat(cat, "cat")

# Q1
"""
x,y => x,-y
"""
q1 = np.copy(cat)
m = np.array([[1, 0], [0, -1]])
q1 = q1 @ m

plotcat(q1, "q1 x,y => x,-y")

# Q2
"""
x,y => -x,y
"""
q2 = np.copy(cat)
m = np.array([[-1, 0], [0, 1]])
q2 = q2 @ m

plotcat(q2, "q2 x,y => -x,y")

# Q3
"""
x,y => y,x
"""
q3 = np.copy(cat)
m = np.array([[0, 1], [1, 0]])
q3 = q3 @ m

plotcat(q3, "q3 x,y => y,x")

# Q4
"""
x,y => -y,-x
"""
q4 = np.copy(cat)
m = np.array([[0, -1], [-1, 0]])
q4 = q4 @ m

plotcat(q4, "q4 x,y => -y,-x")

# Q5
"""
x,y => -x,-y
"""
q5 = np.copy(cat)
m = np.array([[-1, 0], [0, -1]])
q5 = q5 @ m

plotcat(q5, "q5 x,y => -x,-y")

# Q6
"""
x,y => x/3,y
"""
q6 = np.copy(cat)
m = np.array([[1 / 3, 0], [0, 1]])
q6 = q6 @ m

plotcat(q6, "q6 x,y => x/3,y")

# q7
"""
x,y => x,y/5
"""
q7 = np.copy(cat)
m = np.array([[1, 0], [0, 1 / 5]])
q7 = q7 @ m

plotcat(q7, "q7 x,y => x,y/5")

# q8
"""
x,y => 3x,y
"""
q8 = np.copy(cat)
m = np.array([[3, 0], [0, 1]])
q8 = q8 @ m

plotcat(q8, "q8 x,y => 3x,y")

# q9
"""
x,y => 4x,y
"""
q9 = np.copy(cat)
m = np.array([[4, 0], [0, 1]])
q9 = q9 @ m

plotcat(q9, "q9 x,y => 4x,y")

# q10a
"""
x,y => x-1.5y,y
"""
q10a = np.copy(cat)
m = np.array([[1, -1.5], [0, 1]])
q10a = q10a @ m

plotcat(q10a, "q10a")

# q10b
"""
x,y => x+1.5y,y
"""
q10b = np.copy(cat)
m = np.array([[1, 1.5], [0, 1]])
q10b = q10b @ m

plotcat(q10b, "q10b")

# q11a
"""
x,y => x,y-1.5x
"""
q11a = np.copy(cat)
m = np.array([[1, 0], [-1.5, 1]])
q11a = q11a @ m

plotcat(q11a, "q11a")

# q11b
"""
x,y => x,y+1.5x
"""
q11b = np.copy(cat)
m = np.array([[1, 0], [1.5, 1]])
q11b = q11b @ m

plotcat(q11b, "q11b")

# q12
"""
x,y => x,0
"""
q12 = np.copy(cat)
m = np.array([[1, 0], [0, 0]])
q12 = q12 @ m

plotcat(q12, "q12 x,y => x,0")

# q13
"""
x,y => 0,y
"""
q13 = np.copy(cat)
m = np.array([[0, 0], [0, 1]])
q13 = q13 @ m

plotcat(q13, "q13 x,y => 0,y")

# q14
q14 = np.copy(cat)
ang = -pi / 3
rotm = np.array([[cos(ang), -sin(ang)], [sin(ang), cos(ang)]])
q14 = q14 @ rotm

plotcat(q14, "q14")

# q15

q15 = np.copy(cat)
ang = pi / 3
rotm = np.array([[cos(ang), -sin(ang)], [sin(ang), cos(ang)]])
q15 = q15 @ rotm

plotcat(q15, "q15")
exit()
