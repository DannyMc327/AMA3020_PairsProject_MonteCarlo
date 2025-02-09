import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm



def gauss2D(x,y):
    return np.exp(-x**2-y**2)

def f(x,y):
    return 2*x**2 + y**2

approximations = []

#limits of integration
a = -2
b = 2

#amount of random numbes generated between a and b
N= [2,3,4,5,7,10,15,20,25,38,50,75,100,150,200,250,350,500,750,1000,1500,2000,2500,5000]

for n in N:
    integral = 0.0
    randomNumsX = np.zeros(n)
    randomNumsY = np.zeros(n) 
    for i in range (n): 
        randomNumsX[i] = np.random.uniform(a,b)
        randomNumsY[i] = np.random.uniform(a,b)
    for i in range(n):
        for j in range(n):
            integral += gauss2D(randomNumsX[i],randomNumsY[j])
    answer = integral*((b-a)/float(n))**2
    approximations.append(answer)

print(approximations)

#3D plot of gauss2D function
###################################################
#x = np.linspace(-3,3,100)
#y = np.linspace(-3,3,100)
#X, Y = np.meshgrid(x, y)
#fig = plt.figure()
#ax = plt.axes(projection='3d')
#ax.plot3D(x,y,gauss2D(x,y), label='test')
#ax.plot_surface(X, Y, gauss2D(X,Y),cmap=cm.plasma)
####################################################

plt.plot(N, approximations, label='Approximation')
plt.xscale("log")
plt.xlabel(r"$N$")
plt.ylabel(r"Approximation to $f \ (x,y)$")
plt.axhline(y=np.pi, color='r', label=r'$y=\pi$', linestyle='dotted')
plt.grid()
plt.legend()
plt.show()