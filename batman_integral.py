import numpy as np
import matplotlib.pyplot as plt


#batman curve for positive values of y
def batmanUpper(x):
    if(-0.75<x<-0.5 or 0.5<x<0.75):
        return 3 * np.abs(x) + 0.75
    elif(-0.5<x<0.5):
        return 2.25
    elif(-1<x<-0.75 or 0.75<x<1):
        return 9 - 8 * np.abs(x)
    elif(-3<x<-1 or 1<x<3):
        return (6 * np.sqrt(10) / 7) + (1.5 - 0.5 * np.abs(x)) - (6 * np.sqrt(10) / 14) * np.sqrt(4 - (np.abs(x)-1)**2)
    elif(-7<x<-3 or 3<x<7):
        return np.sqrt(9 * (1 - (x/7)**2))

#batman curve for negative values of y
def batmanLower(x):
    if(-7<x<-4 or 4<x<7):
        return -1 * np.sqrt(9 * (1 - (x/7)**2))
    elif(-4<x<4):
        return (np.abs(x/2) - ((3 * np.sqrt(33) - 7)/112) * x**2 - 3) + np.sqrt(1 - (np.abs(np.abs(x)-2)-1)**2)


approximations = []

#limits of integration
a = -7
b = 7

#amount of random numbes generated between a and b
N= [1,2,5,10,25,50,100,250,500,1000,2500,5000,10000,25000,50000,100000,250000,500000,1000000]

for n in N:
    integral = 0.0
    randomNumsX = np.zeros(n)
    for i in range (n): 
        randomNumsX[i] = np.random.uniform(a,b)
    for i in range(n):
        integral += np.abs(batmanLower(randomNumsX[i]))
        integral += np.abs(batmanUpper(randomNumsX[i]))
    answer = (b-a)/(n)*integral 
    approximations.append(answer)

print(approximations)

#Plotting Batman Function
##########################################
x = np.linspace(-7,7,100000)
y1 = np.zeros(100000)
y2 = np.zeros(100000)
for i in range(100000):
    y1[i] = batmanUpper(x[i])
for j in range(100000):
    y2[j] = batmanLower(x[j])

plt.axhline(y=0, color='k', linewidth=0.7)
plt.axvline(x=0, color='k', linewidth=0.7)
plt.plot(x,y1,'b')
plt.plot(x,y2,'b')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()
###########################################

#Plotting Approximations against N
################################################
#plt.plot(N,approximations, label='approximation')
#plt.xscale("log")
#plt.axhline(y=48.42398, color = 'r',linewidth = 0.75, label = r'$y=48.42398$')
#plt.xlabel(r"$N$")
#plt.ylabel("Approximation to integral")
#plt.grid()
#plt.legend()
#plt.show()
#################################################

##errorApproximations = np.zeros(len(N))
#for i in range(len(N)):
#    errorApproximations[i] = np.abs(approximations[i] - 48.42398)

#plt.plot(1/np.sqrt(N),N)
#plt.plot(errorApproximations, N)
#plt.xscale("log")
#plt.yscale("log")
#plt.grid()
#plt.show()