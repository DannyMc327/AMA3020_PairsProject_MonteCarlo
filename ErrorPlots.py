import numpy as np
import matplotlib.pyplot as plt

def f(x,y):
    return 2*x**2 + y**2

#limits of integration
a = 0
b = 2

#amount of random numbes generated between a and b
N= [1,10,100,1000]

listOfApproximations = []

for i in range(100):
    approximations = []
    for n in N:
        integral = 0.0
        randomNumsX = np.zeros(n)
        randomNumsY = np.zeros(n) 
        for i in range (n): 
            randomNumsX[i] = np.random.uniform(a,b)   #integrating between 0 and 2
            randomNumsY[i] = np.random.uniform(a,b-1) #integrating between 0 and 1
        for i in range(n):
            integral += f(randomNumsX[i],randomNumsY[i])
        answer = (b-a)/float(n)*integral 
        approximations.append(answer)

    #print(approximations)
    listOfApproximations.append(approximations)

arrayOfApproximations = np.array(listOfApproximations)
#print(arrayOfApproximations)
averagedApproximations = np.zeros(len(N))
for i in range(len(N)):
    for j in range(len(N)):
        averagedApproximations[i]+=arrayOfApproximations[j,i]
    averagedApproximations[i]=averagedApproximations[i]/len(N)

print(averagedApproximations)

errorApproximations = np.abs(averagedApproximations-6) #exact value of this integral is 6
print(errorApproximations)

lineBestFit = np.polyfit(1/np.sqrt(N),errorApproximations,1)
lineBestFit = np.flip(lineBestFit)
plt.plot(1/np.sqrt(N),errorApproximations, label=r'$E\propto \dfrac{1}{\sqrt{N}}$')
plt.xlabel(r"$\dfrac{1}{\sqrt{N}}$")
plt.ylabel("Actual error")
plt.plot(lineBestFit, label='Line of best fit', linestyle=':', color='r')
plt.grid()
plt.legend()
plt.show()