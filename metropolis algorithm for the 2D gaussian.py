
#Metropolis-Hastings algorithm code for multiple dimensions


###parameters ###


#A burn-in iteration is an initial iteration in a Markov chain (MCMC) that is discarded before
#the chain converges to its target distribution. The burn-in period is the number of iterations
#that are discarded before the chain is considered to have "forgotten" its initial value.

burnin = 0  # number of burn-in iterations
lag = 1  # iterations between successive samples
nsamp = 100000 # number of samples to draw
sig = 1 # standard deviation of Gaussian proposal
x = (-2, -2)# start point

    ### storage ###
X = np.zeros((nsamp,2)) # samples drawn from the Markov chain
acc = np.array((0, 0))  # vector to track the acceptance rate


def targetdist(x):
        probX = np.exp(-x[0]**2-x[1]**2)
        return probX

def MHstep(x0, sig):
        xp = (np.random.normal(loc = x0[0], scale = sig), np.random.normal(loc = x0[1], scale = sig))  # generate candidate from Gaussian
        accprob = targetdist(xp) / targetdist(x0) # acceptance probability
        u = np.random.rand() # uniform random number
        if u <= accprob: # if accepted
            x1 = xp # new point is the candidate
            a = 1 # note the acceptance
        else: # if rejected
            x1 = x0 # new point is the same as the old one
            a = 0 # note the rejection
        return x1, a




# MH routine
for i in range(burnin):
        x,a = MHstep(x,sig); # iterate chain one time step
        acc = acc + np.array((a, 1)) # track accept-reject status

for i in range(nsamp):
    for j in range(lag):
        x,a = MHstep(x,sig) # iterate chain one time step
        acc = acc + np.array((a, 1)) # track accept-reject status
    X[i] = x # store the i-th sample
df = pd.DataFrame(data=X, columns = ['Trace1', 'Trace2'])



X = df['Trace1'].values
Y = df['Trace2'].values


# Create a grid of (X, Y) coordinates
num_bins = 100  # Adjust as needed
x_edges = np.linspace(X.min(), X.max(), num_bins + 1)
y_edges = np.linspace(Y.min(), Y.max(), num_bins + 1)
X_grid, Y_grid = np.meshgrid(x_edges[:-1], y_edges[:-1])

# Calculate the proportion of counts for each bin in the grid
Z = np.zeros_like(X_grid, dtype=float)
for i in range(num_bins):
    for j in range(num_bins):
        mask = (X >= x_edges[i]) & (X < x_edges[i + 1]) &                (Y >= y_edges[j]) & (Y < y_edges[j + 1])
        Z[j, i] = np.sum(mask) / len(X)  # Proportion of counts in the bin

# Create a 3D figure and axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(X_grid, Y_grid, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Proportion of Counts')
ax.set_title('Estimation of Gaussian distribution in 2 dimensions');


plt.show()



# Calculate histogram data
hist_X, bins_X = np.histogram(X, bins=20)  # Adjust bins as needed
hist_Y, bins_Y = np.histogram(Y, bins=20)  # Adjust bins as needed

# Calculate error limits (e.g., standard deviation)
err_X = np.sqrt(hist_X)  # Assuming Poisson error for counts
err_Y = np.sqrt(hist_Y)  # Assuming Poisson error for counts

# Create histogram plot
plt.hist(df.values, bins=20, density=True)  # Keep the original histogram

# Add error bars for orange bars (Trace1)
bin_centers_X = (bins_X[:-1] + bins_X[1:]) / 2
plt.errorbar(bin_centers_X, hist_X, yerr=err_X, fmt='none', ecolor='blue', capsize=3, label='X-Cord')

# Add error bars for blue bars (Trace2)
bin_centers_Y = (bins_Y[:-1] + bins_Y[1:]) / 2
plt.errorbar(bin_centers_Y, hist_Y, yerr=err_Y, fmt='none', ecolor='orange', capsize=3, label='Y-Cord')

# Set labels, title, etc.
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram with Error Limits')
plt.legend()

# Display the plot
plt.show()
