# In this script we simulate cobweb model with ANN with sigmoid activation function
# ANN learns parameters alpha2 and beta2 and converges to REE values
# References used for this code are Evans & Honkapohja (2001) and notes from briandolhansky.com
# code is written by Sarunas Girdenas, August, 2014, sg325@exeter.ac.uk


import matplotlib.pyplot as plt
import numpy as np

# Load data from matlab

Shocks = np.genfromtxt('Shocks_var.txt', delimiter=',') # shocks
g_var  = np.genfromtxt('w_lag.txt', delimiter=',')      # g (exogenous variable)

# Specify parameters for the cobweb model

alpha_1  = 5                        # price equation intercept
c        = -0.5                     # beta_0 + beta_1 in Economic Model
sigma    = 0.5                      # variance of shock
time     = 500                      # simulation horizon
a        = np.zeros([time,1])       # expected price level
p        = np.zeros([time,1])       # actual price level
alpha_2  = np.zeros([time,1])       # parameter alpha_2 in Economic Model
beta_2   = np.zeros([time,1])       # parameter beta_2 in Economic Model
delta    = 1                        # define delta next to w in price equation
g_lag    = np.zeros([len(g_var),1]) # exogenous variable g, lagged by 1, g(t-1)

# create g_lag now

for z in range(1,len(g_var)):
    g_lag[z] = g_var[z-1]

# REE values

a2 = alpha_1/(1-c)
b2 = delta/(1-c)

# initialize OLS parameters

beta_2_initial  = 2 
alpha_2_initial = 1 


# initialize p and a variables of the model

a[1] = alpha_2_initial+beta_2_initial*g_lag[1]
p[1] = alpha_1+c*a[1]+delta*g_lag[1]+sigma*Shocks[1] #Initial values of p and a

# Neural Network initialization

max_iter = 100;                # no of network iterations
alpha_n  = 0.01;             # gradient descent learning rate, calibrate it to change convergence properties of ANN
w        = np.zeros([1,2])     # weights for Neural Network 
grad_t_h = np.zeros([time,2])  # store network activation function



for i in range(2,time):

    print 'sim_no = ', i

    # macroeconomic model

    a[i] = alpha_2[i-1] + beta_2[i-1] * g_var[i-1]
    p[i] = alpha_1 + c * a[i] + delta * g_var[i-1] + sigma * Shocks[i]

    # update parameters alpha_2 and beta_2 using Neural Network

    for k in range(1,max_iter):
            
        X      = np.zeros([i,2])
        X[:,0] = np.ones([1,i])
        X[:,1] = g_lag[0:i].T
        Y      = p[0:i]
        grad_t = np.array([0., 0.])

        for t in range(1,i):

            x_t = X[t,:]
            y_t = Y[t]

            # compute hypothesis

            h = np.dot(w,x_t) - y_t
            grad_t = grad_t + 2*h*x_t*np.exp(-np.dot(w,x_t))/((1+np.exp(-np.dot(w,x_t))**(2)))
            B = 2*h*x_t*np.exp(-np.dot(w,x_t))/((1+np.exp(-np.dot(w,x_t))**(2)))
            
    # update weights

    w = w - alpha_n*grad_t

    # update economic model estimates
    alpha_2[i] = w[0][0]
    beta_2[i]  = w[0][1]

    # store activation function

    grad_t_h[i,:] = B

# plot results

plt.figure
plt.plot(alpha_2,'k-.o')
plt.xlabel('Time Horizon')
plt.title('Coefficient Alpha_2')
plt.show()

plt.figure
plt.plot(beta_2,'m-.o')
plt.xlabel('Time Horizon')
plt.title('Coefficient Alpha_2')
plt.show()

plt.figure
plt.plot(grad_t_h[:,0],'k-.o',label='Activation Function for Alpha_2')
plt.plot(grad_t_h[:,1],'m-.o',label='Activation Function for Beta_2')
plt.xlabel('Time Horizon')
plt.title('Sigmoid Activation Function')
plt.legend(loc='upper right')
plt.show()