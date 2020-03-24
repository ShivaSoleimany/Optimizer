import numpy as np
import matplotlib.pyplot as plt

g = 0.99
d = 20 
T = 2000
N = 1
alpha = 0.1/d



np.random.seed(0)
ws = np.random.randint(0, 2, d)*2 - 1
w = np.zeros(d)
deltas_const, delta_const = np.zeros((N, T)), np.zeros((N, T))
deltas_var, delta_var=np.zeros((N,T)),np.zeros((N,T))



bin=np.zeros(())
####################################################################
def bining(A, b):
  bin=np.zeros(b)
  l=A.shape[1]
  interval=int(l/b)
  mean=0
  for i in range (0,l,interval): 
    k=int(i/interval) 
    mean=np.mean(A[0][k*interval:(k+1)*interval])
    bin[k]=mean
  return bin
######################################################################
#Stationary and constant learning rate
for i in range(N):
    np.random.seed(i)
    for t in range(T):
        x = (np.random.randint(0, 2, d))
        Y = ws.dot(x) + np.random.randn()*0.1
        w = w + alpha*(Y - w.dot(x))*x
        deltas_const[i,t] = (Y - w.dot(x))**2

w = np.zeros(d)

#non-stationary and constant learning rate
for i in range(N):
    for t in range(T):
        temp=int(t/20)
        x = (np.random.randint(0, 2, d))*pow(-1,temp)
        Y = ws.dot(x) + np.random.randn()*0.1
        w = w + alpha*(Y - w.dot(x))*x
        delta_const[i,t] = (Y - w.dot(x))**2

##-----------------------------------------------------------------------
w = np.zeros(d)

#stationary and decreasing learning rate
for i in range(N):
    for t in range(T):
      x = np.random.randint(0, 2, d)
      Y = ws.dot(x) + np.random.randn()*0.1
      alpha=1/(t+1)
      w = w + alpha*(Y - w.dot(x))*x
      deltas_var[i,t] = (Y - w.dot(x))**2

w = np.zeros(d)

#non-stationary and decreasing learning rate
for i in range(N):
    for t in range(T):
      temp=int(t/20)
      x = np.random.randint(0, 2, d)*pow(-1,temp)
      Y = ws.dot(x) + np.random.randn()*0.1
      alpha=1/(t+1)
      w = w + alpha*(Y - w.dot(x))*x
      delta_var[i,t] = (Y - w.dot(x))**2

#-----------------------------------------------------------------------
deltas, delta=bining(deltas_const,100), bining(delta_const,100)
delts, delt=bining(deltas_var,100), bining(delta_var,100)

figure, axes = plt.subplots(nrows=2, ncols=1)

axes[0].plot(deltas, 'b', label='const')
axes[0].plot(delts, 'r', label='decrease')
axes[0].set_ylabel('stationary')
axes[0].legend()


axes[1].plot(delta,'b',label='const')
axes[1].plot(delt,'r',label='decrease')
axes[1].set_ylabel('non-stationary')
axes[1].set_yscale('log')
axes[1].legend()

plt.xlabel('T')
figure.tight_layout()
