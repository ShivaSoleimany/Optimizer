#incremental delta bar delta

import numpy as np
import matplotlib.pyplot as plt


d = 20 
T = 10000
N = 1

h = np.zeros((20, 1))
beta = np.ones((20, 1)) * np.log(0.05)
alpha = np.exp(beta)
theta=0.05

ws = np.random.randint(0, 2, d)*2 - 1
ws=np.reshape(ws,(20,1))

w = np.random.normal(0.0, 1.0, size=(d, 1))
w=np.reshape(w,(20,1))

num_examples=100
delta_temp=np.zeros((1,T))
delta_tempv=np.zeros((1,T))

for i in range(N):
  for t in range(T):
    for example_i in range(num_examples):
      x = np.random.randint(0, 2, d)
      x=np.reshape(x,(20,1))
      Y = ws.T.dot(x) + np.random.randn()*0.1

      estimate = np.dot(w.T, x)
      D = Y - estimate
      beta += (theta * D * x * h)
      alpha = np.exp(beta)
      w += D * np.multiply(alpha, x)
      h = h * np.maximum((1.0 - alpha * (x ** 2)), np.zeros((20, 1))) + (alpha * D * x)

    delta_temp[i,t] = (Y - w.T.dot(x))**2

delta_t=bining(delta_temp,100)
#----------------------------------------------------------------
for i in range(N):
  for t in range(T):
    temp=int(t/20)
    for example_i in range(num_examples):
      x = np.random.randint(0, 2, d)*pow(-1,temp)
      x=np.reshape(x,(20,1))
      Y = ws.T.dot(x) + np.random.randn()*0.1

      estimate = np.dot(w.T, x)
      D = Y - estimate
      beta += (theta * D * x * h)
      alpha = np.exp(beta)
      w += D * np.multiply(alpha, x)
      h = h * np.maximum((1.0 - alpha * (x ** 2)), np.zeros((20, 1))) + (alpha * D * x)

    delta_tempv[i,t] = (Y - w.T.dot(x))**2

delta_t=bining(delta_temp,100)
delta_tv=bining(delta_tempv,100)




#########################################################################

plt.plot(deltas, 'b', label='const')
plt.plot(delta_t, 'r', label='idbd')
plt.ylabel('Error with theta=0.05, stationary')
plt.xlabel('T/100')
plt.legend()


plt.plot(delta, 'b', label='const')
#print(delta)
plt.plot(delta_tv, 'r', label='idbd')
plt.ylabel('Error with theta=0.05, non-stationary')
plt.xlabel('T/100')
plt.legend()
#plt.ylim(0,0.1)
