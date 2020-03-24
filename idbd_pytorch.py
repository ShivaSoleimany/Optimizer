#incremental delta bar delta by Pytorch
import numpy as np
import matplotlib.pyplot as plt
import torch as tor
from torch.nn import MSELoss
from torch.optim import SGD

d = 2
T = 100
N = 1
theta=0.05
num_examples=10

h=tor.zeros((d, ), dtype=tor.float)
beta=tor.ones((d, ))
alpha = tor.exp(beta)

ws =tor.randint(0, 2, (d,), dtype=tor.float, )*2 - 1
w=tor.zeros(d,requires_grad=True)

loss_func = MSELoss()
#opt=SGD([{'params': [p], 'lr': l} for p,l in zip(w, alpha)])

w1=tor.zeros(1,requires_grad=False)
w2=tor.zeros(1,requires_grad=False)

opt1=SGD([w1],lr=alpha[0])
opt2=SGD([w2],lr=alpha[1])

delta_temp=tor.zeros((1,T))
delta_tempv=tor.zeros((1,T))
#--------------------------------------------------------------------
for i in range(N):
  for t in range(T):
    for example_i in range(num_examples):

      x = tor.randint(0, 2, (d,), dtype=tor.float)
      Y = ws.dot(x) + np.random.randn()*0.1

      loss = loss_func(tor.tensor([w1.item(),w2.item()],requires_grad=True ).dot(x), Y)

      estimate = tor.tensor([w1.item(),w2.item()]).dot(x)
      D = Y - estimate
      beta += (theta * D * x * h)
      alpha = tor.exp(beta)
      #print(D)
      w1 += D * tor.mul(alpha[0], x[0])
      w2 += D * tor.mul(alpha[1], x[1])
      #loss.backward()
     # opt1.step()
      #opt2.zero_grad()

      h = h * tor.max((1.0 - alpha * (x ** 2)), tor.zeros(d,)) + (alpha * D * x)

    delta_temp[i,t] = (Y - tor.tensor([w1.item(),w2.item()]).dot(x))**2


d=[]
for t in delta_temp.data[0]:
  d.append(t.item())

#########################################################################

#plt.plot(deltas, 'b', label='const')
plt.plot(d, 'r')
plt.ylabel('Error with theta=0.05, stationary')
plt.xlabel('T/10')
plt.legend()
