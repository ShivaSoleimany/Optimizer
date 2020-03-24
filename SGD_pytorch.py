import torch as tor
from torch.nn import MSELoss
from torch.optim import SGD

g = 0.99
d = 20
T = 1000
N = 1
alpha = 0.1/d

ws = tor.randint(0, 2, (d,), dtype=tor.float)*2 - 1
w = tor.zeros(d, requires_grad=True)
loss_func = MSELoss()



print("lr", 2*alpha)
print("weights", w,'\n\n------------------------------------')
print("weights", [w])
opt = SGD([w], lr=2*alpha)
for i in range(N):
    for t in range(T):
        x = tor.randint(0, 2, (d,), dtype=tor.float)
        Y = ws.dot(x) + np.random.randn()*0.1
        loss = loss_func(w.dot(x), Y)
        loss.backward()
        opt.step()
        opt.zero_grad()
