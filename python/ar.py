#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

def GetMuTilde(buf, buf_idx, a, sigma):
    result = sum([a[ii]*buf[(buf_idx+len(buf)-ii-1)%len(buf)] for ii in range(len(a))]) \
            + sigma*np.random.normal()
    return result
mu = 0.142
a = [0.783, 0.116]
sigma = 0.0148
# sigma = 0.
buf = [0]*3
buf_idx = 0

data = []
for ii in range(200):
    print ii
    mu_tilde = GetMuTilde(buf, buf_idx, a, sigma)
    buf_idx = (buf_idx+1) % (len(a)+1)
    # data.append(mu + mu_tilde)
    data.append(mu_tilde)
data = np.array(data)
plt.figure()
plt.plot(data, '-x')
plt.figure()
plt.specgram(data, Fs=44100.)
plt.show()
