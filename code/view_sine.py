#!/usr/bin/env python3

import pickle
import numpy as np
from matplotlib import pyplot as plt

with open('/tmp/lvae/dump-007.pkl', 'rb') as fp:
    data = pickle.load(fp)

print(data['x'].shape)

fig, axs = plt.subplot_mosaic([[str(i)] for i in range(4)])
for i in range(4):
    ax = axs[str(i)]
    try:
        ax.plot(data['x'][i, 0, :, 0].ravel(), label='x')
        ax.plot(data['x_hat'][i, 0, :, 0].ravel(), label='x_hat')
    except IndexError:
        print(data['x'].shape,
              data['x_hat'].shape)
        # ax.plot(data['x'][i, 0, :].ravel(), label='x')
        # ax.plot(data['x_hat'][i, 0, :].ravel(), label='x_hat')
        ax.plot(data['x'][i].ravel(), label='x')
        ax.plot(data['x_hat'][i].ravel(), label='x_hat')
plt.legend()

# plt.plot(data['x'][0,0,:,1].ravel(), label='x')
# plt.plot(data['x_hat'][0,0,:,1].ravel(), label='x_hat')
plt.show()
