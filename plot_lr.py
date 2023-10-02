import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from tvregdiff import TVRegDiff

alpha = 1e-3

for i in range(6,7):
    print(i)
    file = '../saved_models/GBIP/learning_rate/losses_k0{}.txt'.format(i)
    df = pd.read_fwf(file)
    d = np.array(df['total'])
    s = TVRegDiff(d, 100, alpha, plotflag=False, precondflag=True, diffkernel='sq')
    plt.plot(df['lr'], s)

# file = '../saved_models/GBIP/learning_rate/losses_k0{}.txt'.format(4)
# df = pd.read_fwf(file)
# d = np.array(df['total'])
# for i in range(-4, 0):
#     print(i, 10 ** i)
#     alpha = 10 ** i
#     s = TVRegDiff(d, 100, alpha, plotflag=False, precondflag=True, diffkernel='sq')
#     plt.plot(df['lr'], s)

# for i in range(9, 10):
#     print(i)
#     file = '../saved_models/GBIP/learning_rate/losses_k0{}[lr1].txt'.format(i)
#     df = pd.read_fwf(file)
#     d = np.array(df['total'])
#     s = TVRegDiff(d, 100, alpha, plotflag=False, precondflag=True, diffkernel='sq')
#     plt.plot(df['lr'][:416], s[:416])

plt.ylim(top=1)
plt.xscale('log')
plt.legend('6')
plt.show()
