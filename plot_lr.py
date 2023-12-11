import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# from tvregdiff import TVRegDiff

# alpha = 1e-3

colors = ['#0C2340', '#00B8C8', '#0076C2', '#6F1D77', '#EF60A3', '#A50034', '#E03C31', '#EC6842', '#FFB81C', '#6CC24A', '#009B77']
grays = ['#888', '#9F9F9F', '#BBB', '#CFCFCF', '#EEE']

r1 = 20
r2 = 20

fig, axes = plt.subplots(1, 2, figsize=(14, 3))

# file = "C:\\Users\\benja\\Desktop\\k092.txt"
# file = '../saved_models/GBIP/experiments_full/update_lr/k091/lr4_losses.txt'
# df = pd.read_fwf(file).tail(-400).head(-50)
# df['new'] = df['box'] + df['obj'] + df['cls']
# axes[0].plot(df['lr'], df['new'].rolling(r1).mean(), color=colors[2], zorder=10)
# axes[1].plot(df['lr'], df['new'].rolling(r1).mean().diff().rolling(r2).mean(), color=colors[2], zorder=10)

file = "../saved_models/GBIP/learning_rate/k09AT.txt"
df = pd.read_fwf(file).tail(-400).head(-50)
df['new'] = df['box'] + df['obj'] + df['cls']
axes[0].plot(df['lr'], df['new'].rolling(r1).mean(), color=colors[2], zorder=10)
axes[1].plot(df['lr'], df['new'].rolling(r1).mean().diff().rolling(r2).mean(), color=colors[2], zorder=10)

file = "../saved_models/GBIP/learning_rate/k09OT.txt"
df = pd.read_fwf(file).tail(-400).head(-50)
df['new'] = df['box'] + df['obj'] + df['cls']
axes[0].plot(df['lr'], df['new'].rolling(r1).mean(), color=grays[0])
axes[1].plot(df['lr'], df['new'].rolling(r1).mean().diff().rolling(r2).mean(), color=grays[0])

file = "../saved_models/GBIP/learning_rate/k09AG.txt"
df = pd.read_fwf(file).tail(-400).head(-50)
df['new'] = df['box'] + df['obj'] + df['cls']
axes[0].plot(df['lr'], df['new'].rolling(r1).mean(), color=grays[1])
axes[1].plot(df['lr'], df['new'].rolling(r1).mean().diff().rolling(r2).mean(), color=grays[1])

file = "../saved_models/GBIP/learning_rate/k09ATOT.txt"
df = pd.read_fwf(file).tail(-400).head(-50)
df['new'] = df['box'] + df['obj'] + df['cls']
axes[0].plot(df['lr'], df['new'].rolling(r1).mean(), color=grays[2])
axes[1].plot(df['lr'], df['new'].rolling(r1).mean().diff().rolling(r2).mean(), color=grays[2])

file = "../saved_models/GBIP/learning_rate/k09None.txt"
df = pd.read_fwf(file).tail(-400).head(-50)
df['new'] = df['box'] + df['obj'] + df['cls']
axes[0].plot(df['lr'], df['new'].rolling(r1).mean(), color=colors[7])
axes[1].plot(df['lr'], df['new'].rolling(r1).mean().diff().rolling(r2).mean(), color=colors[7])


# axes[0].set_xlim(4e-6, 3e-2)
axes[0].set_xlim(2e-5, 3e-2)
axes[1].set_xlim(2e-5, 3e-2)
axes[0].set_xscale('log')
axes[1].set_xscale('log')
axes[0].grid(True, 'both')
axes[1].grid(True, 'both')
axes[0].set_xlabel('learning rate')
axes[0].set_ylabel('loss')
axes[1].set_xlabel('learning rate')
axes[1].set_ylabel('Δloss')
axes[0].legend(['AT', 'OT', 'AG', 'AT+OT', 'None'])
axes[1].legend(['AT', 'OT', 'AG', 'AT+OT', 'None'])
plt.savefig("test.png", bbox_inches='tight')
plt.show()


# file = "../saved_models/GBIP/learning_rate/k08AT.txt"
# df = pd.read_fwf(file).tail(-200).head(-50)
# df['new'] = df['box'].cumsum() + df['obj'].cumsum() + df['cls'].cumsum()
# plt.plot(df['lr'], df['new'].rolling(r).mean().diff().rolling(r).mean())

# file = "../saved_models/GBIP/learning_rate/k08OT.txt"
# df = pd.read_fwf(file).tail(-200).head(-50)
# df['new'] = df['box'].cumsum() + df['obj'].cumsum() + df['cls'].cumsum()
# plt.plot(df['lr'], df['new'].rolling(r).mean().diff().rolling(r).mean())

# file = "../saved_models/GBIP/learning_rate/k08ATOT.txt"
# df = pd.read_fwf(file).tail(-200).head(-50)
# df['new'] = df['box'].cumsum() + df['obj'].cumsum() + df['cls'].cumsum()
# plt.plot(df['lr'], df['new'].rolling(r).mean().diff().rolling(r).mean())



# file = '../saved_models/GBIP/learning_rate/losses_k08_AT_OT[lr1].txt'
# file = "C:\\Users\\benja\\Desktop\\k08None.txt"
# df = pd.read_fwf(file)
# d = np.array(df['box'].cumsum()) + np.array(df['obj'].cumsum()) + np.array(df['cls'].cumsum())
# d = np.array(df['total'].cumsum())
# d /= np.arange(1, len(d)+1)
# df['new'] = df['total'][10:-50]
# d2 = df['new'].rolling(10).mean().diff().rolling(10).mean()
# plt.plot(df['lr'], d2*10)
# s = TVRegDiff(d, 100, alpha, plotflag=False, precondflag=True, diffkernel='sq')
# plt.plot(df['lr'][10:-10], s[10:-10])
# n = np.arange(1, len(d)+1)
# df['lr_new'] = df['lr'][200:-50]
# df['total_new'] = df['total'][200:-50]
# d = np.array(df['box'][200:-50].cumsum()) + np.array(df['obj'][200:-50].cumsum()) + np.array(df['cls'][200:-50].cumsum())
# d /= np.arange(1, len(d)+1)
# df['new'] = d
# df = df.tail(-200).head(-50)
# df['new'] = df['box'].cumsum() + df['obj'].cumsum() + df['cls'].cumsum()
# # df['total_new'] = df['total_new'].cumsum() / np.arange(1, len(df['total_new'])+1)
# plt.plot(df['lr'], df['new'].rolling(5).mean().diff().rolling(5).mean())

# file = '../saved_models/GBIP/learning_rate/losses_k08_AT_OT[lr5].txt'
# df = pd.read_fwf(file)
# d = np.array(df['total'])
# d2 = df['total'].rolling(20).mean().diff().rolling(20).mean()
# plt.plot(df['lr'], d2*200)
# s = TVRegDiff(d, 100, alpha, plotflag=False, precondflag=True, diffkernel='sq')
# # plt.plot(df['lr'][10:-130], s[10:-130])
# plt.plot(df['lr'], s)

# for i in range(6,7):
#     print(i)
#     file = '../saved_models/GBIP/learning_rate/losses_k0{}.txt'.format(i)
#     df = pd.read_fwf(file)
#     d = np.array(df['total'])
#     d2 = df['total'].rolling(20).mean().diff().rolling(20).mean()
#     plt.plot(df['lr'], d2*200)
#     s = TVRegDiff(d, 100, alpha, plotflag=False, precondflag=True, diffkernel='sq')
#     plt.plot(df['lr'], s)

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

# plt.ylim((-1,1))
