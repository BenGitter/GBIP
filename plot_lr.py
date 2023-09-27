import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_fwf('./tmp/training298/losses.txt')
# print(df['lr'][0])

plt.plot(df['lr'], df['total'])
plt.xscale('log')
plt.show()