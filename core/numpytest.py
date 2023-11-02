import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

sns.set(style="darkgrid")
colors = sns.dark_palette('orange',12)
sns.set_palette(colors)
x = np.linspace(-np.pi, np.pi, 100)
sin = np.fabs(np.sin(x))
y = {
    'sin': sin,
    'sin2': sin * 2,
    'sin3': sin * 3,
    'sin4': sin * 4,
    'sin5': sin * 5,
    'sin6': sin * 6,
    'sin7': sin * 7,
    'sin8': sin * 8,
    'sin9': sin * 9,
    'sin10': sin * 10,
    'sin11': sin * 11,
    'sin12': sin * 12,
}
df = pd.DataFrame(y, index=x)
df.plot.area()
plt.show()
