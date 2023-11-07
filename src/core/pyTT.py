import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sns.set_style("whitegrid")
sns.set_palette(sns.hls_palette(s=0.6, n_colors=10))

t_x = torch.linspace(-np.pi, np.pi, 200, requires_grad=True)
x = t_x.detach().numpy()

t_sin = torch.sin(t_x)
ori = t_sin.detach().numpy()

t_sin = torch.sum(t_sin)
t_sin.backward()
out = t_x.grad
dif = out.detach().numpy()

print(out)
out = np.array([ori, dif], dtype=np.float64).T
x = pd.DataFrame(out, index=x.T, columns=["ori", "dif"])
x.plot()
print(x)
plt.show()
