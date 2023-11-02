import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

sns.set_style("darkgrid")
sns.set_palette(sns.hls_palette(n_colors=14))
plt.rcParams['font.sans-serif'] = 'DengXian'

df = pd.read_csv("house_info.csv", encoding="utf-8")
district = df['区域'].to_numpy()
D = df['室'].to_numpy()
L = df['厅'].to_numpy()
area = df['房间大小'].to_numpy()
windows = df['窗户朝向'].to_numpy()
fixture = df['装修造型'].to_numpy()

year = df['修建年份'].to_numpy()
hType = df['房源'].to_numpy()
price = df['平米房价'].to_numpy()

districts = np.unique(district)
da = []
db = []
for district_name in districts:
    bool = [district == district_name]
    da.append(price[tuple(bool)])

for df in da:
    db.append(df[:2217])
da = np.asarray(db)
df = pd.DataFrame(da.T, columns=districts)
df.plot.hist(bins=100, alpha=0.5, figsize=(10, 6))
plt.show()
