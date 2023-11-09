import pandas as pd
import numpy as np

info = pd.read_csv("./house_info_new.csv")
columns = info.columns
np_district = info['区域'].to_numpy()

np_D = info['室'].to_numpy()

np_L = info['厅'].to_numpy()

np_area = info['房间大小'].to_numpy()

np_east = info['东'].to_numpy()
np_south = info['南'].to_numpy()
np_west = info['西'].to_numpy()
np_north = info['北'].to_numpy()

np_fixture = info['装修造型'].to_numpy()

np_year = info['修建年份'].to_numpy()

np_hType = info['房源'].to_numpy()
np_lift = info['电梯'].str.strip().to_numpy()
print(np_lift)
lift = []
for x in np.nditer(np_lift):
    if x == '有':
        lift.append(1)
    else:
        lift.append(0)
print(np.asarray(lift))

np_env = info['绿化率'].to_numpy()

np_price = info['平米房价'].to_numpy()
