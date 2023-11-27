import os
import json
import time
import pandas as pd


def get_neighbor_info() -> list:
    neighbor_info = []
    for district_dir in os.listdir("./cache/neighbor_info/"):
        neighbor_info.append(f"./cache/neighbor_info/{district_dir}")
    return neighbor_info


def get_house_info() -> list:
    house_info = []
    for district_dir in os.listdir("./cache/house_info/"):
        for file in os.listdir(f"./cache/house_info/{district_dir}"):
            house_info.append(f"./cache/house_info/{district_dir}/{file}")
    return house_info


def run():
    house_info = get_house_info()
    neighbor_info = get_neighbor_info()
    neighborNo = []
    neighborInfo = []
    for path in neighbor_info:
        with open(path, "r", encoding="utf-8") as f:
            info = json.load(f)
            neighbor_no = path.split("/")[-1].split(".")[0]
            neighborNo.append(neighbor_no)
            neighborInfo.append(info)
    houseInfo = []
    for path in house_info:
        with open(path, "r", encoding="utf-8") as f:
            info = json.load(f)
            info.update(neighborInfo[neighborNo.index(info['neighbor_no'])])
            info.pop('neighbor_no')
            houseInfo.append(info)
    pd.DataFrame(houseInfo).to_csv("./cache/house_info.csv", encoding="utf-8-sig", index=False)
    #
    # for info in houseInfo:
    #     district.append(info['district'])


if __name__ == "__main__":
    pass
