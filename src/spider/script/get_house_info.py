import queue
import tqdm
import time
import multiprocessing
import threading
from lxml import etree
from bs4 import BeautifulSoup
import os
import re

import spider
from spider.script import *


def get_house_raw() -> list:
    house_raw = []
    for district_dir in os.listdir("../cache/house_raw/"):
        # for file in os.listdir(f"./cache/house_raw/{district[i]}"):
        #     house_raw.append(f"./cache/house_raw/{district[i]}/{file}")
        for file in os.listdir(f"../cache/house_raw/{district_dir}"):
            house_raw.append(f"../cache/house_raw/{district_dir}/{file}")
    return house_raw


def get_house_info(house_raw: str):
    file_path = house_raw.replace('house_raw/', 'house_info/')
    with open(house_raw, encoding="utf-8") as f:
        html = f.read()
    # position = re.search(r'resblockPosition:\s*\'([0-9.]+,[0-9.]+)\'', html).group(1).split(',')
    # position = (float(position[0]), float(position[1]))
    soup = BeautifulSoup(html, "lxml")
    house_info = [pair.text.split() for pair in soup.select("li")]
    if len(house_info) != 20:
        raise ValueError(f"Can not find enough items in {house_raw}")
    housePlan = house_info.pop(0)[1]
    houseArea = house_info.pop(0)[1]
    houseType = house_info.pop(0)[1]
    buildingType = house_info.pop(0)[1]
    houseOrientation = house_info.pop(2)[1]
    print(housePlan, houseArea, houseType, buildingType)
    print(house_info)


def process_get_house_info(process_no: int, house_row, process_queue):
    thread_queue = queue.Queue()
    thread_list = []
    for thread_no in range(MAX_THREAD):
        child_thread = threading.Thread(
            name=f"Thread-{thread_no}", target=thread_get_house_info,
            args=(process_no, thread_no, house_row, thread_queue))
        thread_list.append(child_thread)
        child_thread.start()
    while True:
        try:
            completed_task = thread_queue.qsize()
            while completed_task > 0:
                process_queue.put(thread_queue.get_nowait())
                completed_task -= 1
        except queue.Empty:
            pass
        active_threads = [thread for thread in thread_list if thread.is_alive()]
        if not active_threads:  # 如果没有活动的子线程，退出循环
            break
        time.sleep(0.2)
    print(f"\rProcess {process_no} finished")


def thread_get_house_info(process_no: int, thread_no: int, house_raw, queue_list):
    i = process_no + MAX_PROCESS * thread_no
    max_iter = len(house_raw)
    while i < max_iter:
        get_house_info(house_raw[i])
        queue_list.put(i)
        i += MAX_PROCESS * MAX_THREAD
    print(f"\rThread{process_no}-{thread_no} finished")


def run():
    t = time.time()
    pool = multiprocessing.Pool(processes=MAX_PROCESS)

    target = get_house_raw()
    manager = multiprocessing.Manager()
    progress_queue = manager.Queue()
    pbar = tqdm.tqdm(total=len(target))

    process_list = []
    for i in range(MAX_PROCESS):
        process = pool.apply_async(process_get_house_info, args=(i, target, progress_queue))
        process_list.append(process)
    while True:
        completed = progress_queue.qsize()
        pbar.update(completed)
        while completed > 0:
            progress_queue.get_nowait()
            completed -= 1
        if all(process.ready() for process in process_list):
            break
        time.sleep(0.2)
    pool.close()
    pool.join()
    manager.shutdown()
    print(time.time() - t)


if __name__ == "__main__":
    paths = get_house_raw()
    for path in paths:
        get_house_info(path)
# html = spider.request("https://sh.lianjia.com/ershoufang/107107606471.html")
# t = time.time()
# match = re.search(r'resblockPosition:\s*\'([0-9.]+,[0-9.]+)\'', html)
# print(match.group(1), time.time() - t)
# t = time.time()
# html = spider.request("https://sh.lianjia.com/ershoufang/107107606471.html", xpath="//script")
# match = re.search(r'resblockPosition:\s*\'([0-9.]+,[0-9.]+)\'', html)
# print(match.group(1), time.time() - t)
