import queue
import tqdm
import time
import multiprocessing
import threading
from lxml import etree
from bs4 import BeautifulSoup
import os
import re
import json

import spider
from spider.script import *


def get_neighbor_nos() -> list:
    neighbor_nos = []
    for district_dir in os.listdir("./cache/house_info/"):
        for file in os.listdir(f"./cache/house_info/{district_dir}"):
            with open(f"./cache/house_info/{district_dir}/{file}", 'r', encoding='utf-8') as f:
                house_info = json.loads(f.read().replace('\'', '\"'))
            if house_info['neighbor_no'] not in neighbor_nos:
                neighbor_nos.append(house_info['neighbor_no'])
    return neighbor_nos


def get_neighbor(neighbor_no: str):
    html = spider.request(f"{root_site.replace('ershoufang', 'xiaoqu')}{neighbor_no}",
                          file_path="neighbor_raw", filename=neighbor_no)
    try:
        position = re.search(r'resblockPosition:\s*\'([0-9.]+,[0-9.]+)\'', html).group(1).split(',')
    except AttributeError:
        html = spider.request(f"{root_site.replace('ershoufang', 'xiaoqu')}{neighbor_no}",
                              file_path="neighbor_raw", filename=neighbor_no, cache=False)
        print(f"\033[93mRecapture {root_site.replace('ershoufang', 'xiaoqu')}{neighbor_no}\033[0m")
        position = re.search(r'resblockPosition:\s*\'([0-9.]+,[0-9.]+)\'', html).group(1).split(',')
    soup = BeautifulSoup(html, 'lxml')
    items = [pair.text.split() for pair in soup.select("div.xiaoquInfoItem")]
    neighborInfo = {
        'longitude': position[0],
        'latitude': position[1],
    }
    try:
        with open(f'./cache/neighbor_info/{neighbor_no}.txt', 'w', encoding='utf-8') as f:
            f.write(str(neighborInfo).replace("\'", "\""))
    except FileNotFoundError:
        os.makedirs('./cache/neighbor_info')
        with open(f'./cache/neighbor_info/{neighbor_no}.txt', 'w', encoding='utf-8') as f:
            f.write(str(neighborInfo).replace("\'", "\""))


def process_get_neighbor(process_no: int, neighbor_no, process_queue):
    thread_queue = queue.Queue()
    thread_list = []
    for thread_no in range(MAX_THREAD):
        child_thread = threading.Thread(
            name=f"Thread-{thread_no}", target=thread_get_neighbor,
            args=(process_no, thread_no, neighbor_no, thread_queue))
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


def thread_get_neighbor(process_no: int, thread_no: int, neighbor_no, queue_list):
    i = process_no + MAX_PROCESS * thread_no
    max_iter = len(neighbor_no)
    while i < max_iter:
        get_neighbor(neighbor_no[i])
        queue_list.put(i)
        i += MAX_PROCESS * MAX_THREAD
    print(f"\rThread{process_no}-{thread_no} finished")


def run():
    t = time.time()
    pool = multiprocessing.Pool(processes=MAX_PROCESS)

    target = get_neighbor_nos()
    manager = multiprocessing.Manager()
    progress_queue = manager.Queue()
    pbar = tqdm.tqdm(total=len(target))

    process_list = []
    for i in range(MAX_PROCESS):
        process = pool.apply_async(process_get_neighbor, args=(i, target, progress_queue))
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
    get_neighbor("5011000014672")
# html = spider.request("https://sh.lianjia.com/ershoufang/107107606471.html")
# t = time.time()
# match = re.search(r'resblockPosition:\s*\'([0-9.]+,[0-9.]+)\'', html)
# print(match.group(1), time.time() - t)
# t = time.time()
# html = spider.request("https://sh.lianjia.com/ershoufang/107107606471.html", xpath="//script")
# match = re.search(r'resblockPosition:\s*\'([0-9.]+,[0-9.]+)\'', html)
# print(match.group(1), time.time() - t)
