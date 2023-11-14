import queue
import tqdm
import time
import multiprocessing
import threading
import spider
from bs4 import BeautifulSoup
import os
from spider.script import *


def get_catalogues() -> list:
    catalogues = []
    for i in range(len(district)):
        for file in os.listdir(f"./cache/sellListContent/{district[i]}"):
            catalogues.append(f"./cache/sellListContent/{district[i]}/{file}")
    return catalogues


def get_house_raw(catalogue: str):
    file_path = os.path.join("house_raw", catalogue.split('/')[-2])
    with open(catalogue, encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    house_hrefs = []
    for house_href in soup.find_all("a"):
        house_hrefs.append(house_href.get("href"))
    for house_href in house_hrefs:
        spider.request(house_href, file_path=file_path, cache=True, save=True, selector="body")


def process_get_house_raw(process_no: int, catalogues, process_queue):
    thread_queue = queue.Queue()
    thread_list = []
    for thread_no in range(MAX_THREAD):
        child_thread = threading.Thread(
            name=f"Thread-{thread_no}", target=thread_get_house_raw,
            args=(process_no, thread_no, catalogues, thread_queue))
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


def thread_get_house_raw(process_no: int, thread_no: int, catalogues, queue):
    i = process_no + MAX_PROCESS * thread_no
    max_iter = len(catalogues)
    while i < max_iter:
        get_house_raw(catalogues[i])
        queue.put(i)
        i += MAX_PROCESS * MAX_THREAD
    print(f"\rThread{process_no}-{thread_no} finished")


def run():
    t = time.time()
    pool = multiprocessing.Pool(processes=MAX_PROCESS)

    target = get_catalogues()
    manager = multiprocessing.Manager()
    progress_queue = manager.Queue()
    pbar = tqdm.tqdm(total=len(target))

    process_list = []
    for i in range(MAX_PROCESS):
        process = pool.apply_async(process_get_house_raw, args=(i, target, progress_queue))
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