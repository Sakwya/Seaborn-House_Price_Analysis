import queue
import tqdm
import time
import multiprocessing
import threading
import spider
from bs4 import BeautifulSoup
import os
from spider.script import *

Task_Num = MAX_PROCESS * 2


def get_catalogues() -> list:
    catalogues = []
    for i in range(len(district)):
        for file in os.listdir(f"./cache/sellListContent/{district[i]}"):
            catalogues.append(f"./cache/sellListContent/{district[i]}/{file}")
        # for file in os.listdir(f"../cache/sellListContent/{district[i]}"):
        #     catalogues.append(f"../cache/sellListContent/{district[i]}/{file}")
    return catalogues


def get_house_raw(catalogue: str):
    file_path = os.path.join("house_raw", catalogue.split('/')[-2])
    with open(catalogue, encoding="utf-8") as f:
        house_hrefs = f.read().split('\n')
        house_hrefs.remove('')
    for house_href in house_hrefs:
        filename = house_href.split('/')[-1]
        if not os.path.exists(os.path.join('cache', file_path, filename)):
            if spider.request(house_href, file_path=file_path, cache=False, save=True,
                              filename=filename, suffix="", debug=False,
                              xpath="/html/body/div[@class = \"sellDetailPage\"]"
                                    "/div[4]/div[1]/div[2]/div[3]/div[3]/div[2]|"
                                    "/html/body/div[@class = \"sellDetailPage\"]"
                                    "/div[4]/div[1]/div[2]/div[4]/div[1]|"
                                    "/html/body/div[@class = \"sellDetailPage\"]"
                                    "/div[4]/div[1]/div[2]/div[4]/div[2]|"
                                    "/html/body/div[@class = \"sellDetailPage\"]"
                                    "/div[5]//div[@class = \"introContent\"]//ul") is None:
                print(house_href)
        else:
            pass


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


def thread_get_house_raw(task_no: int, thread_no: int, catalogues, queue):
    i = task_no + Task_Num * thread_no
    max_iter = len(catalogues)
    while i < max_iter:
        get_house_raw(catalogues[i])
        queue.put(i)
        i += Task_Num * MAX_THREAD
    print(f"\rThread{task_no}-{thread_no} finished")


def run():
    t = time.time()
    pool = multiprocessing.Pool(processes=MAX_PROCESS)

    target = get_catalogues()
    manager = multiprocessing.Manager()
    progress_queue = manager.Queue()
    pbar = tqdm.tqdm(total=len(target))

    process_list = []
    for i in range(Task_Num):
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


if __name__ == "__main__":
    get_house_raw(get_catalogues()[0])
