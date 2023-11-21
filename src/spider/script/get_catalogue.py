import queue
import tqdm
import time
import multiprocessing
import threading
import spider
from src.spider.script import *


def get_catalogue(index: int):
    district_no = index % 16
    page_no = int(index / 16) + 1
    spider.request(f"{root_site}{district[district_no]}/pg{page_no}{select})",
                   file_path="sellListContent/" + district[district_no],
                   cache=False, save=True, suffix=".txt", filename=str(page_no),use_md5=False,
                   xpath="/html/body/div[1]/div[4]/div[1]/div[4]/ul[@class = \"sellListContent\"]"
                         "//li[@class = \"clear\"]/a/@href")


def process_get_catalogue(process_no: int, max_iter, process_queue):
    thread_queue = queue.Queue()
    thread_list = []
    for thread_no in range(MAX_THREAD):
        child_thread = threading.Thread(
            name=f"Thread-{thread_no}", target=thread_get_catalogue,
            args=(process_no, thread_no, max_iter, thread_queue))
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


def thread_get_catalogue(process_no: int, thread_no: int, max_iter, queue):
    i = process_no + MAX_PROCESS * thread_no
    while i <= max_iter:
        get_catalogue(i)
        queue.put(i)
        i += MAX_PROCESS * MAX_THREAD
    print(f"\rThread{process_no}-{thread_no} finished")


def run():
    t = time.time()
    pool = multiprocessing.Pool(processes=MAX_PROCESS)

    target = 16 * 100
    manager = multiprocessing.Manager()
    progress_queue = manager.Queue()
    pbar = tqdm.tqdm(total=target)

    process_list = []
    for i in range(MAX_PROCESS):
        process = pool.apply_async(process_get_catalogue, args=(i, target, progress_queue))
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
    get_catalogue(1)
