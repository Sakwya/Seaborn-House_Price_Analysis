import os.path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from hashlib import md5
from bs4 import BeautifulSoup

# 创建一个重试机制
retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])

# 创建一个session并将重试机制应用到这个session上
session = requests.Session()
adapter = HTTPAdapter(max_retries=retries)
session.mount('http://', adapter)
session.mount('https://', adapter)


def encode(string: str, suffix: str = ".html"):
    """
    对字符串进行加密
    :param string: 要加密的字符串
    :param suffix： 自动添加的后缀
    :return: 加密后的字符串
    """
    return md5(string.encode("utf-8")).hexdigest() + suffix


def get_html(url: str, **kwargs):
    """
    获取网页的html
    :param url: 网址
    :param kwargs: 请求参数
    :return: 网页的html
    """
    try:
        return session.get(url, **kwargs).text
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e} on {url}")


def request(url: str, cache: bool = True, save: bool = True, file_path: str = "other", selector: str = None, **kwargs):
    """
    请求网页
    :param url: 网址
    :param cache: 是否使用缓存
    :param save: 是否保存
    :param file_path: 保存的路径
    :param selector: 网页内容的css选择器，用于选择性保存内容
    :param kwargs: 请求参数
    :return: 网页的html
    """
    dir_path = os.path.join('cache', file_path)
    if cache or save:
        file_path = os.path.join('cache', file_path, encode(url))
        if cache and os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                html = f.read()
        else:
            html = get_html(url, **kwargs)
    else:
        html = get_html(url, **kwargs)
    if selector is not None:
        soup = BeautifulSoup(html, "html.parser")
        items = soup.select(selector)
        BSoup = BeautifulSoup('', "html.parser")
        for item in items:
            BSoup.append(item)
        html = BSoup.prettify()
    if save:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Create {dir_path}")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(html)
    return html
