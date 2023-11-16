import os.path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from hashlib import md5
from bs4 import BeautifulSoup
from lxml import etree

# 创建一个重试机制
retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])

# 创建一个session并将重试机制应用到这个session上
session = requests.Session()
adapter = HTTPAdapter(max_retries=retries)
session.mount('http://', adapter)
session.mount('https://', adapter)

parser = etree.HTMLParser(encoding="utf-8")


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


def request(url: str, cache: bool = True, save: bool = True, file_path: str = "other",
            xpath: str = None, selector: str = None, suffix: str = ".html", **kwargs):
    """
    请求网页
    :param url: 网址
    :param cache: 是否使用缓存
    :param save: 是否保存
    :param file_path: 保存的路径
    :param xpath: 网页内容的xpath，用于选择性保存内容
    :param selector: 网页内容的css选择器，用于选择性保存内容
    :param suffix: 存储文件的后缀名
    :param kwargs: 请求参数
    :return: 网页的html
    """
    dir_path = os.path.join('cache', file_path)
    if cache or save:
        file_path = os.path.join('cache', file_path, encode(url, suffix=suffix))
        if cache and os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                html = f.read()
        else:
            html = get_html(url, **kwargs)
    else:
        html = get_html(url, **kwargs)

    if xpath is not None:
        if selector is not None:
            print(f"\033[91mCSS selector and xpath are both specified, using xpath.\033[0m")
        html = etree.HTML(html, parser)
        items = html.xpath(xpath)
        if len(items) == 0:
            print(f"\r\033[91mElement (Xpath: \033[93m{xpath}\033[91m) Was Not Found On\033[0m {url}.")
            # if os.path.exists(file_path):
            #     os.remove(file_path)
            return None
        html = ""
        for item in items:
            if isinstance(item, str):
                html = html + item + '\n'
            else:
                html = html + etree.tostring(item, encoding="utf-8").decode("utf-8") + '\n'

    elif selector is not None:
        soup = BeautifulSoup(html, "html.parser")
        items = soup.select(selector)
        BSoup = BeautifulSoup('', "html.parser")
        if len(items) == 0:
            print(f"\r\033[91mElement (CSS selector: \033[93m{selector}\033[91m) Was Not Found On\033[0m {url}.")
            # if os.path.exists(file_path):
            #     os.remove(file_path)
            return None
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
