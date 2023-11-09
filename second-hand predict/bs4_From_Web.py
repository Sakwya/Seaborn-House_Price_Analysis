import urllib.request as request
from bs4 import BeautifulSoup
import re
import concurrent.futures
import multiprocessing

def get_info(url):
    add_list = []
    url = url
    inner_url = 'https://sh.lianjia.com/'
    req = request.urlopen(url)  # url为你想获取的页面的url
    index = req.read()  # index为你获取的整个页面
    soup = BeautifulSoup(index, 'html.parser')

    tag = soup.find_all("h1", class_="main")[0]
    add_list.append(tag.get_text())

    try:
        tag = soup.find_all("div", class_="communityName")[0]
        add_list.append(tag.find("a").get_text())
        inner_url = inner_url + tag.find("a").get("href")
    except IndexError:
        tag = None

    tag = soup.find_all("div", class_="areaName")[0]
    add_list.append(tag.find("a").get_text())

    tag = soup.find_all("div", class_="room")[0]
    add_list.append(tag.find("div").get_text())

    tag = soup.find_all("div", class_="area")[0]
    add_list.append(tag.find("div").get_text())

    tag = soup.find_all("div", class_="type")[0]
    add_list.append(tag.find("div").get_text())

    tag = soup.find_all("div", class_="type")[0]
    add_list.append(tag.find("div", class_="subInfo").get_text())

    tag = soup.find_all("div", class_="room")[0]
    add_list.append(tag.find("div", class_="subInfo").get_text())

    tag = soup.find_all("div", class_="area")[0]
    result = re.match(r'(.*)/(.*)', tag.find("div", class_="subInfo noHidden").get_text())
    if result:
        a = result.group(1)
        b = result.group(2)
    add_list.append(a)
    add_list.append(b)

    tag = soup.find_all("div", class_="base")[0]
    pslx = ''
    tag2 = tag.find_all("li")
    for tag in tag2:
        if tag.find("span").text == '别墅类型':
            pslx = tag.text.replace('别墅类型', '')
            break
    add_list.append(pslx)

    tag = soup.find_all("div", class_="tags clear")[0]
    roomtag = ''
    tag2 = tag.find_all("a")
    for tag in tag2:
        roomtag = roomtag+(tag.get_text().replace('\n', '').replace(' ', '')+' ')
    add_list.append(roomtag.rstrip())

    tag = soup.find_all("div", class_="price")[0]
    add_list.append(tag.find("span", class_="total").get_text())
    add_list.append(tag.find("span", class_="unitPriceValue").get_text())

    tag = soup.find_all("a", class_="supplement")[0]
    add_list.append(tag.get_text())

    tag = soup.find_all("div", class_="base")[0]
    elevator = ''
    tag2 = tag.find_all("li")
    for tag in tag2:
        if tag.find("span").text == '配备电梯':
            elevator = tag.text.replace('配备电梯', '')
            break
    add_list.append(elevator)

    inner_req = request.urlopen(inner_url)  # url为你想获取的页面的url
    inner_index = inner_req.read()  # index为你获取的整个页面
    inner_soup = BeautifulSoup(inner_index, 'html.parser')

    tag = inner_soup.find_all("span", class_="xiaoquInfoContent")[1].get_text()
    add_list.append(inner_soup.find_all("span", class_="xiaoquInfoContent")[1].get_text())
    add_list.append(inner_soup.find_all("span", class_="xiaoquInfoContent")[2].get_text())


    if inner_soup.find_all("span", class_="xiaoquInfoContent")[3].get_text()[-1] == '%':
        add_list.append(inner_soup.find_all("span", class_="xiaoquInfoContent")[3].get_text())
    else:
        add_list.append('')

    add_list.append(inner_soup.find_all("span", class_="xiaoquInfoContent outer")[0].get_text())
    print(add_list)
    return add_list


for i in range(100):
    get_info('https://sh.lianjia.com/ershoufang/107108310885.html')
