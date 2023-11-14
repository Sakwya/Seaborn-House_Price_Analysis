from .get_catalogue import run as get_catalogue
from .get_house_raw import run as get_house_raw

root_site = "https://sh.lianjia.com/ershoufang/"
district = [
    "pudong",
    "minhang",
    "baoshan",
    "xuhui",
    "putuo",
    "yangpu",
    "changning",
    "songjiang",
    "jiading",
    "huangpu",
    "jingan",
    "hongkou",
    "qingpu",
    "fengxian",
    "jinshan",
    "chongming"
]
MAX_THREAD = 5
MAX_PROCESS = 5
