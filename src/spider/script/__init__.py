# root_site = "https://sh.lianjia.com/ershoufang/"
root_site = "https://sh.ke.com/ershoufang/"
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

select = "co12dp1sf1ba0ea500"

MAX_THREAD = 6
MAX_PROCESS = 5

from .get_catalogue import run as get_catalogue
from .get_house_raw import run as get_house_raw
from .get_house_info import run as get_house_info
from .get_neighbor import run as get_neighbor
from .get_info import run as get_info
