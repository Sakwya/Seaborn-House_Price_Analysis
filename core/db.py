import pymysql


class Connect:
    def __init__(self):
        self.db = pymysql.connect(host="localhost",
                                  port=3306,
                                  user="root",
                                  password="root",
                                  database="houseData",
                                  charset="utf8mb4",

                                  )
        pass
