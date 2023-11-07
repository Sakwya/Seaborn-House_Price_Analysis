import os
import pymysql
import tqdm
import threading


class Connection:
    def __init__(self):
        self.__db__ = pymysql.connect(
            host="localhost",
            user="root",
            password="root",
            db="1",
            charset="utf8",
            port=3306
        )

    def execute(self, query: str):
        cursor = self.__db__.cursor()
        cursor.execute(query)
        self.__db__.commit()
        cursor.close()

    def executeX(self, query_list: list):
        cursor = self.__db__.cursor()
        processbar = tqdm.tqdm(total=len(query_list))
        processbar.set_description("Executing")
        for query in query_list:
            cursor.execute(query)
            processbar.update()
        processbar.close()
        self.__db__.commit()
        cursor.close()

    def get_db(self):
        return self.__db__

    def __del__(self):
        self.__db__.close()


if __name__ == "__main__":
    db = Connection()
    with open('1.sql', 'r+', encoding='utf-8') as f:
        sqls = f.readlines()
    db.executeX(sqls)
