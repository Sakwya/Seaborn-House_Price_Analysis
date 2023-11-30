import sqlite3

import flask
from flask import current_app, g


# 1.创建数据库连接
def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(
            current_app.config['DATABASE'],
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        g.db.row_factory = sqlite3.Row
    return g.db


# 2. 关闭数据库连接
def close_db(self):
    db = g.pop('db', None)
    if db is not None:
        db.close()


def init_app(app: flask.Flask):
    db = sqlite3.connect(
        current_app.config['DATABASE'],
        detect_types=sqlite3.PARSE_DECLTYPES
    )
    cursor = db.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()

    app.teardown_appcontext(close_db)
