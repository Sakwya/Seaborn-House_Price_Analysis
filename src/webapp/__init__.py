from flask import Flask, render_template, g, session, redirect, url_for, current_app
import os


def create_app():
    app = Flask(__name__,
                static_url_path="/static",  # 访问静态资源的url前缀，默认值是static
                static_folder="static",  # 设置静态文件的目录，默认值是static
                template_folder="templates"  # 设置模板文件的目录，默认值是templates
                )
    with app.app_context():
        app.config.from_mapping(
            SECRET_KEY='dev',
            DATABASE='Seaborn.sqlite'
        )

    # @app.before_request
    # def before_request():
    #     with current_app.app_context():
    #         if hasattr(current_app, 'p'):
    #             print(f"production in")
    #         else:
    #             print(f"production not in")
    #         current_app.p = p if p else Production(grammars, props)
    #
    # with app.app_context():
    #     app.p = p if p else Production(grammars, props)
    @app.route('/')
    def index():
        return render_template('index.html', title="基于Seaborn的房价可视化")

    # from .db import init_app
    # init_app(app)

    from .api import bp
    app.register_blueprint(bp)
    return app
