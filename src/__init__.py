import os
from flask import Flask

def create_app(env = os.environ['FLASK_ENV']):
    app = Flask(__name__)
    app.config.from_pyfile(get_config_filename(app, env))
    register_blueprints(app)
    return app


def get_config_filename(app, env):
    root = os.path.dirname(app.root_path)
    return f'{root}/config/{env}.cfg'


def register_blueprints(app):
    from src.utilities import add_error_handling, utilities_blueprint

    add_error_handling(app)
    app.register_blueprint(utilities_blueprint)
