from flask import Flask

def create_app(config_filename=None):
    app = Flask(__name__)
    register_blueprints(app)
    return app


def register_blueprints(app):
    from src.utilities import add_error_handling, utilities_blueprint

    add_error_handling(app)
    app.register_blueprint(utilities_blueprint)
