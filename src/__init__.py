import os, rq
from flask import Flask
from config import get_config
from redis import Redis


def create_app(env = os.environ['FLASK_ENV']):
    app = Flask(__name__)
    app.config.from_object(get_config(env))
    app.redis = Redis.from_url(app.config['REDIS_URL'])
    app.training_queue = rq.Queue(app.config['REDIS_TRAINING_QUEUE'], connection=app.redis)
    app.prediction_queue = rq.Queue(app.config['REDIS_PREDICTION_QUEUE'], connection=app.redis)
    app.fetch_queue = rq.Queue(app.config['REDIS_FETCH_QUEUE'], connection=app.redis)
    app.frontend_url = app.config['FRONTEND_URL']
    register_blueprints(app)
    return app


def get_config_filename(app, env):
    root = os.path.dirname(app.root_path)
    return f'{root}/config/{env}.cfg'


def register_blueprints(app):
    from src.utilities import add_error_handling, utilities_blueprint
    from src.ml import ml_blueprint

    add_error_handling(app)
    app.register_blueprint(utilities_blueprint)
    app.register_blueprint(ml_blueprint)
