from flask import Flask
from src.utilities import register_utility_routes

def create_app(config_filename=None):
    app = Flask(__name__)
    register_routes(app)
    return app

def register_routes(app):
    register_utility_routes(app)
