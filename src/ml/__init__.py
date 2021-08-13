from flask import Blueprint

ml_blueprint = Blueprint('ml', __name__)

from . import routes
