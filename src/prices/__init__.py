from flask import Blueprint

prices_blueprint = Blueprint('prices', __name__)

from . import routes
