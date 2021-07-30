from flask import make_response, jsonify
from flask import Blueprint

utilities_blueprint = Blueprint('utilities', __name__)

from . import routes


def add_error_handling(app):
    @app.errorhandler(404)
    def page_not_found(e):
        content = { 'status': 'error', 'code': e.code, 'msg': e.description }
        return make_response(jsonify(content), 404)
