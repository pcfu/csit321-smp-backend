from flask import make_response, jsonify
from flask import Blueprint
from werkzeug.exceptions import HTTPException


utilities_blueprint = Blueprint('utilities', __name__)

from . import routes


def add_error_handling(app):
    @app.errorhandler(Exception)
    def handle_error(e):
        if isinstance(e, HTTPException):
            return e

        content = { 'status': 'error', 'code': 500, 'message': str(e) }
        return make_response(jsonify(content), 500)


    @app.errorhandler(404)
    @app.errorhandler(422)
    def page_not_found(e):
        content = { 'status': 'error', 'code': e.code, 'message': e.description }
        return make_response(jsonify(content), e.code)
