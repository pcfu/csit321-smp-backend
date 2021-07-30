from flask import jsonify
from . import utilities_blueprint


@utilities_blueprint.route("/", methods=['GET'])
def root():
    return jsonify({ 'status': 'ok', 'message': 'Hello, stranger!' })
