from flask import make_response, jsonify

def register_utility_routes(app):
    @app.route("/", methods=['GET'])
    def root():
        return jsonify({ 'status': 'ok', 'message': 'Hello, stranger!' })

    @app.errorhandler(404)
    def page_not_found(e):
        content = { 'status': 'error', 'code': e.code, 'msg': e.description }
        return make_response(jsonify(content), 404)
