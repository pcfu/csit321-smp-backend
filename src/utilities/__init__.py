from flask import jsonify

def register_utility_routes(app):
    @app.route("/", methods=['GET'])
    def root():
        return jsonify({ 'status': 'ok', 'message': 'Hello, stranger!' })

    @app.errorhandler(404)
    def page_not_found(e):
        return jsonify({ 'status': 'error', 'code': e.code, 'msg': e.description })
