from flask import current_app, request, jsonify
from . import ml_blueprint
from src.lib.tasks import ExampleTask


@ml_blueprint.route('/ml/train', methods=['POST'])
def train():
    # placeholder logic
    task = ExampleTask()
    delay = int(request.args.get('delay', 1))
    job = current_app.task_queue.enqueue(task.run, delay)
    return jsonify({ 'status': 'ok', 'message': f'Job #{job.get_id()} queued' })


@ml_blueprint.route('/ml/predict', methods=['GET'])
def predict():
    # placeholder logic

    return jsonify({ 'status': 'ok' })
