from flask import current_app, request, jsonify
from . import ml_blueprint
from src.lib.tasks import ExampleTask
import pickle


@ml_blueprint.route('/ml/train', methods=['POST'])
def train():
    # placeholder logic
    task = ExampleTask()
    task.message = request.args.get('message', 'sample text')

    serialized = pickle.dumps(task)
    current_app.redis.set('pickled_obj', serialized)
    deserialized = pickle.loads(current_app.redis.get('pickled_obj'))
    return jsonify({ 'status': 'ok', 'message': deserialized.message })


@ml_blueprint.route('/ml/predict', methods=['GET'])
def predict():
    # placeholder logic
    task = ExampleTask()
    delay = int(request.args.get('delay', 1))
    job = current_app.task_queue.enqueue(task.run, delay)
    return jsonify({ 'status': 'ok', 'message': f'Job #{job.get_id()} queued' })
