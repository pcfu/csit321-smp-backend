from flask import current_app, request, jsonify
from . import ml_blueprint
from src.lib.jobs import ModelTrainingJob
import pickle


@ml_blueprint.route('/ml/model_training', methods=['POST'])
def train():
    ### Example logic
    # serialized = pickle.dumps(job)
    # current_app.redis.set('pickled_obj', serialized)
    # deserialized = pickle.loads(current_app.redis.get('pickled_obj'))

    params = {}
    required_params = ['model', 'stocks']
    for param in required_params:
        if param not in request.args:
            return jsonify({ 'status': 'error', 'message': f'missing parameter "{param}"' })
        params[param] = request.args.get(param)

    for stock in stocks:
        training = ModelTrainingJob()
        job = current_app.training_queue.enqueue(
            training.run,
            model_config=params['model'],
            stock=stock
        )

    return jsonify({ 'status': 'ok', 'message': 'Model training jobs enqueued' })


@ml_blueprint.route('/ml/predict', methods=['GET'])
def predict():
    # placeholder logic
    task = ModelTrainingJob()
    delay = int(request.args.get('delay', 1))
    job = current_app.prediction_queue.enqueue(task.run, delay)
    return jsonify({ 'status': 'ok', 'message': f'Job #{job.get_id()} queued' })
