from flask import jsonify
from . import ml_blueprint
from src.lib.parameter_checking import require_params
from src.lib.jobs.job_enqueuing import enqueue_training_job


@ml_blueprint.route('/ml/model_training', methods=['POST'])
@require_params('training_list', 'model_params', 'data_range')
def train(training_list, model_params, data_range):
    results = []
    for trng in training_list:
        res = enqueue_training_job(list(trng.values()), model_params, data_range)
        results.append(res)
    return jsonify({ 'status': 'ok', 'message': 'Job requests processed', 'results': results })


@ml_blueprint.route('/ml/predict', methods=['GET'])
def predict():
    # placeholder
    return jsonify({ 'status': 'ok', 'message': 'placeholder' })
