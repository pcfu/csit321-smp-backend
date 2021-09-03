from flask import jsonify
from . import ml_blueprint
from src.lib.parameter_checking import require_params
from src.lib.jobs.job_enqueuing import enqueue_training_job, enqueue_report


@ml_blueprint.route('/ml/model_training', methods=['POST'])
@require_params('training_list', 'model_params', 'data_range')
def train(training_list, model_params, data_range):
    results = []
    for trng in training_list:
        trng_id, stock_id = trng.values()
        res = enqueue_training_job(trng_id, stock_id, model_params, data_range)
        results.append(res)
    return jsonify(enqueue_report('training', results))


@ml_blueprint.route('/ml/predict', methods=['GET'])
def predict():
    # placeholder
    return jsonify({ 'status': 'ok', 'message': 'placeholder' })
