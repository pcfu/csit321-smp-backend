from flask import jsonify
from . import ml_blueprint
from src.lib.parameter_checking import require_params
from src.lib.jobs import job_enqueuing as jq


@ml_blueprint.route('/ml/model_training', methods=['POST'])
@require_params('training_list', 'model')
def train(training_list, model):
    results = []
    for trng in training_list:
        res = jq.enqueue_training_job(*trng.values(), *model.values())
        results.append(res)
    return jsonify(jq.enqueue_result('training', results))


@ml_blueprint.route('/ml/prediction', methods=['GET'])
@require_params('training_id', 'stock_id')
def predict(training_id, stock_id):
    res = jq.enqueue_prediction_job(training_id, stock_id)
    return jsonify(jq.enqueue_result('prediction', res))
