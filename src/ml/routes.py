from flask import current_app, request, jsonify
from . import ml_blueprint
from src.lib.parameter_checking import require_params
from src.lib.jobs import ModelTrainingJob


@ml_blueprint.route('/ml/model_training', methods=['POST'])
@require_params('config_id', 'stocks', 'data_range')
def train(config_id, stocks, data_range):
    results = []
    for stock_id in stocks:
        res = enqueue_training_job(config_id, stock_id, data_range)
        results.append(res)
    return jsonify({ 'status': 'ok', 'message': 'Job requests processed', 'results': results })


@ml_blueprint.route('/ml/predict', methods=['GET'])
def predict():
    # placeholder logic
    task = ModelTrainingJob()
    delay = int(request.args.get('delay', 1))
    job = current_app.prediction_queue.enqueue(task.run, delay)
    return jsonify({ 'status': 'ok', 'message': f'Job #{job.get_id()} queued' })


def enqueue_training_job(config_id, stock_id, data_range):
    jid = f'modal_config_{config_id}_stock_{stock_id}'
    result = { 'config_id': config_id, 'stock_id': stock_id }

    try:
        training = ModelTrainingJob(config_id, stock_id, *data_range)
        job = current_app.training_queue.enqueue(training.run, job_id=jid, job_timeout=1200)
        result.update({ 'status': 'ok', 'job_id': job.get_id() })
    except Exception as e:
        result.update({ 'status': 'error', 'error_message': str(e) })

    return result
