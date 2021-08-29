from flask import current_app
from . import ModelTrainingJob


TIMEOUT = 1200  # 1200 secs == 20 mins


def enqueue_training_job(training_id, config_id, stock_id, data_range):
    jid = f'modal_config_{config_id}_stock_{stock_id}'
    result = { "training_id": training_id, "config_id": config_id, "stock_id": stock_id }

    try:
        training = ModelTrainingJob(training_id, config_id, stock_id, *data_range)
        job = current_app.training_queue.enqueue(
            training.run, job_id=jid, job_timeout=TIMEOUT
        )
        result.update({ 'status': 'ok', 'job_id': job.get_id() })
    except Exception as e:
        result.update({ 'status': 'error', 'error_message': str(e) })

    return result
