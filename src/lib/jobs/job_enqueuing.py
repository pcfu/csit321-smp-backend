from flask import current_app
from . import ModelTrainingJob


TIMEOUT = 1200  # 1200 secs == 20 mins


def enqueue_training_job(ids, model_params, data_range):
    result = { "training_id": ids[0], "config_id": ids[1], "stock_id": ids[2] }
    jid = f'model_config_{result.get("config_id")}_stock_{result.get("stock_id")}'

    try:
        training = ModelTrainingJob(ids, model_params, data_range)
        job = current_app.training_queue.enqueue(
            training.run, job_id=jid, job_timeout=TIMEOUT
        )
        result.update({ 'status': 'ok', 'job_id': job.get_id() })
    except Exception as e:
        result.update({ 'status': 'error', 'error_message': str(e) })

    return result
