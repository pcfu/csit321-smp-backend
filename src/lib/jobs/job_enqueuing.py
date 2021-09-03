from datetime import datetime
from flask import current_app
from . import ModelTrainingJob, PricePredictionJob


TIMEOUT = 1200  # 1200 secs == 20 mins
DT_FMT  = "%Y-%m-%dT%H:%M:%S.%f"


def enqueue_training_job(tid, sid, model_params, data_range):
    result = { "training_id": tid, "stock_id": sid }
    jid = get_job_id('training', tid)

    try:
        training = ModelTrainingJob(tid, sid, model_params, data_range)
        job = current_app.training_queue.enqueue(
            training.run, job_id=jid, job_timeout=TIMEOUT
        )
        result.update({ 'status': 'ok', 'job_id': job.get_id() })
    except Exception as e:
        result.update({ 'status': 'error', 'error_message': str(e) })

    return result


def enqueue_prediction_job(training_id, stock_id, data_range):
    result = { 'training_id': training_id }
    jid = get_job_id('prediction', training_id)

    try:
        prediction = PricePredictionJob(training_id, stock_id, data_range)
        job = current_app.prediction_queue.enqueue(
            prediction.run, job_id=jid, job_timeout=TIMEOUT
        )
        result.update({ 'status': 'ok', 'job_id': job.get_id() })
    except Exception as e:
        result.update({ 'status': 'error', 'error_message': str(e) })

    return result


def get_job_id(job_type, training_id):
    ts = datetime.strftime(datetime.now(), DT_FMT)
    return f'{job_type.upper()}_TRAIN{training_id}_{ts}'


def enqueue_result(job_type, results):
    add_s = 's' if isinstance(results, list) else ''
    base_msg = f'{job_type.capitalize()} job{add_s} request processed'
    if is_success(results):
        return enqueue_success(base_msg, results)
    elif is_fail(results):
        return enqueue_fail(base_msg, results)
    else:
        return enqueue_partial(base_msg, results)


def is_success(results):
    if isinstance(results, list):
        return all(res['status'] == 'ok' for res in results)
    else:
        return results['status'] == 'ok'


def is_fail(results):
    if isinstance(results, list):
        return all(res['status'] == 'error' for res in results)
    else:
        return results['status'] == 'error'


def enqueue_success(message, results):
    return { 'status': 'success', 'message': message, 'results': results }


def enqueue_fail(message, results):
    message += ' with errors for all jobs'
    return { 'status': 'failed', 'message': message, 'results': results }


def enqueue_partial(message, results):
    message += ' with errors for some jobs'
    return { 'status': 'partial', 'message': message, 'results': results }
