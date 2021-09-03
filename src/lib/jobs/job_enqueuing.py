from datetime import datetime
from flask import current_app
from . import ModelTrainingJob


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


def get_job_id(job_type, training_id):
    ts = datetime.strftime(datetime.now(), DT_FMT)
    return f'{job_type.upper()}_TRAIN{training_id}_{ts}'


def enqueue_report(job_type, results):
    add_s = 's' if isinstance(results, list) else ''
    msg = f'{job_type.capitalize()} job{add_s} request processed'
    if has_error(results):
        return enqueue_partial(msg, results)
    else:
        return enqueue_success(msg, results)


def has_error(results):
    if isinstance(results, list):
        return any(res['status'] == 'error' for res in results)
    else:
        return results['status'] == 'error'


def enqueue_success(message, results):
    return { 'status': 'ok', 'message': message, 'results': results }


def enqueue_partial(message, results):
    message += f' with one or more errors'
    return { 'status': 'partial', 'message': message, 'results': results }
