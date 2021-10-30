from datetime import datetime
from flask import current_app
from . import PriceUpdateJob, TisUpdateJob, \
              ModelTrainingJob, PricePredictionJob, RecommendationJob


TIMEOUT = 1200  # 1200 secs == 20 mins
DT_FMT  = "%Y-%m-%dT%H:%M:%S.%f"


def enqueue_price_update_job(symbols, days):
    result = {}
    timestamp = int(datetime.timestamp(datetime.now()))
    jid = f"price_update_{timestamp}"
    args = [symbols, days]
    _enqueue_job(current_app.retrieval_queue, PriceUpdateJob, jid, args, result)
    return result


def enqueue_tis_update_job(stock_id, prices, n_last_data):
    result = {}
    timestamp = int(datetime.timestamp(datetime.now()))
    jid = f"tis_update_{timestamp}"
    args = [stock_id, prices, n_last_data]
    _enqueue_job(current_app.retrieval_queue, TisUpdateJob, jid, args, result)
    return result


def enqueue_training_job(tid, sid, symbol, model_name, model_class, model_params):
    result = { "training_id": tid, "stock_id": sid }
    jid = _get_job_id('training', tid)
    args = [tid, sid, symbol, model_name, model_class, model_params]
    _enqueue_job(current_app.training_queue, ModelTrainingJob, jid, args, result)
    return result


def enqueue_prediction_job(training_id, stock_id):
    result = { 'training_id': training_id }
    jid = _get_job_id('prediction', training_id)
    args = [training_id, stock_id]
    _enqueue_job(current_app.prediction_queue, PricePredictionJob, jid, args, result)
    return result


def enqueue_recommendation_job(stock_id, model_path, last_date):
    result = { 'stock_id': stock_id }
    jid = _get_job_id('recommendation', stock_id)
    args = [stock_id, model_path, last_date]
    _enqueue_job(current_app.prediction_queue, RecommendationJob, jid, args, result)
    return result


def enqueue_result(job_type, results):
    add_s = 's' if isinstance(results, list) else ''
    base_msg = f'{job_type.title()} job{add_s} request processed'
    if _is_success(results):
        return _enqueue_success(base_msg, results)
    elif _is_fail(results):
        return _enqueue_fail(base_msg, results)
    else:
        return _enqueue_partial(base_msg, results)


def _enqueue_job(queue, job_cls, job_id, job_args, base_result):
    try:
        task = job_cls(*job_args).run
        job = queue.enqueue(task, job_id=job_id, job_timeout=TIMEOUT)
        base_result.update({ 'status': 'ok', 'job_id': job.get_id() })
    except Exception as e:
        base_result.update({ 'status': 'error', 'error_message': str(e) })


def _get_job_id(job_type, training_id):
    ts = datetime.strftime(datetime.now(), DT_FMT)
    return f'{job_type.upper()}_{training_id}_{ts}'


def _is_success(results):
    if isinstance(results, list):
        return all(res['status'] == 'ok' for res in results)
    else:
        return results['status'] == 'ok'


def _is_fail(results):
    if isinstance(results, list):
        return all(res['status'] == 'error' for res in results)
    else:
        return results['status'] == 'error'


def _enqueue_success(message, results):
    return { 'status': 'success', 'message': message, 'results': results }


def _enqueue_fail(message, results):
    message += ' with errors for all jobs'
    return { 'status': 'failed', 'message': message, 'results': results }


def _enqueue_partial(message, results):
    message += ' with errors for some jobs'
    return { 'status': 'partial', 'message': message, 'results': results }
