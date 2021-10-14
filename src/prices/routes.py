from flask import jsonify
from . import prices_blueprint
from src.lib.parameter_checking import require_params
from src.lib.jobs import job_enqueuing as jq


@prices_blueprint.route('/prices', methods=['GET'])
@require_params('symbols', 'days')
def prices(symbols, days):
    res = jq.enqueue_price_update_job(symbols, days)
    return jsonify(jq.enqueue_result('price retrieval', res))


@prices_blueprint.route('/tis', methods=['GET'])
@require_params('stock_id', 'prices', 'n_last_data')
def technical_indicators(stock_id, prices, n_last_data):
    res = jq.enqueue_tis_update_job(stock_id, prices, n_last_data)
    return jsonify(jq.enqueue_result('tis update', res))
