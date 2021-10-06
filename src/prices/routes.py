from flask import jsonify
from . import prices_blueprint
from src.lib.parameter_checking import require_params
from src.lib.jobs import job_enqueuing as jq


@prices_blueprint.route('/prices', methods=['GET'])
@require_params('symbols', 'days')
def prices(symbols, days):
    res = jq.enqueue_price_update_job(symbols, days)
    return jsonify(jq.enqueue_result('price retrieval', res))
