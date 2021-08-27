import json
from datetime import datetime


class ValidationMixin:
    def _validate_json(self, json_string):
        try:
            js = json.loads(json_string)
            if not isinstance(js, dict):
                raise ValueError()
            return js
        except:
            return False


    def _validate_integer(self, value):
        return isinstance(value, int) or (isinstance(value, str) and value.isnumeric())


    def _validate_string(self, string):
        try:
            return string.upper()
        except:
            return False


    def _validate_date_format(self, datestring):
        try:
            datetime.strptime(datestring, "%Y-%m-%d")
            return True
        except:
            return False


    def _raise_json_error(self, var_name):
        raise ValueError(f'Invalid argument: {var_name} ; invalid json string')


    def _raise_date_error(self, var_name, datestring):
        raise ValueError(f'Invalid argument: {var_name} ; incorrect format {datestring}')


    def _raise_type_error(self, var_name, expected_type, received_val):
        received_type = type(received_val).__name__
        raise ValueError(
            f'Invalid argument: {var_name} ; ' +
            f'expected {expected_type} ; ' +
            f'got {received_type}'
        )
