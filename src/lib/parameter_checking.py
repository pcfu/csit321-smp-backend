from flask import request
from werkzeug.exceptions import UnprocessableEntity


def require_params(*params):
    def outer_wrapper(fn):
      def inner_wrapper():
          for param in params:
              if param not in request.args:
                  raise UnprocessableEntity(f'missing parameter "{param}"')

          return fn()

      return inner_wrapper
    return outer_wrapper
