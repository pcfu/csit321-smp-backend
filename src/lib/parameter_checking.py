from flask import request
from werkzeug.exceptions import UnprocessableEntity


def require_params(*params):
    def outer_wrapper(fn):
      def inner_wrapper():
          specified_params = request.json
          for param in params:
              if param not in specified_params:
                  raise UnprocessableEntity(f'missing parameter "{param}"')

          return fn(**{k: v for k, v in specified_params.items() if k in params })

      inner_wrapper.__name__ = fn.__name__
      return inner_wrapper
    return outer_wrapper
