import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = os.environ['TENSORFLOW_LOG_LEVEL']


class UrlBuilder:
    def build_frontend_url(env):
        host = os.environ['FRONTEND_HOST']
        if env == 'production':
            return f'https://{host}'
        else:
            return f'http://{host}:3000'


class Config:
    SECRET_KEY = 'bad_secret_key'
    DEBUG = True
    WORKER_TYPES = ['training', 'prediction', 'retrieval']
    REDIS_URL = os.environ['REDIS_URL'] or 'redis://'
    REDIS_TRAINING_QUEUE = os.environ['REDIS_TRAINING_QUEUE'] or 'training-queue'
    REDIS_PREDICTION_QUEUE = os.environ['REDIS_PREDICTION_QUEUE'] or 'prediction-queue'
    REDIS_RETRIEVAL_QUEUE = os.environ['REDIS_RETRIEVAL_QUEUE'] or 'retrieval-queue'
    FRONTEND_URL = UrlBuilder.build_frontend_url('development')


class TestConfig(Config):
    TESTING = True


class ProdConfig(Config):
    DEBUG = False
    TESTING = False
    FRONTEND_URL = UrlBuilder.build_frontend_url('production')


def get_config(env='development'):
    if env == 'production':
        return ProdConfig
    elif env == 'test':
        return TestConfig
    else:
        return Config
