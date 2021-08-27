import os

class Config:
    SECRET_KEY = 'bad_secret_key'
    DEBUG = True
    WORKER_TYPES = ['training', 'prediction', 'fetch']
    REDIS_URL = os.environ['REDIS_URL'] or 'redis://'
    REDIS_TRAINING_QUEUE = os.environ['REDIS_TRAINING_QUEUE'] or 'training-queue'
    REDIS_PREDICTION_QUEUE = os.environ['REDIS_PREDICTION_QUEUE'] or 'prediction-queue'
    REDIS_FETCH_QUEUE = os.environ['REDIS_FETCH_QUEUE'] or 'fetch-queue'


class TestConfig(Config):
    TESTING = True


def get_config(env='development'):
    if env == 'test':
        return TestConfig
    else:
        return Config
