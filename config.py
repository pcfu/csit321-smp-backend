import os

class Config:
    SECRET_KEY = 'bad_secret_key'
    DEBUG = True
    REDIS_URL = os.environ['REDIS_URL'] or 'redis://'
    REDIS_QUEUE = os.environ['REDIS_QUEUE'] or 'smp-backend-queue'


class TestConfig(Config):
    TESTING = True


def get_config(env='development'):
    if env == 'test':
        return TestConfig
    else:
        return Config
