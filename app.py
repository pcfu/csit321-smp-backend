import click, redis
from rq import Connection, Worker
from src import create_app

app = create_app()


@app.cli.command('start_worker')
@click.argument('worker_type', default='training')
def start_worker(worker_type):
    if worker_type in app.config['WORKER_TYPES']:
        with Connection(app.redis):
            queue = getattr(app, f'{worker_type}_queue')
            worker = Worker(queue.name)
            worker.work()
    else:
        print(f'Unknown worker type: {worker_type}')
