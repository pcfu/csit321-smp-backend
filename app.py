import redis
from rq import Connection, Worker
from src import create_app

app = create_app()


@app.cli.command('start_worker')
def start_worker():
    with Connection(app.redis):
        worker = Worker(app.task_queue.name)
        worker.work()
