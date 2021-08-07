import time
from rq import get_current_job

class ExampleTask:
    def run(self, delay=1):
        job = get_current_job()

        print('Starting task...')
        time.sleep(delay)
        print("===========================================")
        print(f'JOB #{job.get_id()} - DO SOMETHING HERE')
        print("===========================================")
