from parsl.config import Config
from parsl.executors.threads import ThreadPoolExecutor

def local_threading(threads=int):
    return Config(
        executors=[
            ThreadPoolExecutor(
                max_threads=threads,
                label='local_threads'
            )
        ]
    )