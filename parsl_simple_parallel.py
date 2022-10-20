import parsl
from parsl import python_app
from parsl.config import Config
from parsl.executors.threads import ThreadPoolExecutor
import itertools
import random
import time
import timeit
from utils import timer_wrapper

local_threads = Config(
    executors=[
        ThreadPoolExecutor(
            max_threads=6,
            label='local_threads'
        )
    ]
)

parsl.clear()
parsl.load(local_threads)

@python_app
def multiply(index, a=None, b=None):
    time.sleep(random.randint(0, 3))
    return a * b

test_vars = [i for i in range(1, 10)]
params = itertools.product(test_vars, test_vars)

print(multiply.__name__)

i=0
outputs = list()
for param in params:
    i += 1
    outputs.append(multiply(i, *param))

for output in outputs:
    print(output.result())