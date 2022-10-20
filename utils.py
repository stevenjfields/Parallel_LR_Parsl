import timeit
import functools

def timer_wrapper(func):
    def __init__(self, func):
        self.func = func

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if "index" in kwargs.keys:
            start = timeit.default_timer()
            func(*args, **kwargs)
            stop = timeit.default_timer()
            print(f'Run {kwargs["index"]} finished in {stop-start} seconds.')
        else:
            func(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        wrapper(*args, **kwargs)