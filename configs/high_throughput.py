from parsl.config import Config
from parsl.executors import HighThroughputExecutor  
from parsl.providers import LocalProvider
from parsl.channels import LocalChannel

def local():
    return Config(
        executors=[HighThroughputExecutor(
            label = 'local_throughput',
            worker_debug=True,
            cores_per_worker=1,
            provider=LocalProvider(
                channel=LocalChannel(),
                init_blocks=1,
                max_blocks=1
            )
        )],
        strategy=None,
    )