import asyncio
import json
from itertools import islice
from functools import cache
from pathlib import Path
from typing import Any, Iterable, TypeVar

import yaml
from git.repo import Repo
from jinja2 import Environment, FileSystemLoader
from joblib import Memory
import elasticsearch

import logging
from rich.logging import RichHandler
# Set the logging level of the Elasticsearch client to 'WARNING' or higher to hide 'INFO' logs
logging.getLogger('elasticsearch').setLevel(logging.WARNING)

# If your logs are showing low-level HTTP connection details from urllib3, also set this to 'WARNING'
logging.getLogger('urllib3').setLevel(logging.WARNING)
FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO",
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

logger = logging.getLogger("rich")

T = TypeVar("T")


def flatten(l: Iterable[Iterable[T]]) -> list[T]:
    return [item for sublist in l for item in sublist]


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


def batched(iterable: Iterable[T], n: int) -> Iterable[tuple[T, ...]]:
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


@cache
def load_jsonl(path: Path) -> list:
    with path.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f.readlines()]


@cache
def get_project_root() -> Path:
    repo = Repo(".", search_parent_directories=True)
    path = repo.working_tree_dir

    assert path is not None, "You are not currently in a Git repo!"

    return Path(path)


@cache
def load_pipeline_config() -> dict[str, Any]:
    with (get_project_root() / "params.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_project_cache() -> Memory:
    return Memory(get_project_root() / ".joblib_cache", verbose=0)


@cache
def get_jinja_environment() -> Environment:
    return Environment(loader=FileSystemLoader(get_project_root() / "templates"))


async def gather_with_concurrency(n: int, *coros):
    semaphore = asyncio.Semaphore(n)

    async def sem_coro(coro):
        async with semaphore:
            return await coro

    return await asyncio.gather(*(sem_coro(c) for c in coros))


def limit_global_concurrency(n: int):
    def decorator(func):
        semaphore = asyncio.Semaphore(n)

        async def wrapper(*args, **kwargs):
            async def sem_coro(coro):
                async with semaphore:
                    return await coro

            return await sem_coro(func(*args, **kwargs))

        return wrapper

    return decorator


def cached(cache, keyfunc):
    def decorator(func):

        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                key = keyfunc(*args, **kwargs)
                if key in cache:
                    return cache[key]
                else:
                    result = await func(*args, **kwargs)
                    cache[key] = result
                    return result

            return async_wrapper
        else:
            def wrapper(*args, **kwargs):
                key = keyfunc(*args, **kwargs)
                if key in cache:
                    return cache[key]
                else:
                    result = func(*args, **kwargs)
                    cache[key] = result
                    return result

            return wrapper

    return decorator
