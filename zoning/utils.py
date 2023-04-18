from functools import cache
import json
from pathlib import Path

from git.repo import Repo

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

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
