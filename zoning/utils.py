from functools import cache
import json
from pathlib import Path

from git.repo import Repo

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
