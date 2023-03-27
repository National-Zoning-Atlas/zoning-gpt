from functools import cache
from pathlib import Path

from git.repo import Repo

@cache
def get_project_root() -> Path:
    repo = Repo(".", search_parent_directories=True)
    path = repo.working_tree_dir

    assert path is not None, "You are not currently in a Git repo!"

    return Path(path)
