from pathlib import Path
from typing import Any, Generic

from .file_io import Data, Extension
from .dir_manager import DirManager, DirManagerMode


class DatabaseDir(Generic[Data]):

    def __init__(self, path: Path, io_handler: Extension, mode: DirManagerMode = "open"):
        self.dir_manager: DirManager[Data] = DirManager(path, io_handler, mode)

    def __getitem__(self, name: str) -> Data:
        return self.load(name)

    def __setitem__(self, name: str, data: Data) -> None:
        self.save(name, data)

    def load(self, name: str, **kwargs: Any) -> Data:
        return self.dir_manager.load(name, **kwargs)

    def save(self, name: str, data: Data, **kwargs: Any) -> None:
        self.dir_manager.save(name, data, **kwargs)
