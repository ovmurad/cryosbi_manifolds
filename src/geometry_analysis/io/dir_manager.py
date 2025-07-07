from pathlib import Path
from typing import (
    Any,
    Dict,
    Generic,
    Hashable,
    Iterator,
    Literal,
    Optional,
    Sequence,
    Type,
    TypeAlias,
    TypeVar,
)

from .file_io import Data, Extension, IOHandler, PathLike, io_handler_factory
from .utils import get_file_paths_in_dir, is_empty_dir

# Dir Manager Modes

# safe: file_path must exist during init, cannot overwrite at save.
# append: if file_path doesn't exist at init make dir, cannot overwrite at save.
# append: if file_path doesn't exist at init make dir, if file exists at save, skip.
# overwrite: if file_path doesn't exist at init make dir, overwrite at save.
# new: if file_path doesn't exist at init make dir, if it does exist, empty it.
# overwrite at save(Warning: USE WITH CAUTION!!!).

DirManagerMode: TypeAlias = Literal["safe", "open", "append", "overwrite", "new"]

_KeyType = TypeVar("_KeyType", bound=Hashable)


class DirManager(Generic[Data]):

    __slots__ = ("io_handler", "dir_path", "mode")

    @staticmethod
    def _validate_dir_path(dir_path: PathLike, mode: DirManagerMode) -> Path:

        dir_path = Path(dir_path)

        if ext := dir_path.suffix:
            raise IOError(f"The provided dir path '{ext}' has an extension!")

        if mode == "safe":
            if not dir_path.exists():
                raise FileNotFoundError(f"Dir path '{dir_path}' does not exist!")
            if not dir_path.is_dir():
                raise FileExistsError(f"Dir path '{dir_path}' is a file!")
            print(f"Loaded dir '{dir_path}'.")
        elif mode in {"append", "overwrite", "new", "open"}:
            if not dir_path.exists():
                print(f"Created dir '{dir_path}'.")
                dir_path.mkdir()
            else:
                print(f"Loaded dir '{dir_path}'.")
        else:
            raise NotImplementedError(f"'{mode}' is not a valid DirManager Mode!")

        return dir_path

    @staticmethod
    def _validate_io_handler(io_handler: Extension) -> IOHandler[Data]:
        return io_handler_factory(io_handler)

    def __init__(
        self,
        dir_path: PathLike,
        io_handler: Extension,
        mode: DirManagerMode = "append",
    ):

        self.io_handler: IOHandler[Data] = self._validate_io_handler(io_handler)
        self.dir_path: Path = self._validate_dir_path(dir_path, mode)

        if mode == "new":
            self.delete_multiple()

        self.mode: DirManagerMode = mode

    @property
    def is_empty(self) -> bool:
        return is_empty_dir(self.dir_path)

    @property
    def ext(self) -> Extension:
        return self.io_handler.ext

    @property
    def names(self) -> Iterator[str]:
        return (file_path.name for file_path in self.paths)

    @property
    def paths(self) -> Iterator[Path]:
        return self.get_file_paths()

    def name_to_file_path(self, name: str) -> Path:
        return self.dir_path / f"{name}.{self.io_handler.ext}"

    def ask_file_path(self, name: str) -> Path | None:
        file_path = self.name_to_file_path(name)
        return file_path if file_path.exists() else None

    def get_file_path(self, name: str) -> Path:
        if (file_path := self.ask_file_path(name)) is None:
            raise FileNotFoundError(f"No file '{name}' was found in {self.dir_path}!")
        return file_path

    def ask_file_paths(
        self, names: Optional[Sequence[str]] = None, pattern: str = ".*"
    ) -> Iterator[Path | None]:
        if names is None:
            return get_file_paths_in_dir(self.dir_path, pattern, self.ext)
        return (self.ask_file_path(name) for name in names)

    def get_file_paths(
        self, names: Optional[Sequence[str]] = None, pattern: str = ".*"
    ) -> Iterator[Path]:
        if names is None:
            return get_file_paths_in_dir(self.dir_path, pattern, self.ext)
        return (self.get_file_path(name) for name in names)

    def get_names(
        self, names: Optional[Sequence[str]] = None, pattern: str = ".*"
    ) -> Iterator[str]:
        return (file_path.stem for file_path in self.get_file_paths(names, pattern))

    def save(self, name: str, data: Data, **kwargs: Any) -> None:

        file_path = self.name_to_file_path(name)

        if self.mode == "append" and file_path.exists():
            return
        elif self.mode in {"safe", "open"} and file_path.exists():
            raise FileExistsError(f"File '{file_path}' already exists!")

        self.io_handler.save(file_path, data, **kwargs)

    def load(self, name: str, **kwargs: Any) -> Data:
        return self.io_handler.load(self.get_file_path(name), **kwargs)

    def ask(self, name: str, **kwargs: Any) -> Data | None:
        file_path = self.ask_file_path(name)
        return None if file_path is None else self.io_handler.load(file_path, **kwargs)

    def delete(self, name: str, **kwargs: Any) -> None:
        return self.io_handler.delete(self.get_file_path(name), **kwargs)

    def save_multiple(self, data: Dict[Hashable, Data], **kwargs: Any) -> None:
        for name, datum in data.items():
            self.save(str(name), datum, **kwargs)

    def load_multiple(
        self,
        names: Optional[Sequence[str]] = None,
        pattern: str = ".*",
        key_type: Type[_KeyType] = str,
        dict_type: Type[Dict] = dict,
        **kwargs: Any,
    ) -> Dict[_KeyType, Data]:

        data = dict_type()
        for file_path in self.get_file_paths(names, pattern):
            data[key_type(file_path.stem)] = self.io_handler.load(file_path, **kwargs)
        return data

    def ask_multiple(
        self,
        names: Optional[Sequence[str]] = None,
        pattern: str = ".*",
        key_type: Type[_KeyType] = str,
        dict_type: Type[Dict] = dict,
        **kwargs: Any,
    ) -> Dict[_KeyType, Data | None]:

        data = dict_type()
        for file_path in self.ask_file_paths(names, pattern):
            data[key_type(file_path.stem)] = (
                None if file_path is None else self.io_handler.load(file_path, **kwargs)
            )

        return data

    def delete_multiple(
        self,
        names: Optional[Sequence[str]] = None,
        pattern: str = ".*",
        **kwargs: Any,
    ) -> None:
        for file_path in self.get_file_paths(names, pattern):
            self.io_handler.delete(file_path, **kwargs)
