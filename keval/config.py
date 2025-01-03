"""Defines the keval base config class."""

import inspect
import sys
from dataclasses import dataclass, is_dataclass
from pathlib import Path
from typing import Generic, Protocol, Self, TypeVar, cast

from omegaconf import DictConfig, OmegaConf
from omegaconf.base import SCMode


@dataclass(kw_only=True)
class EnvironmentConfig:
    """Defines the environment config class."""


@dataclass(kw_only=True)
class EvaluationConfig:
    """Defines the evaluation config class."""


@dataclass(kw_only=True)
class EvaluatorConfig:
    """Defines the evaluator config class."""


@dataclass(kw_only=True)
class Config:
    """Defines the master config class."""

    environment: EnvironmentConfig
    evaluation: EvaluationConfig
    evaluator: EvaluatorConfig


ConfigType = TypeVar("ConfigType", bound=Config)

RawConfigType = Config | dict | DictConfig | str | Path


def _load_as_dict(path: str | Path) -> DictConfig:
    cfg = OmegaConf.load(path)
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Config file at {path} must be a dictionary, not {type(cfg)}!")
    return cfg


def get_config(cfg: RawConfigType, task_path: Path) -> DictConfig:
    if isinstance(cfg, (str, Path)):
        cfg = Path(cfg)
        if cfg.exists():
            try:
                cfg = _load_as_dict(cfg)
            except Exception as e:
                raise FileNotFoundError(f"Could not load config file at {cfg}!") from e
        elif task_path is not None and len(cfg.parts) == 1 and (other_cfg_path := task_path.parent / cfg).exists():
            try:
                cfg = _load_as_dict(other_cfg_path)
            except Exception as e:
                raise FileNotFoundError(f"Could not load config file at {other_cfg_path}!") from e
        else:
            raise FileNotFoundError(f"Could not find config file at {cfg}!")
    elif isinstance(cfg, dict):
        cfg = OmegaConf.create(cfg)
    elif is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)
    return cast(DictConfig, cfg)


class HasConfigProtocol(Protocol):
    cfg: ConfigType

    def __init__(self, cfg: ConfigType) -> None: ...


class HasConfigMixin(Generic[ConfigType]):
    def __init__(self, cfg: ConfigType) -> None:
        super().__init__()

        if not isinstance(cfg, Config):
            raise ValueError(f"Config must be a subclass of {Config.__name__}, not {type(cfg).__name__}!")

        self.cfg = cfg

    @classmethod
    def get_config_class(cls) -> type[Config]:
        """Recursively retrieves the config class from the generic type.

        Returns:
            The parsed config class.

        Raises:
            ValueError: If the config class cannot be found, usually meaning
            that the generic class has not been used correctly.
        """
        if hasattr(cls, "__orig_bases__"):
            for base in cls.__orig_bases__:
                if hasattr(base, "__args__"):
                    for arg in base.__args__:
                        if issubclass(arg, Config):
                            return arg

        raise ValueError(
            "The config class could not be parsed from the generic type, which usually means that the class is not "
            "being instantiated correctly. The config class should be specified as a generic argument to the class, "
            "e.g. `class MyEnvironment(BaseEnvironment[Config]): ...`"
        )

    @classmethod
    def get_config(cls, *cfgs: RawConfigType, use_cli: bool | list[str] = True) -> Config:
        """Builds the structured config from the provided config classes.

        Args:
            cfgs: The config classes to merge. If a string or Path is provided,
                it will be loaded as a YAML file.
            use_cli: Whether to allow additional overrides from the CLI.

        Returns:
            The merged configs.
        """
        task_path = Path(inspect.getfile(cls))
        cfg = OmegaConf.structured(cls.get_config_class())
        cfg = OmegaConf.merge(cfg, *(get_config(other_cfg, task_path) for other_cfg in cfgs))
        if use_cli:
            args = use_cli if isinstance(use_cli, list) else sys.argv[1:]
            if "-h" in args or "--help" in args:
                sys.stderr.write(OmegaConf.to_yaml(cfg))
                sys.stderr.flush()
                sys.exit(0)

            # Attempts to load any paths as configs.
            is_path = [Path(arg).is_file() or (task_path / arg).is_file() for arg in args]
            paths = [arg for arg, is_path in zip(args, is_path) if is_path]
            non_paths = [arg for arg, is_path in zip(args, is_path) if not is_path]
            if paths:
                cfg = OmegaConf.merge(cfg, *(get_config(path, task_path) for path in paths))
            cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(non_paths))

        return cast(
            Config,
            OmegaConf.to_container(
                cfg,
                resolve=True,
                throw_on_missing=True,
                structured_config_mode=SCMode.INSTANTIATE,
            ),
        )

    @classmethod
    def config_str(cls, *cfgs: RawConfigType, use_cli: bool | list[str] = True) -> str:
        return OmegaConf.to_yaml(cls.get_config(*cfgs, use_cli=use_cli))

    @classmethod
    def get_task(cls, *cfgs: RawConfigType, use_cli: bool | list[str] = True) -> Self:
        """Builds the task from the provided config classes.

        Args:
            cfgs: The config classes to merge. If a string or Path is provided,
                it will be loaded as a YAML file.
            use_cli: Whether to allow additional overrides from the CLI.

        Returns:
            The task.
        """
        cfg = cls.get_config(*cfgs, use_cli=use_cli)
        return cls(cfg)
