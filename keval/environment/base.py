"""Defines the base environment class.

In keval, an environment is a class that defines the dynamics of the system,
similar to how a Gymnasium environment works.
"""

from abc import ABC, abstractmethod
from typing import Generic

from kinfer import proto as K

from keval.config import ConfigType, HasConfigMixin


class BaseEnvironment(ABC, HasConfigMixin[ConfigType], Generic[ConfigType]):
    """Base environment class."""

    @abstractmethod
    def get_schema(self) -> K.IO:
        """Get the input and output schema for the environment.

        Returns:
            The kinfer IO schema for the environment.
        """
