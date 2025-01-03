"""Example of evaluating the performance of a humanoid robot walking policy using keval.

To run this example, use:

```bash
python -m examples.humanoid output_dir=/path/to/output/dir
```
"""

from dataclasses import dataclass

import keval


@dataclass
class HumanoidConfig(keval.Config):
    """Humanoid evaluation configuration."""

    num_episodes: int = 10
    num_steps: int = 100


class HumanoidEnvironment(keval.BaseEnvironment[keval.Config]):
    """Humanoid environment."""

    def __init__(self) -> None:
        super().__init__()


class HumanoidEvaluation(keval.BaseEvaluation[keval.Config]):
    """Humanoid evaluation."""

    def __init__(self) -> None:
        super().__init__()


class HumanoidEvaluator(keval.BaseEvaluator[keval.Config]):
    """Humanoid evaluator."""

    def __init__(self) -> None:
        super().__init__()


if __name__ == "__main__":
    keval.run(
        environment=HumanoidEnvironment,
        evaluation=HumanoidEvaluation,
        evaluator=HumanoidEvaluator,
        num_episodes=10,
    )
