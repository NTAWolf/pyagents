# pyagents

**pyagents** is a small collection of tools and agents for a university project: Make an agent that learns to play Pong, the Atari 2600 game. 


## Quick code overview

- It is assumed that `ale_python_interface` (ALE) is installed.
- It is assumed that there is a directory on the same level as pyagents containing Atari ROMs, which must adhere to the naming standard defined by ALE.

- [`game_manager.GameManager`](game_manager.py) handles the interface between our agents, ALE, and a logging system.
- The [`agents`](agents) subpackage contains agents that all fulfill the interface defined by [`agents.Agent`](agents.py).
- The [`experiments`](experiments) subpackage contains expirement recipes, that when imported by [`run_experiment.py`](run_experiment.py) conduct a full experiment with a [`GameManager`](game_manager.py) and an agent.
- You conduct an experiment by e.g. `./run_experiment.py sarsa2`.
- [`evaluate_experiment.py`](evaluate_experiment.py) is used like `run_experiment.py`, and provides automated experiment evaluation.
- Pong-specific preprocessing tools are found in [`util.pongcess`](util/pongcess.py)


## Purpose of sharing

Do note that the performance is not impressive. We share the code in the hope that it may help others by providing ideas for a way of structuring the task, and by showing our take on the implementation on different tools such as [`CircularList`](util/collections.py).

If you would like to see an approach to dynamic module loading, check out [`experiments/__init__.py`](experiments/__init__.py) and its usage in [`run_experiment.py`](run_experiment.py) and [`evaluate_experiment.py`](evaluate_experiment.py).