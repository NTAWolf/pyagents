# pyagents

**pyagents** is a small collection of tools and agents for a university project: Make an agent that learns to play Atari 2600 games. 

This repository is primarily intended for collaboration with @wji. As such, this README is downprioritized.
Regardless, the following provides a quick overview of the codebase for the interested.

## Quick code overview

- It is assumed that `ale_python_interface` (ALE) is installed.
- It is assumed that there is a directory on the same level as pyagents containing Atari ROMs, which must adhere to the naming standard defined by ALE.
- Visualization relies on `vispy`. You can choose to not visualize stuff when you set up the `GameManager`.

- **`game_manager.GameManager`** handles the interface between our agents, ALE, a logging system, and `vispy`.
- The **`agents`** subpackage contains agents that all fulfill the interface defined by **`agents.Agent`**.
- The **`experiments`** subpackage contains expirement recipes, that when imported by `run_experiment.py` conduct a full experiment with a `GameManager` and an agent.
- **`evaluate_experiment`** is (like everything else) WIP. It is intended to provide automated experiment evaluations.
