# Limited-Oracle-Budget
Learning When to Ask for Help: Reinforcement Learning with a Limited Oracle Budget

## Overview

This project implements tabular reinforcement learning agents in discrete
environments (**CliffWalking-v1** and **FrozenLake-v1**) augmented with a
*limited oracle help mechanism*.

The agent may take a special **help action** at any step to query an oracle
policy.  Each query consumes one unit from a fixed per-episode **help budget**.
The remaining budget is encoded into the state representation, allowing the
agent to learn a policy over both the environment state and its available
assistance.

Three sets of experiments compare performance across:
| Dimension | Values |
|-----------|--------|
| Help budget | 0 (no oracle), 5, 10, 20, unlimited |
| Oracle quality | perfect, noisy (80 %), noisy (50 %), random |
| Help penalty | 0.0, −0.1, −0.5 |

Evaluation metrics: cumulative reward, success rate, help usage fraction.

## Project Structure

```
Limited-Oracle-Budget/
├── requirements.txt
├── src/
│   ├── environments/
│   │   └── oracle_env.py        # OracleEnv wrapper (augmented state + help action)
│   ├── agents/
│   │   ├── q_learning.py        # Tabular Q-learning agent
│   │   └── sarsa.py             # Tabular SARSA agent
│   ├── oracle/
│   │   ├── oracle.py            # PerfectOracle, NoisyOracle, RandomOracle
│   │   └── value_iteration.py   # Optimal policy via value iteration
│   └── utils/
│       ├── metrics.py           # EpisodeMetrics tracker
│       └── plotting.py          # Plotting utilities
├── experiments/
│   ├── training.py              # Training loops and multi-seed runner
│   └── run_experiments.py       # Main CLI experiment runner
└── tests/
    ├── test_oracle_env.py
    ├── test_agents.py
    └── test_oracle.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Running Experiments

```bash
# Full suite (both environments, both algorithms, 3 seeds, 2000 episodes)
python experiments/run_experiments.py

# Quick smoke test (300 episodes, 2 seeds)
python experiments/run_experiments.py --quick

# Single environment and algorithm
python experiments/run_experiments.py --env CliffWalking-v1 --algo qlearning

# Custom episode / seed count
python experiments/run_experiments.py --episodes 5000 --seeds 5 --output my_results
```

Figures are saved to `results/` (or the directory given by `--output`).

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```
