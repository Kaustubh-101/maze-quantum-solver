# maze-quantum-solver

Grover Q-Learning maze solver with A* and Dijkstra.

The default maze is 6×6 and the Q-learning run uses 100 episodes to keep
runtime manageable.

## Setup and run

```powershell
python -m pip install -r requirements.txt
python maze_grover_qlearning.py
```

The script saves `maze_results.png` and `maze_qca_circuit.png` in the
current directory.
