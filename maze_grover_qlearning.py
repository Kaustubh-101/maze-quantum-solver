"""
maze_grover_qlearning.py
══════════════════════════════════════════════════════════════════
Complicated Maze Solver — Three Algorithms Side by Side:

  1. Grover Q-Learning  (Quantum RL + Grover amplitude amplification)
  2. A* Search          (classical optimal pathfinder)
  3. Dijkstra           (classical uniform-cost pathfinder)

Outputs:
  • maze_results.png       — side-by-side path visualization + timing
  • maze_qca_circuit.png   — QCA gate formation for Grover Q-Learning
  • timing_comparison.png  — bar chart: time + steps for all 3 algos

Install:
  pip install pennylane numpy matplotlib

Run:
  python maze_grover_qlearning.py
══════════════════════════════════════════════════════════════════
"""

import time
import heapq
from functools import lru_cache
import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from collections import defaultdict
from matplotlib.colors import LinearSegmentedColormap

# ══════════════════════════════════════════════════════════════════
# 1.  MAZE DEFINITION  (6×6 maze, 0=path, 1=wall)
# ══════════════════════════════════════════════════════════════════
MAZE = np.array([
    [0,0,1,1,1,1],
    [1,0,1,1,1,1],
    [1,0,0,0,1,1],
    [1,1,1,0,1,1],
    [1,1,1,0,0,0],
    [1,1,1,1,1,0],
], dtype=int)

ROWS, COLS = MAZE.shape
START = (0, 0)
GOAL  = (ROWS - 1, COLS - 1)
N_ACTIONS = 4          # up, down, left, right
N_QUBITS  = 2          # 2^2 = 4 actions
ACTIONS   = [(-1,0),(1,0),(0,-1),(0,1)]   # up/down/left/right
ACTION_LABELS = ['↑','↓','←','→']

# ══════════════════════════════════════════════════════════════════
# 2.  HELPER: valid moves
# ══════════════════════════════════════════════════════════════════
def valid_neighbors(r, c):
    neighbors = []
    for i, (dr, dc) in enumerate(ACTIONS):
        nr, nc = r + dr, c + dc
        if 0 <= nr < ROWS and 0 <= nc < COLS and MAZE[nr][nc] == 0:
            neighbors.append((i, nr, nc))
    return neighbors

# ══════════════════════════════════════════════════════════════════
# 3.  QUANTUM GROVER CIRCUIT
# ══════════════════════════════════════════════════════════════════
dev = qml.device("default.qubit", wires=N_QUBITS)

def grover_oracle(best_action, n_qubits):
    bits = [(best_action >> (n_qubits - 1 - i)) & 1 for i in range(n_qubits)]
    for i, b in enumerate(bits):
        if b == 0:
            qml.PauliX(wires=i)
    qml.ctrl(qml.PauliZ, control=list(range(n_qubits - 1)))(wires=n_qubits - 1)
    for i, b in enumerate(bits):
        if b == 0:
            qml.PauliX(wires=i)

def diffusion_operator(n_qubits):
    for w in range(n_qubits):
        qml.Hadamard(wires=w)
        qml.PauliX(wires=w)
    qml.ctrl(qml.PauliZ, control=list(range(n_qubits - 1)))(wires=n_qubits - 1)
    for w in range(n_qubits):
        qml.PauliX(wires=w)
        qml.Hadamard(wires=w)

@qml.qnode(dev)
def grover_circuit(best_action, n_iterations):
    for w in range(N_QUBITS):
        qml.Hadamard(wires=w)
    for _ in range(n_iterations):
        grover_oracle(best_action, N_QUBITS)
        diffusion_operator(N_QUBITS)
    return qml.probs(wires=range(N_QUBITS))

@lru_cache(maxsize=N_ACTIONS)
def _grover_probabilities(best_action):
    n_iter = max(1, int(np.floor(np.pi / 4 * np.sqrt(N_ACTIONS))))
    probs = np.array(grover_circuit(best_action, n_iter))[:N_ACTIONS]
    normalized = np.abs(probs) / (np.sum(np.abs(probs)) + 1e-12)
    return tuple(normalized)

def grover_select(q_vals, explore=True):
    best  = int(np.argmax(q_vals))
    probs  = np.array(_grover_probabilities(best))
    action = np.random.choice(N_ACTIONS, p=probs) if explore else int(np.argmax(probs))
    return action, probs

# ══════════════════════════════════════════════════════════════════
# 4.  GROVER Q-LEARNING
# ══════════════════════════════════════════════════════════════════
def run_grover_qlearning(episodes=100, alpha=0.7, gamma=0.95):
    print("\n[1/3] Running Grover Q-Learning...")
    Q = defaultdict(lambda: np.zeros(N_ACTIONS))
    rewards_log, steps_log = [], []
    best_path = None
    best_reward = -np.inf
    explore_eps = int(episodes * 0.65)

    t0 = time.perf_counter()

    for ep in range(episodes):
        state = START
        total_reward = 0
        path = [state]
        visited = {state}
        done = False
        steps = 0
        explore = ep < explore_eps

        while not done and steps < ROWS * COLS * 2:
            r, c = state
            nbrs = valid_neighbors(r, c)
            if not nbrs:
                break

            # Only consider valid actions
            valid_mask = [False] * N_ACTIONS
            for idx, _, _ in nbrs:
                valid_mask[idx] = True

            q_valid = Q[state].copy()
            for i in range(N_ACTIONS):
                if not valid_mask[i]:
                    q_valid[i] = -9999

            action, probs = grover_select(q_valid, explore=explore)

            # If selected action is invalid, pick random valid
            if not valid_mask[action]:
                action = np.random.choice([idx for idx, _, _ in nbrs])

            # Step
            dr, dc = ACTIONS[action]
            nr, nc = r + dr, c + dc
            next_state = (nr, nc)

            if next_state == GOAL:
                reward = 100.0
                done = True
            elif next_state in visited:
                reward = -2.0
            else:
                reward = -0.5

            visited.add(next_state)
            best_next = np.max(Q[next_state])
            td = reward + gamma * best_next - Q[state][action]
            Q[state][action] += alpha * td

            state = next_state
            path.append(state)
            total_reward += reward
            steps += 1

        rewards_log.append(total_reward)
        steps_log.append(steps)

        if total_reward > best_reward and done:
            best_reward = total_reward
            best_path = path[:]

        if (ep + 1) % 100 == 0:
            avg = np.mean(rewards_log[-100:])
            sr  = sum(1 for r in rewards_log[-100:] if r > 0) / 100
            print(f"  ep {ep+1:4d} | avg_reward={avg:8.2f} | success={sr:.0%}")

    elapsed = time.perf_counter() - t0

    # Extract greedy path after training
    state = START
    greedy_path = [state]
    visited = {state}
    for _ in range(ROWS * COLS * 3):
        if state == GOAL:
            break
        r, c = state
        nbrs = valid_neighbors(r, c)
        if not nbrs:
            break
        q_valid = Q[state].copy()
        for i in range(N_ACTIONS):
            if not any(idx == i for idx, _, _ in nbrs):
                q_valid[i] = -9999
        action = int(np.argmax(q_valid))
        dr, dc = ACTIONS[action]
        next_state = (r + dr, c + dc)
        if next_state in visited:
            break
        visited.add(next_state)
        greedy_path.append(next_state)
        state = next_state

    final_path = greedy_path if greedy_path[-1] == GOAL else best_path
    print(f"  Done in {elapsed:.3f}s | path_len={len(final_path) if final_path else '?'}")
    return final_path, elapsed, rewards_log, steps_log, Q

# ══════════════════════════════════════════════════════════════════
# 5.  A* SEARCH
# ══════════════════════════════════════════════════════════════════
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def run_astar():
    print("\n[2/3] Running A* Search...")
    t0 = time.perf_counter()
    open_set = [(0 + heuristic(START, GOAL), 0, START, [START])]
    visited = set()

    while open_set:
        f, g, current, path = heapq.heappop(open_set)
        if current in visited:
            continue
        visited.add(current)
        if current == GOAL:
            elapsed = time.perf_counter() - t0
            print(f"  Done in {elapsed:.6f}s | path_len={len(path)}")
            return path, elapsed
        r, c = current
        for _, nr, nc in valid_neighbors(r, c):
            nxt = (nr, nc)
            if nxt not in visited:
                ng = g + 1
                heapq.heappush(open_set, (ng + heuristic(nxt, GOAL), ng, nxt, path + [nxt]))

    elapsed = time.perf_counter() - t0
    print("  A* failed to find path")
    return None, elapsed

# ══════════════════════════════════════════════════════════════════
# 6.  DIJKSTRA
# ══════════════════════════════════════════════════════════════════
def run_dijkstra():
    print("\n[3/3] Running Dijkstra Search...")
    t0 = time.perf_counter()
    dist = {START: 0}
    prev = {START: None}
    pq   = [(0, START)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, float('inf')):
            continue
        if u == GOAL:
            break
        r, c = u
        for _, nr, nc in valid_neighbors(r, c):
            v = (nr, nc)
            nd = d + 1
            if nd < dist.get(v, float('inf')):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    # Reconstruct path
    path, node = [], GOAL
    while node is not None:
        path.append(node)
        node = prev.get(node)
    path.reverse()

    elapsed = time.perf_counter() - t0
    if path and path[0] == START:
        print(f"  Done in {elapsed:.6f}s | path_len={len(path)}")
        return path, elapsed
    print("  Dijkstra failed to find path")
    return None, elapsed

# ══════════════════════════════════════════════════════════════════
# 7.  QCA GATE FORMATION VISUALIZATION
#     Shows the quantum circuit gates used in Grover iteration
# ══════════════════════════════════════════════════════════════════
def draw_qca_circuit(Q_table, sample_state=(0,0)):
    """
    Draw the QCA (Quantum Circuit Architecture) gate formation
    for one Grover Q-Learning step at a given maze state.
    Shows: H gates → Oracle → Diffusion → Measurement
    """
    print("\n[+] Drawing QCA gate formation...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor('#06080f')

    # ── LEFT: Full circuit diagram ──
    ax = axes[0]
    ax.set_facecolor('#06080f')
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, 3.5)
    ax.set_title('QCA Gate Formation — Grover Q-Learning Step',
                 color='#00d4ff', fontsize=11, pad=12, fontfamily='monospace')

    # qubit wires
    wire_colors = ['#00d4ff', '#00ff9d']
    for q in range(N_QUBITS):
        y = N_QUBITS - 1 - q
        ax.plot([0.2, 9.8], [y, y], color='#1c2538', lw=1.5, zorder=1)
        ax.text(-0.05, y, f'q{q}', color=wire_colors[q],
                fontsize=10, va='center', ha='right', fontfamily='monospace')

    def gate_box(ax, x, y, label, color='#1c2538', text_color='#00d4ff',
                 w=0.55, h=0.5):
        rect = plt.Rectangle((x - w/2, y - h/2), w, h,
                              facecolor=color, edgecolor=text_color,
                              linewidth=1.2, zorder=3)
        ax.add_patch(rect)
        ax.text(x, y, label, color=text_color, fontsize=9,
                ha='center', va='center', fontfamily='monospace',
                fontweight='bold', zorder=4)

    def ctrl_dot(ax, xc, yc, yt, color='#ffb830'):
        ax.plot([xc, xc], [min(yc,yt), max(yc,yt)],
                color=color, lw=1.5, zorder=2)
        ax.plot(xc, yc, 'o', color=color, ms=7, zorder=4)
        ax.plot(xc, yt, '+', color=color, ms=10, mew=2, zorder=4)

    # Section labels
    def section(ax, x, label, color):
        ax.text(x, 2.9, label, color=color, fontsize=8,
                ha='center', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#0d1220',
                          edgecolor=color, lw=0.8))
        ax.axvline(x - 0.55, color=color, lw=0.4, ls='--', alpha=0.3)

    # ── State prep: |0⟩ init ──
    section(ax, 0.6, '|0⟩ init', '#5a7090')
    for q in range(N_QUBITS):
        y = N_QUBITS - 1 - q
        ax.text(0.4, y, '|0⟩', color='#5a7090', fontsize=9,
                ha='center', va='center', fontfamily='monospace')

    # ── Hadamard layer ──
    section(ax, 1.6, 'H ⊗ H', '#00d4ff')
    for q in range(N_QUBITS):
        y = N_QUBITS - 1 - q
        gate_box(ax, 1.6, y, 'H', '#0a1525', '#00d4ff')

    # ── Oracle: phase flip on best action ──
    q_vals = Q_table.get(sample_state, np.zeros(N_ACTIONS))
    best_a = int(np.argmax(q_vals))
    bits   = [(best_a >> (N_QUBITS - 1 - i)) & 1 for i in range(N_QUBITS)]

    section(ax, 3.2, f'Oracle\n(mark a={best_a})', '#ff4d6d')
    for q in range(N_QUBITS):
        y = N_QUBITS - 1 - q
        if bits[q] == 0:
            gate_box(ax, 2.7, y, 'X', '#1a0a0f', '#ff4d6d')
    ctrl_dot(ax, 3.5, 1, 0, color='#ff4d6d')
    for q in range(N_QUBITS):
        y = N_QUBITS - 1 - q
        if bits[q] == 0:
            gate_box(ax, 4.1, y, 'X', '#1a0a0f', '#ff4d6d')

    # ── Diffusion operator ──
    section(ax, 5.8, 'Diffusion\n2|ψ⟩⟨ψ|−I', '#00ff9d')
    for q in range(N_QUBITS):
        y = N_QUBITS - 1 - q
        gate_box(ax, 5.0, y, 'H', '#0a1a10', '#00ff9d')
        gate_box(ax, 5.6, y, 'X', '#0a1a10', '#00ff9d')
    ctrl_dot(ax, 6.2, 1, 0, color='#00ff9d')
    for q in range(N_QUBITS):
        y = N_QUBITS - 1 - q
        gate_box(ax, 6.8, y, 'X', '#0a1a10', '#00ff9d')
        gate_box(ax, 7.4, y, 'H', '#0a1a10', '#00ff9d')

    # ── Measurement ──
    section(ax, 9.0, 'Measure', '#ffb830')
    for q in range(N_QUBITS):
        y = N_QUBITS - 1 - q
        rect = plt.Rectangle((8.6, y - 0.28), 0.8, 0.56,
                              facecolor='#1a1400', edgecolor='#ffb830',
                              lw=1.2, zorder=3)
        ax.add_patch(rect)
        arc = mpatches.Arc((9.0, y - 0.05), 0.5, 0.4,
                           theta1=0, theta2=180,
                           color='#ffb830', lw=1.2, zorder=4)
        ax.add_patch(arc)
        ax.annotate('', xy=(9.25, y + 0.2), xytext=(9.0, y - 0.05),
                    arrowprops=dict(arrowstyle='->', color='#ffb830', lw=1.2))

    # Q-val annotation
    ax.text(5.0, -0.42, f'State: {sample_state}  |  Q-values: {np.round(q_vals,2)}  |  Best action: {ACTION_LABELS[best_a]} (a={best_a})',
            color='#5a7090', fontsize=8, ha='center', fontfamily='monospace')

    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor('#1c2538')

    # ── RIGHT: Grover amplitude bars across iterations ──
    ax2 = axes[1]
    ax2.set_facecolor('#06080f')
    ax2.set_title('Grover Amplitude Amplification\nP(best action) per iteration',
                  color='#00d4ff', fontsize=11, pad=12, fontfamily='monospace')

    iters_range = range(0, 7)
    colors_iter = plt.cm.plasma(np.linspace(0.3, 0.9, N_ACTIONS))

    for iter_k in iters_range:
        theta = np.arcsin(1 / np.sqrt(N_ACTIONS))
        probs = []
        for a in range(N_ACTIONS):
            if a == best_a:
                p = np.sin((2 * iter_k + 1) * theta) ** 2
            else:
                p = (1 - np.sin((2 * iter_k + 1) * theta) ** 2) / (N_ACTIONS - 1)
            probs.append(p)

        x_base = iter_k * 1.3
        bar_w  = 0.25
        for a, p in enumerate(probs):
            bar_color = '#00d4ff' if a == best_a else '#1c3a50'
            edge_c    = '#00d4ff' if a == best_a else '#1c2538'
            ax2.bar(x_base + a * bar_w, p, width=bar_w * 0.85,
                    color=bar_color, edgecolor=edge_c, lw=0.5, zorder=3)

        ax2.text(x_base + bar_w * 1.5, -0.04, f'k={iter_k}',
                 color='#5a7090', fontsize=8, ha='center', fontfamily='monospace')

    ax2.axhline(1/N_ACTIONS, color='#ffb830', lw=1, ls='--',
                label=f'Uniform 1/N = {1/N_ACTIONS:.2f}', zorder=2)
    ax2.axhline(1.0, color='#5a7090', lw=0.5, ls=':', alpha=0.5)
    ax2.set_ylim(-0.08, 1.15)
    ax2.set_xlim(-0.3, 9.0)
    ax2.set_ylabel('Probability', color='#5a7090', fontfamily='monospace')
    ax2.set_xticks([])
    ax2.tick_params(colors='#5a7090')
    ax2.legend(fontsize=8, facecolor='#0d1220', labelcolor='#ffb830',
               edgecolor='#1c2538')

    # legend for bars
    best_patch  = mpatches.Patch(color='#00d4ff', label=f'Best action a={best_a} ({ACTION_LABELS[best_a]})')
    other_patch = mpatches.Patch(color='#1c3a50', label='Other actions')
    ax2.legend(handles=[best_patch, other_patch,
               mpatches.Patch(color='none', label=f'Uniform = {1/N_ACTIONS:.2f}')],
               fontsize=8, facecolor='#0d1220', labelcolor='#d8e4f0',
               edgecolor='#1c2538', loc='upper right')

    for spine in ax2.spines.values():
        spine.set_edgecolor('#1c2538')
    ax2.yaxis.label.set_color('#5a7090')
    ax2.tick_params(colors='#5a7090')

    plt.tight_layout(pad=2)
    plt.savefig('maze_qca_circuit.png', dpi=150, bbox_inches='tight',
                facecolor='#06080f')
    print("  Saved: maze_qca_circuit.png")
    plt.show()

# ══════════════════════════════════════════════════════════════════
# 8.  MAZE + PATH VISUALIZATION
# ══════════════════════════════════════════════════════════════════
def draw_maze_comparison(grover_path, astar_path, dijkstra_path,
                          grover_time, astar_time, dijkstra_time,
                          rewards_log):
    print("\n[+] Drawing maze comparison...")
    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor('#06080f')
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.3)

    cmap_maze = LinearSegmentedColormap.from_list(
        'maze', ['#0d1a2e', '#1c2538'])

    def draw_single_maze(ax, path, title, path_color, time_s, algo_label):
        ax.set_facecolor('#06080f')
        # walls
        for r in range(ROWS):
            for c in range(COLS):
                if MAZE[r][c] == 1:
                    ax.add_patch(plt.Rectangle(
                        (c, ROWS - 1 - r), 1, 1,
                        facecolor='#1c2e45', edgecolor='#0d1a2e', lw=0.3))
                else:
                    ax.add_patch(plt.Rectangle(
                        (c, ROWS - 1 - r), 1, 1,
                        facecolor='#06080f', edgecolor='#0d1525', lw=0.2))

        # start / goal
        ax.add_patch(plt.Rectangle((START[1], ROWS-1-START[0]), 1, 1,
                     facecolor='#00ff9d', alpha=0.8, zorder=3))
        ax.add_patch(plt.Rectangle((GOAL[1], ROWS-1-GOAL[0]), 1, 1,
                     facecolor='#ff4d6d', alpha=0.8, zorder=3))
        ax.text(START[1]+0.5, ROWS-1-START[0]+0.5, 'S',
                color='#06080f', ha='center', va='center',
                fontsize=7, fontweight='bold', fontfamily='monospace', zorder=4)
        ax.text(GOAL[1]+0.5, ROWS-1-GOAL[0]+0.5, 'G',
                color='#06080f', ha='center', va='center',
                fontsize=7, fontweight='bold', fontfamily='monospace', zorder=4)

        # path
        if path:
            for i, (r, c) in enumerate(path):
                alpha = 0.3 + 0.7 * i / len(path)
                ax.add_patch(plt.Rectangle(
                    (c, ROWS-1-r), 1, 1,
                    facecolor=path_color, alpha=alpha * 0.6, zorder=2))
            # path line
            xs = [c + 0.5 for r, c in path]
            ys = [ROWS - 1 - r + 0.5 for r, c in path]
            ax.plot(xs, ys, color=path_color, lw=1.5, alpha=0.9, zorder=5)
            ax.plot(xs[0], ys[0], 'o', color='#00ff9d', ms=5, zorder=6)
            ax.plot(xs[-1], ys[-1], '*', color='#ff4d6d', ms=8, zorder=6)

        path_len = len(path) - 1 if path else 0
        ax.set_xlim(0, COLS); ax.set_ylim(0, ROWS)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'{title}\n{algo_label}  |  steps={path_len}  |  t={time_s*1000:.2f}ms',
                     color=path_color, fontsize=9, pad=6, fontfamily='monospace')
        for spine in ax.spines.values():
            spine.set_edgecolor('#1c2538')

    ax1 = fig.add_subplot(gs[0, 0])
    draw_single_maze(ax1, grover_path,   'Grover Q-Learning', '#00d4ff', grover_time,   'Quantum RL')
    ax2 = fig.add_subplot(gs[0, 1])
    draw_single_maze(ax2, astar_path,    'A* Search',         '#ffb830', astar_time,    'Classical')
    ax3 = fig.add_subplot(gs[0, 2])
    draw_single_maze(ax3, dijkstra_path, 'Dijkstra',          '#00ff9d', dijkstra_time, 'Classical')

    # overlay all paths on one maze
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.set_facecolor('#06080f')
    for r in range(ROWS):
        for c in range(COLS):
            color = '#1c2e45' if MAZE[r][c] == 1 else '#06080f'
            ax4.add_patch(plt.Rectangle((c, ROWS-1-r), 1, 1,
                facecolor=color, edgecolor='#0d1525', lw=0.2))
    ax4.add_patch(plt.Rectangle((START[1], ROWS-1-START[0]), 1, 1,
                  facecolor='#00ff9d', alpha=0.9, zorder=3))
    ax4.add_patch(plt.Rectangle((GOAL[1], ROWS-1-GOAL[0]), 1, 1,
                  facecolor='#ff4d6d', alpha=0.9, zorder=3))
    for path, col, lbl in [
        (grover_path,   '#00d4ff', 'Grover Q-RL'),
        (astar_path,    '#ffb830', 'A*'),
        (dijkstra_path, '#00ff9d', 'Dijkstra'),
    ]:
        if path:
            xs = [c+0.5 for r,c in path]
            ys = [ROWS-1-r+0.5 for r,c in path]
            ax4.plot(xs, ys, color=col, lw=1.5, alpha=0.85,
                     label=lbl, zorder=5)
    ax4.set_xlim(0,COLS); ax4.set_ylim(0,ROWS)
    ax4.set_xticks([]); ax4.set_yticks([])
    ax4.legend(fontsize=7, facecolor='#0d1220', labelcolor='#d8e4f0',
               edgecolor='#1c2538', loc='lower right')
    ax4.set_title('All paths — overlay',
                  color='#d8e4f0', fontsize=9, pad=6, fontfamily='monospace')
    for spine in ax4.spines.values():
        spine.set_edgecolor('#1c2538')

    # ── Reward curve ──
    ax5 = fig.add_subplot(gs[1, :2])
    ax5.set_facecolor('#06080f')
    ax5.plot(rewards_log, alpha=0.2, color='#00d4ff', lw=0.7)
    roll = np.convolve(rewards_log, np.ones(30)/30, mode='valid')
    ax5.plot(np.arange(29, len(rewards_log)), roll,
             color='#00d4ff', lw=2, label='30-ep avg')
    ax5.axhline(0, color='#5a7090', lw=0.5, ls='--')
    ax5.set_title('Grover Q-Learning — reward per episode',
                  color='#00d4ff', fontsize=9, fontfamily='monospace')
    ax5.set_xlabel('Episode', color='#5a7090', fontfamily='monospace')
    ax5.set_ylabel('Reward', color='#5a7090', fontfamily='monospace')
    ax5.tick_params(colors='#5a7090')
    ax5.grid(color='#1c2538', alpha=0.5)
    ax5.legend(fontsize=8, facecolor='#0d1220', labelcolor='#00d4ff', edgecolor='#1c2538')
    for spine in ax5.spines.values(): spine.set_edgecolor('#1c2538')

    # ── Timing comparison ──
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor('#06080f')
    algos  = ['Grover\nQ-RL', 'A*', 'Dijkstra']
    times  = [grover_time, astar_time, dijkstra_time]
    colors = ['#00d4ff', '#ffb830', '#00ff9d']
    bars = ax6.bar(algos, times, color=colors, edgecolor='#1c2538', lw=0.8)
    for bar, t in zip(bars, times):
        ax6.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + max(times)*0.01,
                 f'{t*1000:.2f}ms', color='#d8e4f0',
                 ha='center', fontsize=8, fontfamily='monospace')
    ax6.set_title('Time comparison', color='#d8e4f0', fontsize=9, fontfamily='monospace')
    ax6.set_ylabel('Time (s)', color='#5a7090', fontfamily='monospace')
    ax6.tick_params(colors='#5a7090')
    ax6.grid(color='#1c2538', alpha=0.4, axis='y')
    for spine in ax6.spines.values(): spine.set_edgecolor('#1c2538')

    # ── Steps comparison ──
    ax7 = fig.add_subplot(gs[1, 3])
    ax7.set_facecolor('#06080f')
    path_lens = [
        len(grover_path) - 1 if grover_path else 0,
        len(astar_path)  - 1 if astar_path  else 0,
        len(dijkstra_path)-1 if dijkstra_path else 0,
    ]
    bars2 = ax7.bar(algos, path_lens, color=colors, edgecolor='#1c2538', lw=0.8)
    for bar, s in zip(bars2, path_lens):
        ax7.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + max(path_lens)*0.01,
                 str(s), color='#d8e4f0',
                 ha='center', fontsize=8, fontfamily='monospace')
    ax7.set_title('Path length (steps)', color='#d8e4f0', fontsize=9, fontfamily='monospace')
    ax7.set_ylabel('Steps', color='#5a7090', fontfamily='monospace')
    ax7.tick_params(colors='#5a7090')
    ax7.grid(color='#1c2538', alpha=0.4, axis='y')
    for spine in ax7.spines.values(): spine.set_edgecolor('#1c2538')

    fig.suptitle(f'Maze Solver — Grover Q-Learning  vs  A*  vs  Dijkstra  |  {ROWS}×{COLS} Maze',
                 color='#d8e4f0', fontsize=12, fontfamily='monospace', y=0.98)

    plt.savefig('maze_results.png', dpi=150, bbox_inches='tight',
                facecolor='#06080f')
    print("  Saved: maze_results.png")
    plt.show()

# ══════════════════════════════════════════════════════════════════
# 9.  MAIN
# ══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("═" * 56)
    print("  Maze Solver: Grover Q-Learning + A* + Dijkstra")
    print(f"  Maze size: {ROWS}×{COLS}  |  Start:{START} → Goal:{GOAL}")
    print("═" * 56)

    # Run all three
    grover_path, grover_time, rewards_log, steps_log, Q_table = \
        run_grover_qlearning(episodes=100, alpha=0.7, gamma=0.95)

    astar_path,    astar_time    = run_astar()
    dijkstra_path, dijkstra_time = run_dijkstra()

    # Print summary
    print("\n" + "═" * 56)
    print("  RESULTS SUMMARY")
    print("═" * 56)
    for name, path, t in [
        ("Grover Q-RL", grover_path,   grover_time),
        ("A*",          astar_path,    astar_time),
        ("Dijkstra",    dijkstra_path, dijkstra_time),
    ]:
        steps = len(path) - 1 if path else 0
        print(f"  {name:<14} steps={steps:3d}  time={t*1000:8.3f}ms")
    print("═" * 56)

    # Draw outputs
    draw_maze_comparison(grover_path, astar_path, dijkstra_path,
                         grover_time, astar_time, dijkstra_time,
                         rewards_log)

    # QCA gate diagram (use a state near the start)
    sample = (2, 3)
    draw_qca_circuit(dict(Q_table), sample_state=sample)

    print("\n[✓] All outputs saved:")
    print("    maze_results.png     — path visualization + timing")
    print("    maze_qca_circuit.png — QCA gate formation diagram")
