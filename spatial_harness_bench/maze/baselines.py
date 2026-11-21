"""Reference maze agents speaking the JSON-lines protocol: lower and upper bounds for a suite.

    python -m spatial_harness_bench.maze --suite --seed 1 --output runs/x -- \\
        python -m spatial_harness_bench.maze.baselines dfs

right-hand, left-hand  keep one hand on the wall, no memory (v1 solved every level; v2: none)
random                 random open door, no memory
dfs                    remembers visited rooms and backtracks (always fits the limit)
"""

import argparse
import json
import random
import re
import sys

DIRECTIONS = ("вперёд", "вправо", "назад", "влево")
DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def open_doors(text: str) -> list[int]:
    return [d for d, name in enumerate(DIRECTIONS) if re.search(f"{name} — открыта", text)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("agent", choices=("right-hand", "left-hand", "random", "dfs"))
    parser.add_argument("--seed", type=int, default=0, help="random agent only")
    args = parser.parse_args()
    rng = random.Random(args.seed)
    heading = 0
    hand = 1 if args.agent == "right-hand" else -1
    position = (0, 0)  # dead reckoning; the agent never learns absolute coordinates
    visited = {position}
    stack: list[int] = []  # directions taken into unexplored rooms
    for line in sys.stdin:
        room = json.loads(line)
        if room["type"] == "end":
            break
        doors = open_doors(room["text"])
        if args.agent in {"right-hand", "left-hand"}:
            move = next((heading + o) % 4 for o in (hand, 0, -hand, 2) if (heading + o) % 4 in doors)
            heading = move
        elif args.agent == "random":
            move = rng.choice(doors)
        else:
            fresh = [d for d in doors
                     if (position[0] + DELTAS[d][0], position[1] + DELTAS[d][1]) not in visited]
            if fresh:
                move = fresh[0]
                stack.append(move)
            else:
                move = (stack.pop() + 2) % 4
            position = (position[0] + DELTAS[move][0], position[1] + DELTAS[move][1])
            visited.add(position)
        print(json.dumps({"move": DIRECTIONS[move]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
