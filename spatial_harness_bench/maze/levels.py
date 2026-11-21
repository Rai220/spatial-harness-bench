"""Difficulty bands for the ten benchmark tasks (version 2). Both bounds are inclusive.

Version 1 used tree mazes. In a tree, following one wall visits every edge at most twice, so a
memoryless agent always found the exit within the limit of two decisions per room. Version 2
opens extra doors to create loops and accepts only maps on which every wall follower (either
hand, any initial heading) circles forever. Finding the exit then requires remembering where
the agent has been. The limit stays at two decisions per room: depth-first exploration with
memory needs at most 2·(rooms−1) moves, so a perfect mapper always fits.
"""

from dataclasses import dataclass
from itertools import permutations
import random
import statistics

from .game import OPEN, Maze

VERSION = 2
LOOPS = 0.15


@dataclass(frozen=True)
class Level:
    size: int
    shortest: tuple[int, int]
    explore: tuple[int, int]


LEVELS = (
    Level(5, (3, 6), (15, 24)),
    Level(6, (7, 8), (25, 32)),
    Level(7, (9, 10), (33, 45)),
    Level(8, (11, 12), (46, 58)),
    Level(10, (13, 14), (59, 80)),
    Level(11, (15, 16), (81, 100)),
    Level(12, (17, 18), (101, 120)),
    Level(13, (19, 20), (121, 145)),
    Level(14, (21, 23), (146, 170)),
    Level(15, (24, 28), (171, 210)),
)


def wall_follower(maze: Maze, hand: int, heading: int, limit: int) -> int | None:
    """Steps to the exit keeping a hand on the wall (hand=+1 right, −1 left), or None if it loops."""
    cell, seen, steps = maze.start, set(), 0
    while steps < limit:
        if cell == maze.exit:
            return steps
        if (cell, heading) in seen:
            return None
        seen.add((cell, heading))
        for offset in (hand, 0, -hand, 2):
            direction = (heading + offset) % 4
            if maze.rooms[cell][direction] == OPEN:
                heading, cell, steps = direction, maze.neighbor(cell, direction), steps + 1
                break
        else:
            return None
    return steps if cell == maze.exit else None


def wall_followers_fail(maze: Maze, limit: int) -> bool:
    return all(wall_follower(maze, hand, heading, limit) is None
               for hand in (1, -1) for heading in range(4))


def dfs_steps(maze: Maze, order: tuple[int, ...]) -> int:
    """Moves of a depth-first explorer that remembers visited rooms and backtracks."""
    cell, visited, stack, steps = maze.start, {maze.start}, [], 0
    while cell != maze.exit:
        for direction in order:
            other = maze.neighbor(cell, direction)
            if maze.rooms[cell][direction] == OPEN and other not in visited:
                stack.append(cell)
                cell = other
                visited.add(other)
                break
        else:
            cell = stack.pop()
        steps += 1
    return steps


def explore_cost(maze: Maze) -> int:
    """Median DFS moves over all 24 direction preferences: the cost of finding an unknown exit."""
    return int(statistics.median(dfs_steps(maze, order) for order in permutations(range(4))))


def generate_level(number: int, seed: int) -> tuple[Maze, int, int]:
    """Return the accepted maze, its replay seed and its exploration cost."""
    if not 1 <= number <= len(LEVELS):
        raise ValueError(f"Номер уровня должен быть от 1 до {len(LEVELS)}.")
    level = LEVELS[number - 1]
    rng = random.Random(f"{seed}-{number}")
    candidate = seed
    while True:
        maze = Maze.generate(level.size, candidate, LOOPS)
        if level.shortest[0] <= maze.shortest_distance() <= level.shortest[1]:
            if wall_followers_fail(maze, 2 * level.size**2):
                cost = explore_cost(maze)
                if level.explore[0] <= cost <= level.explore[1]:
                    return maze, candidate, cost
        candidate = rng.getrandbits(63)
