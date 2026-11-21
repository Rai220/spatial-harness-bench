from collections import deque
from dataclasses import dataclass, field
import random


# A room's four entries are always in this order. Opposite directions have
# indices that differ by two, so each interior door can be set symmetrically.
DIRECTIONS = ("вперёд", "вправо", "назад", "влево")
DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))
OPEN, CLOSED, WALL = "открыта", "закрыта", "нет двери"
MIN_SIZE, MAX_SIZE = 5, 15


@dataclass
class Maze:
    size: int
    rooms: list[list[str]]
    start: int
    exit: int

    def neighbor(self, cell: int, direction: int) -> int | None:
        row, col = divmod(cell, self.size)
        dr, dc = DELTAS[direction]
        row, col = row + dr, col + dc
        if 0 <= row < self.size and 0 <= col < self.size:
            return row * self.size + col
        return None

    @classmethod
    def generate(cls, size: int, seed: int, loops: float = 0.0):
        """A random spanning tree; then `loops` of the remaining interior walls become open doors.

        With loops=0 the maze is a tree (version 1). Loops use their own random stream, so the
        tree and the closed doors of a seed do not depend on the loop density.
        """
        if not MIN_SIZE <= size <= MAX_SIZE:
            raise ValueError(f"Размер стороны должен быть от {MIN_SIZE} до {MAX_SIZE}.")
        rng = random.Random(seed)
        count = size * size
        start, goal = rng.sample(range(count), 2)
        maze = cls(size, [[WALL] * 4 for _ in range(count)], start, goal)

        # Grow a randomized spanning tree from an independent root. Every
        # new room receives an open door back to the connected component.
        stack = [rng.randrange(count)]
        seen = set(stack)
        while stack:
            cell = stack[-1]
            choices = []
            for d in range(4):
                other = maze.neighbor(cell, d)
                if other is not None and other not in seen:
                    choices.append((d, other))
            if not choices:
                stack.pop()
                continue
            d, other = rng.choice(choices)
            maze.rooms[cell][d] = maze.rooms[other][(d + 2) % 4] = OPEN
            seen.add(other)
            stack.append(other)

        for cell, doors in enumerate(maze.rooms):
            for d in (1, 2):  # visit each interior boundary just once
                other = maze.neighbor(cell, d)
                if other is not None and doors[d] == WALL and rng.choice((True, False)):
                    doors[d] = maze.rooms[other][(d + 2) % 4] = CLOSED
        if not 0 <= loops <= 1:
            raise ValueError("Доля петель должна быть от 0 до 1.")
        if loops:
            blocked = [(cell, d) for cell in range(count) for d in (1, 2)
                       if maze.neighbor(cell, d) is not None and maze.rooms[cell][d] != OPEN]
            extra = random.Random(f"loops-{seed}")
            for cell, d in extra.sample(blocked, round(loops * len(blocked))):
                maze.rooms[cell][d] = maze.rooms[maze.neighbor(cell, d)][(d + 2) % 4] = OPEN
        return maze

    def shortest_distance(self) -> int:
        queue = deque([(self.start, 0)])
        seen = {self.start}
        while queue:
            cell, distance = queue.popleft()
            if cell == self.exit:
                return distance
            for d, state in enumerate(self.rooms[cell]):
                if state == OPEN:
                    other = self.neighbor(cell, d)
                    if other is not None and other not in seen:
                        seen.add(other)
                        queue.append((other, distance + 1))
        raise ValueError("Выход недостижим.")

    def render(self) -> str:
        side = 2 * self.size + 1
        grid = [["#"] * side for _ in range(side)]
        for cell, doors in enumerate(self.rooms):
            row, col = divmod(cell, self.size)
            y, x = 2 * row + 1, 2 * col + 1
            grid[y][x] = "S" if cell == self.start else "E" if cell == self.exit else " "
            for (dy, dx), state in zip(DELTAS, doors):
                grid[y + dy][x + dx] = {OPEN: " ", CLOSED: "+", WALL: "#"}[state]
        return "\n".join("".join(row) for row in grid) + "\n"


@dataclass
class Walk:
    maze: Maze
    limit: int
    cell: int = field(init=False)
    attempts: int = 0

    def __post_init__(self):
        if self.limit < 1:
            raise ValueError("Лимит ходов должен быть положительным.")
        self.cell = self.maze.start

    @property
    def status(self):
        if self.cell == self.maze.exit:
            return "solved"
        if self.attempts >= self.limit:
            return "limit"
        return "running"

    def observation(self) -> str:
        doors = self.maze.rooms[self.cell]
        return "Двери в текущей комнате. " + ", ".join(
            f"{name} — {state}" for name, state in zip(DIRECTIONS, doors)
        ) + "."

    def step(self, action: str) -> str:
        if self.status != "running":
            raise ValueError("Попытка уже завершена.")
        self.attempts += 1
        action = action.strip().casefold().replace("вперед", "вперёд")
        if action not in DIRECTIONS:
            return "Неизвестное направление. Вы остались на месте."
        d = DIRECTIONS.index(action)
        if self.maze.rooms[self.cell][d] != OPEN:
            return "Пройти нельзя. Вы остались на месте."
        other = self.maze.neighbor(self.cell, d)
        assert other is not None
        self.cell = other
        return "Вы нашли выход!" if self.cell == self.maze.exit else "Вы перешли в соседнюю комнату."


def instructions(size: int, limit: int, *, loops: bool = False) -> str:
    return (
        f"Найди выход из лабиринта {size}×{size}. Старт и выход могут находиться внутри. "
        + ("В лабиринте есть петли: в одну комнату можно прийти разными путями. " if loops else "") +
        "Ты видишь только четыре стороны текущей комнаты. Пройти можно через открытую дверь. "
        "Закрытые двери остаются закрытыми. Направления фиксированы на всей карте: "
        "вперёд — север, назад — юг, вправо — восток, влево — запад. Поворотов нет. "
        "На каждом ходу выбери одно направление и дождись следующего описания. "
        "Можно пользоваться памятью, вести историю или рисовать свою карту в файлах. "
        f"При входе в комнату с выходом попытка завершится. Лимит — {limit} решений."
    )
