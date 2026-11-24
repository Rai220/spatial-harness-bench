"""AD&DBench v3-lite.2 weighted route tasks, with private ground truth.

The route states and public payloads are bit-compatible with the route subset
of fixed-probes-0.3-lite.2. Never send private probes or seed to a model.
"""
from __future__ import annotations
from collections import deque
import copy
import hashlib
import heapq
import json
import unicodedata
from typing import Any, Mapping

VERSION = 'fixed-probes-0.1.0'
DIFFICULTIES = ('easy', 'medium', 'hard')
_DIRECTIONS = ('north', 'east', 'south', 'west')
_VECTORS = {'north': (0, -1), 'east': (1, 0), 'south': (0, 1), 'west': (-1, 0)}
_OPPOSITE = {'north': 'south', 'south': 'north', 'east': 'west', 'west': 'east'}
_ROOM_NAMES = ('Янтарный зал', 'Медная галерея', 'Лунная кузница', 'Башня туманов',
               'Сад колоколов', 'Красный двор', 'Каменный архив', 'Зал фонарей',
               'Старый колодец', 'Хрустальная лестница', 'Мшистый мост', 'Тихая башня',
               'Серебряная ротонда', 'Забытый склеп', 'Зеркальный зал', 'Звёздный двор')
FIXED_PROBES_V3_LITE = 'fixed-probes-0.3-lite.2'

class _Stream:
    """Stable per-task SHA-256 stream; no dependence on process hash order."""
    def __init__(self, seed: str, domain: str):
        self.seed, self.domain, self.counter = seed, domain, 0

    def below(self, n: int) -> int:
        if n < 1:
            raise ValueError('empty random range')
        limit = (1 << 256) - (1 << 256) % n
        while True:
            payload = json.dumps([VERSION, self.seed, self.domain, self.counter],
                                 ensure_ascii=False, separators=(',', ':'))
            self.counter += 1
            value = int.from_bytes(hashlib.sha256(payload.encode('utf-8')).digest(), 'big')
            if value < limit:
                return value % n

    def shuffle(self, values: list[Any]) -> None:
        for i in range(len(values) - 1, 0, -1):
            j = self.below(i + 1)
            values[i], values[j] = values[j], values[i]


def _probe(probe_id: str, capability: str, difficulty: str, context: dict[str, Any],
           question: str, answer: str) -> dict[str, Any]:
    return {'id': probe_id, 'capability': capability, 'difficulty': difficulty,
            'context': context, 'question': question, 'answer': answer}


def _rng(seed: str, domain: str) -> _Stream:
    return _Stream(seed, f'{FIXED_PROBES_V3_LITE}:{domain}')


def _route(seed: str, index: int, target: str, target_rank: int):
    rng = _rng(seed, f'route:{index}')
    side = 5 + rng.below(2)
    positions = [(x, y) for y in range(side) for x in range(side)]
    for _ in range(5000):
        edges = {p: {} for p in positions}
        visited, stack = {positions[0]}, [positions[0]]
        while stack:
            x, y = stack[-1]
            candidates = [(d, (x + dx, y + dy)) for d, (dx, dy) in _VECTORS.items()
                          if (x + dx, y + dy) in edges and (x + dx, y + dy) not in visited]
            if not candidates:
                stack.pop()
                continue
            direction, other = candidates[rng.below(len(candidates))]
            cost = 1 + rng.below(9)
            edges[(x, y)][direction] = (other, cost)
            edges[other][_OPPOSITE[direction]] = ((x, y), cost)
            visited.add(other)
            stack.append(other)
        missing = [(p, direction, (p[0] + dx, p[1] + dy))
                   for p in positions for direction, (dx, dy) in _VECTORS.items()
                   if direction in ('east', 'south') and (p[0] + dx, p[1] + dy) in edges
                   and direction not in edges[p]]
        rng.shuffle(missing)
        for p, direction, other in missing[:max(2, len(missing) * 2 // 3)]:
            cost = 1 + rng.below(9)
            edges[p][direction] = (other, cost)
            edges[other][_OPPOSITE[direction]] = (p, cost)
        choices = []
        for start in positions:
            if len(edges[start]) != 4 or len({v[1] for v in edges[start].values()}) != 4:
                continue
            ranked = sorted(edges[start], key=lambda d: (edges[start][d][1], d))
            if ranked.index(target) != target_rank:
                continue
            distance, ways, path = {start: 0}, {start: 1}, {start: []}
            queue = [(0, start)]
            while queue:
                cost, pos = heapq.heappop(queue)
                if cost != distance[pos]:
                    continue
                for direction, (nxt, weight) in edges[pos].items():
                    new = cost + weight
                    if new < distance.get(nxt, float('inf')):
                        distance[nxt], ways[nxt], path[nxt] = new, ways[pos], path[pos] + [direction]
                        heapq.heappush(queue, (new, nxt))
                    elif new == distance.get(nxt):
                        ways[nxt] = min(2, ways[nxt] + ways[pos])
            hops, first_steps = {start: 0}, {start: set()}
            pending = deque([start])
            while pending:
                pos = pending.popleft()
                for direction, (nxt, _) in edges[pos].items():
                    first = {direction} if pos == start else first_steps[pos]
                    if nxt not in hops:
                        hops[nxt] = hops[pos] + 1
                        first_steps[nxt] = set(first)
                        pending.append(nxt)
                    elif hops[nxt] == hops[pos] + 1:
                        first_steps[nxt].update(first)
            for end in positions:
                if end != start and ways.get(end) == 1 and 5 <= len(path[end]) <= 8 and path[end][0] == target:
                    # Every unweighted-shortest path must start differently.
                    if target not in first_steps[end]:
                        choices.append((start, end))
        if choices:
            start, end = choices[rng.below(len(choices))]
            labels = list(_ROOM_NAMES) + [f'Локация {i:02d}' for i in range(1, 34)]
            rng.shuffle(labels)
            names = {p: labels[i] for i, p in enumerate(positions)}
            rooms = [{'room': names[p], 'exits': {d: {'room': names[nxt], 'cost': cost}
                      for d, (nxt, cost) in edges[p].items()}} for p in positions]
            rng.shuffle(rooms)
            return _probe(f'v3_lite_spatial_route_{index + 1}', 'spatial_reasoning', 'challenge-lite',
                          {'observed_map': rooms, 'rules': 'Указаны все проходы; каждый обратим и доступен. '
                                                          'Стоимость каждого прохода положительна.'},
                          f'Вы находитесь в «{names[start]}». Какой первый выход ведёт в «{names[end]}» '
                          'путём с наименьшей суммарной стоимостью? '
                          'Ответьте одним из north, east, south, west.', target)
    raise RuntimeError('could not generate v3-lite route satisfying constraints')


def generate_routes(seed: str | int) -> list[dict[str, Any]]:
    """Return the four route questions in their original order for a seed."""
    if not isinstance(seed, (str, int)) or isinstance(seed, bool):
        raise ValueError('seed must be a string or integer')
    value = str(seed)
    routes, ranks = list(_DIRECTIONS), list(range(4))
    _rng(value, 'route:answer-order').shuffle(routes)
    _rng(value, 'route:first-edge-rank-order').shuffle(ranks)
    return [_route(value, index, answer, ranks[index]) for index, answer in enumerate(routes)]


def public_probe(probe: Mapping[str, Any]) -> dict[str, Any]:
    """Copy only model-facing fields, with JSON response protocol and no seed/key."""
    return {'id': probe['id'], 'capability': probe['capability'],
            'difficulty': probe['difficulty'], 'context': copy.deepcopy(probe['context']),
            'question': probe['question'],
            'response_format': 'Верните только JSON-объект с полями id и answer (строки).' }


def _normalize(text: str) -> str:
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().strip().split())


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError('duplicate JSON key')
        output[key] = value
    return output


def check_response(probe: Mapping[str, Any], response: Any) -> bool:
    """Verify exactly {id, answer}; malformed/non-JSON outputs count as wrong.

    Case, Unicode normalization and incidental whitespace are normalized; no
    substring, fuzzy matching, implicit translation, or LLM judge is used.
    """
    if isinstance(response, str):
        try:
            response = json.loads(response, object_pairs_hook=_strict_object)
        except (ValueError, TypeError, RecursionError):
            return False
    if not isinstance(response, Mapping) or set(response) != {'id', 'answer'}:
        return False
    answer = response.get('answer')
    return (response.get('id') == probe.get('id') and isinstance(answer, str)
            and isinstance(probe.get('answer'), str)
            and _normalize(answer) == _normalize(probe['answer']))
