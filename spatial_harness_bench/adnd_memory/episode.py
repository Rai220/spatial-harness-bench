"""Self-contained port of AD&DBench's memory-navigation-episodes-0.1.0.

The original stream domain and fixed-probes version are deliberately retained
so an identical level and seed produce the same encounter sequence.
"""
from __future__ import annotations

import hashlib
import json

VERSION = 'memory-navigation-episodes-0.1.0'
STREAM_VERSION = 'fixed-probes-0.1.0'
LEVELS = ('easy', 'medium', 'hard')
DISTRACTORS = {'easy': 4, 'medium': 12, 'hard': 32}
ITEMS = ('янтарный ключ', 'серебряный кубок', 'медный компас', 'нефритовый амулет')
INSTRUCTIONS = ('Вы получаете наблюдение каждого хода, а не всю историю заново. '
    'На первой стоянке названо главное поручение. На остальных встречаются другие поручения. '
    'Двигайтесь действием {"type":"move","direction":"forward"}; на воротах '
    'ответьте {"type":"answer","item":"название предмета"}. Выберите предмет главного поручения. '
    'Только один JSON-объект за ход.')


class _Stream:
    def __init__(self, seed: str, domain: str):
        self.seed, self.domain, self.counter = seed, domain, 0

    def below(self, n: int) -> int:
        if n < 1:
            raise ValueError('empty random range')
        limit = (1 << 256) - (1 << 256) % n
        while True:
            payload = json.dumps([STREAM_VERSION, self.seed, self.domain, self.counter],
                                 ensure_ascii=False, separators=(',', ':'))
            self.counter += 1
            value = int.from_bytes(hashlib.sha256(payload.encode('utf-8')).digest(), 'big')
            if value < limit:
                return value % n

    def shuffle(self, values):
        for i in range(len(values) - 1, 0, -1):
            j = self.below(i + 1)
            values[i], values[j] = values[j], values[i]


class MemoryGame:
    def __init__(self, seed, level):
        if not isinstance(seed, (str, int)) or isinstance(seed, bool) or level not in LEVELS:
            raise ValueError('invalid seed or level')
        self.seed, self.level = str(seed), level
        self.config = {'vector': 'memory', 'level': level, 'distractors': DISTRACTORS[level]}
        rng = _Stream(self.seed, f'{VERSION}:memory:{level}')
        people = list(range(1, DISTRACTORS[level] + 2))
        rng.shuffle(people)
        self.target_person = f'Путник {people[0]:02d}'
        self.answer = ITEMS[rng.below(len(ITEMS))]
        self.records = [{'person': self.target_person, 'item': self.answer, 'main': True}]
        for number in people[1:]:
            self.records.append({'person': f'Путник {number:02d}',
                                 'item': ITEMS[rng.below(len(ITEMS))], 'main': False})
        self.index, self.turn, self.invalid_actions = 0, 0, 0
        self.max_turns = len(self.records) + 4
        self.done, self.terminal_reason, self.message = False, None, ''

    def observe(self):
        obs = {'scenario': 'memory', 'turn': self.turn, 'max_turns': self.max_turns,
               'done': self.done, 'message': self.message}
        if self.index < len(self.records):
            rec = self.records[self.index]
            obs['location'] = 'Стоянка'
            obs['encounter'] = {'person': rec['person'], 'request': rec['item'], 'main': rec['main']}
            obs['exits'] = ['forward']
        else:
            obs.update(location='Ворота', query={'person': self.target_person,
                'question': 'Какой предмет просил этот путник в главном поручении?',
                'choices': list(ITEMS)}, exits=[])
        return obs

    def step(self, action):
        if self.done:
            raise RuntimeError('game is over')
        self.turn += 1
        valid = False
        if self.index < len(self.records) and action == {'type': 'move', 'direction': 'forward'}:
            self.index += 1
            valid, self.message = True, 'Вы прошли к следующей стоянке.'
        elif (self.index == len(self.records) and isinstance(action, dict)
              and set(action) == {'type', 'item'} and action['type'] == 'answer'
              and action['item'] in ITEMS):
            valid = True
            self.done = True
            self.terminal_reason = 'completed' if action['item'] == self.answer else 'incorrect_answer'
            self.message = 'Ответ принят.'
        else:
            self.message = 'Недопустимое действие.'
        if not valid:
            self.invalid_actions += 1
        if not self.done and self.turn >= self.max_turns:
            self.done, self.terminal_reason = True, 'turn_limit'
        return self.observe()

    def result(self):
        # Never send this evaluator-only object to a model or publish it as-is.
        return {'engine_version': VERSION, 'seed': self.seed, 'config': dict(self.config),
                'turns': self.turn, 'terminal_reason': self.terminal_reason,
                'completed': self.terminal_reason == 'completed', 'invalid_actions': self.invalid_actions}
