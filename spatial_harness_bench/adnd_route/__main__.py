"""Evaluate four independently generated AD&DBench weighted routes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
from tempfile import TemporaryDirectory

from ..gateway import GatewayError, run_stage
from .task import FIXED_PROBES_V3_LITE, _strict_object, check_response, generate_routes, public_probe

TASK = 'adnd-route-v1'
MAX_ANSWER_BYTES = 1_000_000


def _private_write(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as handle:
        handle.write(data)


def _format_valid(probe, raw: str) -> bool:
    try:
        data = json.loads(raw, object_pairs_hook=_strict_object)
    except (ValueError, TypeError, RecursionError):
        return False
    return (isinstance(data, dict) and set(data) == {'id', 'answer'}
            and isinstance(data['id'], str) and data['id'] == probe['id']
            and isinstance(data['answer'], str))


def run(command: list[str], output: Path, seed: str) -> dict:
    """Run four fresh sessions; persist private state and separate errors."""
    if not command or not isinstance(seed, str) or not seed:
        raise ValueError('a command and a non-empty seed are required')
    probes = generate_routes(seed)
    publics = [public_probe(p) for p in probes]
    dataset = json.dumps(publics, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf8')
    source = Path(__file__).with_name('task.py').read_bytes()
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    _private_write(output / 'seed.txt', (seed + '\n').encode('utf8'))
    items = []
    for probe, public in zip(probes, publics):
        task_dir = output / probe['id']
        task_dir.mkdir(mode=0o700)
        question = ('Ответьте на задачу. Запишите ответ в файл answer.json в текущей папке. '
                    'Формат — ровно один JSON-объект с полями id и answer (строки).\n'
                    + json.dumps(public, ensure_ascii=False, sort_keys=True))
        _private_write(task_dir / 'prompt.txt', question.encode('utf8'))
        item = {'task': probe['id'], 'prompt_sha256': hashlib.sha256(question.encode('utf8')).hexdigest(),
                'public_sha256': hashlib.sha256(json.dumps(public, ensure_ascii=False, sort_keys=True,
                                            separators=(',', ':')).encode('utf8')).hexdigest()}
        with TemporaryDirectory(prefix='adnd-route-') as temporary:
            work = Path(temporary)
            try:
                run_stage(command, work, question, env=os.environ.copy())
            except GatewayError as exc:
                item.update(exc.report('solve'), points=None)
            except (OSError, subprocess.CalledProcessError):
                item.update(status='error', reason='harness', points=None)
            else:
                answer = work / 'answer.json'
                try:
                    if answer.is_file() and not answer.is_symlink():
                        with answer.open('rb') as handle:
                            data = handle.read(MAX_ANSWER_BYTES + 1)
                    else:
                        data = b''
                    if len(data) > MAX_ANSWER_BYTES:
                        data = b''
                    _private_write(task_dir / 'answer.json', data)
                except OSError:
                    item.update(status='error', reason='artifact', points=None)
                else:
                    raw = data.decode('utf8', errors='replace')
                    valid = _format_valid(probe, raw)
                    correct = check_response(probe, raw)
                    item.update(status='complete', points=int(correct), format_valid=valid,
                                semantic_correct=correct if valid else None)
        items.append(item)
    complete = all(item['status'] == 'complete' for item in items)
    report = {'task': TASK, 'version': FIXED_PROBES_V3_LITE,
              'status': 'complete' if complete else 'error',
              'score': sum(item['points'] for item in items) if complete else None,
              'max_score': len(probes), 'items': items,
              'dataset_sha256': hashlib.sha256(dataset).hexdigest(),
              'generator_sha256': hashlib.sha256(source).hexdigest(),
              'note': 'Four questions share a seed and are not independent worlds.'}
    _private_write(output / 'result.json', (json.dumps(report, ensure_ascii=False, indent=2) + '\n').encode('utf8'))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--seed', help='development or shared seed; use --seed-file to avoid process-list exposure')
    group.add_argument('--seed-file', type=Path, help='private file containing the seed')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('Укажите команду харнеса после --')
    try:
        seed = args.seed if args.seed is not None else (
            args.seed_file.read_text(encoding='utf8').strip() if args.seed_file else secrets.token_hex(16))
        result = run(command, args.output, seed)
    except (OSError, ValueError) as exc:
        print(f'Ошибка запуска: {type(exc).__name__}', file=sys.stderr)
        return 1
    print(f"{result['task']}: {result['score'] if result['score'] is not None else 'error'}/{result['max_score']}")
    return int(result['status'] != 'complete')


if __name__ == '__main__':
    raise SystemExit(main())
