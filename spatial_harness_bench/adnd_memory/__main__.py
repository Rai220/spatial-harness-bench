"""Run the AD&DBench quest-memory episode through a JSON-lines adapter.

Only each current observation and public instructions go to the adapter.
Seed, private answer and historical records are never sent to the adapter.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import secrets
import select
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory, TemporaryFile
from time import monotonic

from ..gateway import GatewayError, gateway_failure, retry, tail
from .episode import INSTRUCTIONS, LEVELS, VERSION, MemoryGame

POLICIES = ('full', 'window', 'notes')
MAX_RESPONSES_PER_TURN = 32
MAX_PROTOCOL_LINE = 1_048_576


class AdapterProtocolError(RuntimeError):
    """A stable wire error code without untrusted output or private game data."""


def _write_line(stream, value, timeout):
    """Bound the entire write, including an adapter that never reads stdin."""
    data = (json.dumps(value, ensure_ascii=False) + '\n').encode('utf8')
    view = memoryview(data)
    fd, deadline = stream.fileno(), monotonic() + timeout
    os.set_blocking(fd, False)
    while view:
        remaining = deadline - monotonic()
        if remaining <= 0 or not select.select([], [fd], [], remaining)[1]:
            raise TimeoutError('adapter write timed out')
        try:
            n = os.write(fd, view)
        except BlockingIOError:
            continue
        if n <= 0:
            raise BrokenPipeError('adapter stdin closed')
        view = view[n:]


class _LineReader:
    def __init__(self, stream):
        self.fd = stream.fileno()
        self.buffer = b''
        os.set_blocking(self.fd, False)

    def read(self, timeout):
        deadline = monotonic() + timeout
        while b'\n' not in self.buffer:
            if len(self.buffer) > MAX_PROTOCOL_LINE:
                raise AdapterProtocolError('line_too_long')
            remaining = deadline - monotonic()
            if remaining <= 0 or not select.select([self.fd], [], [], remaining)[0]:
                raise TimeoutError('adapter read timed out')
            try:
                data = os.read(self.fd, 65536)
            except BlockingIOError:
                continue
            if not data:
                if self.buffer:
                    raise AdapterProtocolError('incomplete_line')
                return ''
            self.buffer += data
        line, self.buffer = self.buffer.split(b'\n', 1)
        if len(line) > MAX_PROTOCOL_LINE:
            raise AdapterProtocolError('line_too_long')
        try:
            return line.decode('utf8')
        except UnicodeDecodeError:
            raise AdapterProtocolError('invalid_utf8') from None


def _write_private(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w', encoding='utf8') as out:
        json.dump(value, out, ensure_ascii=False, indent=2)
        out.write('\n')


def _attempt(seed, level, policy, window_turns, command, io_timeout):
    game, trace = MemoryGame(seed, level), []
    with TemporaryDirectory(prefix='adnd-memory-notes-') as work, TemporaryFile() as stderr:
        agent = None
        try:
            child_env = {k: v for k, v in os.environ.items() if k not in ('EVAL_SEED', 'ADND_MEMORY_SEED')
                         and (len(str(seed)) < 8 or v != str(seed))}
            agent = subprocess.Popen(command, cwd=work, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=stderr, start_new_session=True, env=child_env)
            reader = _LineReader(agent.stdout)
            while not game.done:
                observation = game.observe()
                envelope = {'type': 'observation', 'turn': game.turn, 'observation': observation}
                if game.turn == 0:
                    envelope.update(instructions=INSTRUCTIONS, policy=policy, window_turns=window_turns,
                                    max_responses_per_turn=MAX_RESPONSES_PER_TURN)
                try:
                    _write_line(agent.stdin, envelope, io_timeout)
                except BrokenPipeError:
                    agent.wait(timeout=min(5, io_timeout))
                    if gateway_failure(agent.returncode, tail(stderr)):
                        raise GatewayError('inference gateway failed')
                    raise AdapterProtocolError('stdin_closed') from None
                line = reader.read(io_timeout)
                if not line:
                    # The adapter ended early; gateway errors retain their separate status.
                    try:
                        agent.wait(timeout=min(5, io_timeout))
                    except subprocess.TimeoutExpired:
                        raise AdapterProtocolError('stdout_closed') from None
                    if gateway_failure(agent.returncode, tail(stderr)):
                        raise GatewayError('inference gateway failed')
                    raise AdapterProtocolError('early_exit')
                try:
                    response = json.loads(line)
                except ValueError:
                    # A model's invalid action consumes a game turn; malformed
                    # wire protocol itself is an adapter error, not a game loss.
                    raise AdapterProtocolError('invalid_json') from None
                if isinstance(response, dict) and response.get('type') == 'gateway_error':
                    failure = GatewayError('inference gateway failed')
                    if response.get('exhausted') is True:
                        failure.exhausted = True
                        failure.attempts = int(response.get('attempts', 1))
                    raise failure
                if isinstance(response, dict) and response.get('type') == 'output_limit':
                    # A truncated generation is a technical exclusion under
                    # the AD&DBench protocol, not an incorrect game action.
                    return {'status': 'error', 'reason': 'output_limit', 'score': None,
                            'turns': game.turn, 'invalid_actions': game.invalid_actions}, trace
                if isinstance(response, dict) and response.get('type') == 'response_limit':
                    if response.get('responses') != MAX_RESPONSES_PER_TURN:
                        raise AdapterProtocolError('invalid_response_limit')
                    return {'status': 'limit', 'reason': 'response_limit', 'score': 0,
                            'turns': game.turn, 'invalid_actions': game.invalid_actions}, trace
                if not isinstance(response, dict) or set(response) != {'action'}:
                    raise AdapterProtocolError('invalid_action_envelope')
                action = response['action']
                after = game.step(action)
                trace.append({'observation': observation, 'action': action, 'next_observation': after})
            try:
                _write_line(agent.stdin, {'type': 'end', 'status': game.terminal_reason}, io_timeout)
                agent.stdin.close()
                if agent.wait(timeout=min(15, io_timeout)):
                    if gateway_failure(agent.returncode, tail(stderr)):
                        raise GatewayError('inference gateway failed')
                    raise AdapterProtocolError('shutdown_failed')
            except BrokenPipeError:
                raise AdapterProtocolError('shutdown_pipe_closed') from None
            outcome = game.result()
            return {'status': outcome['terminal_reason'], 'reason': None,
                    'score': int(outcome['completed']), 'turns': outcome['turns'],
                    'invalid_actions': outcome['invalid_actions']}, trace
        finally:
            if agent is not None:
                # The adapter starts a fresh session; clean up its descendants as
                # well, even if the direct child exited before them.
                try:
                    os.killpg(agent.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                agent.wait()
                for pipe in (agent.stdin, agent.stdout):
                    if pipe is not None and not pipe.closed:
                        try:
                            pipe.close()
                        except (OSError, BrokenPipeError):
                            pass


def run(command, output, *, seed, level='medium', policy='full', window_turns=4,
        save_private=False, io_timeout=300):
    if not command or level not in LEVELS or policy not in POLICIES:
        raise ValueError('invalid command, level or policy')
    if type(window_turns) is not int or window_turns < 1:
        raise ValueError('window_turns must be positive')
    if type(io_timeout) not in (int, float) or not 0 < io_timeout <= 86400:
        raise ValueError('io_timeout must be positive')
    output = Path(output)
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    try:
        # Each retry has a fresh process and empty note directory and replays the
        # same seed. An exhausted transport failure is never treated as a loss.
        result, trace = retry(lambda: _attempt(seed, level, policy, window_turns, command, io_timeout))
    except GatewayError as exc:
        result, trace = {'status': 'error', 'reason': 'gateway', 'score': None,
                         'gateway_attempts': exc.attempts}, []
    except (TimeoutError, subprocess.TimeoutExpired):
        result, trace = {'status': 'error', 'reason': 'timeout', 'score': None}, []
    except AdapterProtocolError as exc:
        result, trace = {'status': 'error', 'reason': 'harness',
                         'adapter_error': str(exc), 'score': None}, []
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
        result, trace = {'status': 'error', 'reason': 'harness', 'score': None}, []
    outcome = result.pop('status')
    if outcome == 'error':
        result.update(status='error', outcome=None, items=[])
    else:
        # The existing stats collector recognizes complete score + items; keep
        # game outcome separate so a turn limit is not mistaken for success.
        result.update(status='complete', outcome=outcome, items=[{'points': result['score']}])
    # stats.rows groups by task ID and has no policy dimension for ordinary
    # reports, so separate levels and policies to avoid silently pooling them.
    task = f'adnd-memory-road-{VERSION}-{level}-{policy}'
    if policy != 'full':
        task += f'-w{window_turns}'
    result.update(task=task, task_base='adnd-memory-road',
                  version=VERSION, level=level, policy=policy,
                  window_turns=window_turns if policy != 'full' else None, max_score=1,
                  notes_allowed=policy == 'notes', io_timeout_seconds=io_timeout,
                  policy_enforcement='adapter' if command else None)
    _write_private(output / 'result.json', result)
    if save_private:
        _write_private(output / 'private-run.json', {'seed': str(seed), 'trace': trace})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--level', choices=LEVELS, default='medium')
    parser.add_argument('--policy', choices=POLICIES, default='full')
    parser.add_argument('--window-turns', type=int, default=4)
    parser.add_argument('--io-timeout', type=float, default=300,
                        help='maximum seconds per adapter read or write (default: 300)')
    parser.add_argument('--seed', help='private seed; generated when omitted')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--save-private', action='store_true', help='save seed and trace in a private file')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('Укажите команду адаптера после --')
    if args.seed is None and not args.save_private:
        parser.error('Укажите приватный --seed или --save-private для воспроизводимости')
    try:
        result = run(command, args.output, seed=args.seed if args.seed is not None else secrets.token_hex(16),
                     level=args.level, policy=args.policy, window_turns=args.window_turns,
                     save_private=args.save_private, io_timeout=args.io_timeout)
    except (OSError, ValueError):
        print('Ошибка настройки или выходного каталога.', file=sys.stderr)
        return 1
    print(f"{result.get('outcome') or result['status']}: {result['score'] if result['score'] is not None else 'нет оценки'}")
    return int(result['status'] == 'error')


if __name__ == '__main__':
    sys.exit(main())
