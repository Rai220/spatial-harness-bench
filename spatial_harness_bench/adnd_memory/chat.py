"""OpenAI-compatible JSONL adapter for the AD&D memory episode.

History/window/notes policies are declared by the runner in the first message.
Run this file by absolute path when the package is not installed.
"""
from __future__ import annotations

import argparse
from collections import deque
from http.client import IncompleteRead
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from spatial_harness_bench.gateway import GatewayError, TEMPFAIL, retry
from spatial_harness_bench.adnd_memory.episode import ITEMS


def _tool(name, description, fields):
    return {'type': 'function', 'function': {'name': name, 'description': description,
        'parameters': {'type': 'object', 'properties': {k: {'type': 'string'} for k in fields},
                       'required': list(fields), 'additionalProperties': False}}}


TOOLS = [_tool('read_file', 'Прочитать собственную заметку.', ['name']),
         _tool('write_file', 'Создать или заменить собственную заметку.', ['name', 'text']),
         _tool('append_file', 'Дописать в собственную заметку.', ['name', 'text'])]


def _file_tool(call):
    try:
        args = json.loads(call['arguments'])
        name = args['name']
        if not isinstance(name, str) or not name or Path(name).name != name or name in ('.', '..'):
            return 'Укажите простое имя файла в рабочей папке.'
        path = Path(name)
        if path.is_symlink():
            return 'Файл недоступен.'
        if call['name'] == 'read_file':
            return path.read_text(encoding='utf8')
        if call['name'] not in ('write_file', 'append_file') or not isinstance(args.get('text'), str):
            return 'Неизвестная операция или неверные данные.'
        with path.open('a' if call['name'] == 'append_file' else 'w', encoding='utf8') as file:
            file.write(args['text'])
        return 'Сохранено.'
    except (ValueError, OSError, KeyError, TypeError):
        return 'Ошибка работы с заметкой.'


def _unique(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError('duplicate key')
        obj[key] = value
    return obj


def canonical_action(content):
    try:
        action = json.loads(content, object_pairs_hook=_unique)
        if action == {'type': 'move', 'direction': 'forward'}:
            return action
        if (isinstance(action, dict) and set(action) == {'type', 'item'}
                and action['type'] == 'answer' and isinstance(action['item'], str)
                and action['item'] in ITEMS):
            return action
    except (TypeError, ValueError, RecursionError):
        pass
    return {'type': 'invalid'}


def _endpoint():
    base = os.environ['OPENAI_BASE_URL'].rstrip('/')
    parsed = urlsplit(base)
    if (parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username
            or parsed.password or parsed.query or parsed.fragment
            or (parsed.scheme == 'http' and parsed.hostname not in ('localhost', '127.0.0.1', '::1'))):
        raise ValueError('invalid endpoint')
    return base + '/chat/completions'


def complete(endpoint, key, payload, http_timeout=60):
    headers = {'Content-Type': 'application/json'}
    if key:
        headers['Authorization'] = 'Bearer ' + key
    request = urllib.request.Request(endpoint, data=json.dumps(payload, ensure_ascii=False).encode('utf8'),
                                     headers=headers)
    def send():
        try:
            with urllib.request.urlopen(request, timeout=http_timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            exc.close()
            raise GatewayError('HTTP error') from None
        except (TimeoutError, ConnectionError, IncompleteRead, urllib.error.URLError):
            raise GatewayError('connection error') from None
    return retry(send)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--effort', help='optional provider reasoning_effort')
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--max-tokens', type=int, help='optional output limit per response')
    parser.add_argument('--http-timeout', type=float, default=60, help='HTTP timeout in seconds')
    args = parser.parse_args()
    if not 0 <= args.temperature <= 2 or (args.max_tokens is not None and args.max_tokens < 1) or not 0 < args.http_timeout <= 86400:
        parser.error('invalid generation settings')
    endpoint, key = _endpoint(), os.environ.get('OPENAI_API_KEY')
    history = deque()
    system, policy, window_turns, max_responses = None, None, None, None
    for line in sys.stdin:
        packet = json.loads(line)
        if packet.get('type') == 'end':
            return 0
        if packet.get('type') != 'observation' or not isinstance(packet.get('observation'), dict):
            raise ValueError('invalid observation packet')
        if system is None:
            system, policy = packet['instructions'], packet['policy']
            window_turns, max_responses = packet['window_turns'], packet['max_responses_per_turn']
            if policy not in ('full', 'window', 'notes') or type(window_turns) is not int or window_turns < 1:
                raise ValueError('invalid policy')
            if policy == 'notes':
                system += ' Вы можете читать и записывать собственные заметки предоставленными инструментами.'
            else:
                system += ' Инструменты и внешние заметки в этом режиме недоступны.'
        current = [{'role': 'user', 'content': json.dumps(packet['observation'], ensure_ascii=False, sort_keys=True)}]
        for _ in range(max_responses):
            prior = list(history) if policy == 'full' else list(history)[-(window_turns - 1):] if window_turns > 1 else []
            messages = [{'role': 'system', 'content': system}]
            for turn in prior:
                messages.extend(turn)
            messages.extend(current)
            payload = {'model': args.model, 'messages': messages, 'temperature': args.temperature}
            if args.effort:
                payload['reasoning_effort'] = args.effort
            if args.max_tokens is not None:
                payload['max_tokens'] = args.max_tokens
            if policy == 'notes':
                payload['tools'] = TOOLS
            body = complete(endpoint, key, payload, args.http_timeout)
            choice = body['choices'][0]
            if choice.get('finish_reason') == 'length':
                print(json.dumps({'type': 'output_limit'}), flush=True)
                return 0
            msg = choice['message']
            if not msg.get('tool_calls') and not isinstance(msg.get('content'), str):
                raise ValueError('missing response content')
            assistant = {'role': 'assistant', 'content': msg.get('content') or ''}
            calls = msg.get('tool_calls') or []
            if calls:
                if policy != 'notes':
                    print(json.dumps({'action': {'type': 'invalid'}}), flush=True)
                    current.append({'role': 'assistant', 'content': '{"type":"invalid"}'})
                    break
                assistant['tool_calls'] = calls
                current.append(assistant)
                for call in calls:
                    current.append({'role': 'tool', 'tool_call_id': call['id'],
                                    'content': _file_tool(call['function'])})
                continue
            action = canonical_action(assistant['content'])
            current.append({'role': 'assistant', 'content': json.dumps(action, ensure_ascii=False, sort_keys=True)})
            print(json.dumps({'action': action}, ensure_ascii=False), flush=True)
            break
        else:
            print(json.dumps({'type': 'response_limit', 'responses': max_responses}), flush=True)
            return 0
        history.append(current)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except GatewayError as exc:
        print(json.dumps({'type': 'gateway_error', 'exhausted': True, 'attempts': exc.attempts}), flush=True)
        sys.exit(TEMPFAIL)
    except (ValueError, KeyError, TypeError, IndexError, OSError):
        # Provider response bodies, URLs and keys are deliberately not logged.
        print('Adapter configuration or response error.', file=sys.stderr)
        sys.exit(1)
