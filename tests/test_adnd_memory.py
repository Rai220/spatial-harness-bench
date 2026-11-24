import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from spatial_harness_bench.adnd_memory import chat
from spatial_harness_bench.adnd_memory.__main__ import run, _write_line
from spatial_harness_bench.adnd_memory.episode import DISTRACTORS, ITEMS, MemoryGame, VERSION
from spatial_harness_bench.stats import collect, rows


class EpisodeTests(unittest.TestCase):
    def test_port_matches_original_on_public_development_vectors(self):
        expected = {'easy': ('Путник 05', 'медный компас', 5, 9),
                    'medium': ('Путник 03', 'серебряный кубок', 13, 17),
                    'hard': ('Путник 13', 'медный компас', 33, 37)}
        for level, sample in expected.items():
            game = MemoryGame('dev-compat-v1', level)
            self.assertEqual((game.target_person, game.answer, len(game.records), game.max_turns), sample)

    def test_original_observations_success_wrong_and_invalid(self):
        for level in DISTRACTORS:
            for seed in range(20):
                game = MemoryGame(seed, level)
                first = game.observe()
                self.assertTrue(first['encounter']['main'])
                person, target = first['encounter']['person'], first['encounter']['request']
                for _ in range(DISTRACTORS[level]):
                    obs = game.step({'type': 'move', 'direction': 'forward'})
                    self.assertFalse(obs['encounter']['main'])
                    self.assertNotEqual(obs['encounter']['person'], person)
                gate = game.step({'type': 'move', 'direction': 'forward'})
                self.assertEqual(gate['query']['person'], person)
                self.assertNotIn('encounter', gate)
                self.assertEqual(gate['query']['choices'], list(ITEMS))
                game.step({'type': 'answer', 'item': target})
                self.assertEqual(game.result()['terminal_reason'], 'completed')
                with self.assertRaises(RuntimeError):
                    game.step({'type': 'move', 'direction': 'forward'})
        game = MemoryGame('dev', 'easy')
        target = game.observe()['encounter']['request']
        game.step({'type': 'move', 'direction': 'wrong'})
        self.assertEqual(game.invalid_actions, 1)
        for _ in range(DISTRACTORS['easy'] + 1):
            game.step({'type': 'move', 'direction': 'forward'})
        game.step({'type': 'answer', 'item': next(x for x in ITEMS if x != target)})
        self.assertEqual(game.terminal_reason, 'incorrect_answer')

    def test_seed_never_in_public_observation_and_repeatable(self):
        seed = 'development-private-marker-6732'
        a, b = MemoryGame(seed, 'hard'), MemoryGame(seed, 'hard')
        self.assertEqual(a.records, b.records)
        for _ in range(DISTRACTORS['hard'] + 2):
            self.assertNotIn(seed, json.dumps(a.observe(), ensure_ascii=False))
            if a.index < len(a.records):
                a.step({'type': 'move', 'direction': 'forward'})
        for bad in (None, True, [], {}):
            with self.assertRaises(ValueError):
                MemoryGame(bad, 'easy')


class RunnerTests(unittest.TestCase):
    def test_jsonl_episode_private_boundary_and_recording(self):
        # One persistent mock adapter sees the first encounter and returns it
        # at the gate. It has no access to the episode's hidden game object.
        agent = r'''import json,sys
saved=None
for line in sys.stdin:
 p=json.loads(line)
 if p['type']=='end': break
 o=p['observation']
 if o.get('encounter',{}).get('main'): saved=o['encounter']['request']
 a={'type':'answer','item':saved} if o['location']=='Ворота' else {'type':'move','direction':'forward'}
 print(json.dumps({'action':a},ensure_ascii=False),flush=True)
'''
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'episode'
            result = run([sys.executable, '-c', agent], out, seed='dev-compat-v1', level='medium',
                         policy='full', save_private=True)
            self.assertEqual((result['status'], result['outcome'], result['score'], result['turns']), ('complete', 'completed', 1, 14))
            self.assertNotIn('dev-compat-v1', json.dumps(result))
            self.assertEqual(json.loads((out/'private-run.json').read_text())['seed'], 'dev-compat-v1')
            self.assertEqual((out.stat().st_mode & 0o777), 0o700)
            self.assertEqual(((out/'result.json').stat().st_mode & 0o777), 0o600)
            stat = list(rows(result))
            self.assertEqual(stat[0][:6], (f'adnd-memory-road-{VERSION}-medium-full', '-', 1, 1, 1, 1))
            self.assertEqual(collect([f'model={out}'])[('model', f'adnd-memory-road-{VERSION}-medium-full', '-')].scores, [1])
            with self.assertRaises(FileExistsError):
                run([sys.executable, '-c', agent], out, seed='dev', level='easy')

    def test_gateway_failure_is_not_loss_and_empty_action_is_a_game_turn(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, SPATIAL_GATEWAY_RETRIES='0'):
            result = run([sys.executable, '-c', 'raise SystemExit(75)'], Path(tmp)/'gateway',
                         seed='dev', level='easy')
            self.assertEqual((result['status'], result['reason'], result['score']), ('error','gateway',None))
            agent = "import sys,json\nfor line in sys.stdin:\n p=json.loads(line)\n if p['type']=='end':break\n print(json.dumps({'action':{'type':'invalid'}}),flush=True)"
            result = run([sys.executable, '-c', agent], Path(tmp)/'invalid', seed='dev', level='easy')
            self.assertEqual((result['status'], result['outcome'], result['score'], result['invalid_actions']), ('complete', 'turn_limit', 0, 9))
            self.assertEqual(list(rows(result))[0][:6], (f'adnd-memory-road-{VERSION}-easy-full', '-', 0, 1, 0, 1))
            self.assertEqual(collect([f'model={Path(tmp)/"gateway"}'])[('model',f'adnd-memory-road-{VERSION}-easy-full','-')].errors, 1)
            self.assertFalse((Path(tmp)/'invalid'/'private-run.json').exists())

    def test_timeout_and_output_limit_are_technical(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run([sys.executable, '-c', 'import time; time.sleep(60)'], Path(tmp)/'timeout',
                         seed='dev', level='easy', io_timeout=0.15)
            self.assertEqual((result['status'], result['reason'], result['score']), ('error', 'timeout', None))
            agent = "import sys,json\nfor line in sys.stdin:\n print(json.dumps({'type':'output_limit'}),flush=True)"
            result = run([sys.executable, '-c', agent], Path(tmp)/'limit', seed='dev', level='easy')
            self.assertEqual((result['status'], result['reason'], result['score']), ('error', 'output_limit', None))
            self.assertEqual(collect([f'model={Path(tmp)/"limit"}'])[("model", result['task'], '-')].errors, 1)

    def test_blocked_write_has_deadline(self):
        read_fd, write_fd = os.pipe()
        try:
            os.set_blocking(write_fd, False)
            while True:
                try:
                    os.write(write_fd, b'x' * 4096)
                except BlockingIOError:
                    break
            with os.fdopen(write_fd, 'wb', buffering=0) as stream:
                write_fd = None
                with self.assertRaises(TimeoutError):
                    _write_line(stream, {'x': 'x'}, 0.05)
        finally:
            os.close(read_fd)
            if write_fd is not None:
                os.close(write_fd)

    def test_window_settings_do_not_pool(self):
        agent = "import sys,json\nfor line in sys.stdin:\n p=json.loads(line)\n if p['type']=='end':break\n print(json.dumps({'action':{'type':'invalid'}}),flush=True)"
        with tempfile.TemporaryDirectory() as tmp:
            a = run([sys.executable, '-c', agent], Path(tmp)/'a', seed='dev', level='easy', policy='window', window_turns=2)
            b = run([sys.executable, '-c', agent], Path(tmp)/'b', seed='dev', level='easy', policy='window', window_turns=8)
            self.assertNotEqual(a['task'], b['task'])
            self.assertEqual(len(collect([f'model={Path(tmp)/"a"}', f'model={Path(tmp)/"b"}'])), 2)


class ChatPolicyTests(unittest.TestCase):
    def test_truncated_response_and_http_timeout(self):
        packet = {'type':'observation', 'turn':0, 'observation':{}, 'instructions':'RULES',
                  'policy':'full', 'window_turns':1, 'max_responses_per_turn':32}
        output = io.StringIO()
        with patch.object(chat, 'complete', return_value={'choices':[{'finish_reason':'length','message':{'content':None}}]}) as request, \
                patch.object(sys, 'argv', ['chat','--model','mock']), \
                patch.dict(os.environ, {'OPENAI_BASE_URL':'https://example.com/v1'}, clear=False), \
                patch.object(sys, 'stdin', io.StringIO(json.dumps(packet))), contextlib.redirect_stdout(output):
            self.assertEqual(chat.main(), 0)
            self.assertEqual(json.loads(output.getvalue()), {'type':'output_limit'})
            self.assertEqual(request.call_count, 1)
        with patch.object(chat.urllib.request, 'urlopen', return_value=io.BytesIO(b'{}')) as request:
            self.assertEqual(chat.complete('https://example.com', '', {}, 7), {})
            self.assertEqual(request.call_args.kwargs['timeout'], 7)

    def test_strict_action_parser(self):
        self.assertEqual(chat.canonical_action('{"type":"move","direction":"forward"}'),
                         {'type':'move','direction':'forward'})
        for text in ('', '{"type":"move","direction":"forward","note":"x"}',
                     '{"type":"move","type":"move","direction":"forward"}',
                     '{"type":"answer","item":"unknown"}'):
            self.assertEqual(chat.canonical_action(text), {'type':'invalid'})

    def _adapter_requests(self, policy, window=2):
        g = MemoryGame('dev-compat-v1', 'easy')
        packets = []
        for turn in range(4):
            obj = {'type':'observation','turn':turn,'observation':g.observe()}
            if turn == 0:
                obj.update(instructions='RULES', policy=policy, window_turns=window, max_responses_per_turn=32)
            packets.append(json.dumps(obj, ensure_ascii=False))
            g.step({'type':'move','direction':'forward'})
        packets.append('{"type":"end"}')
        calls = []
        def fake(endpoint, key, payload, http_timeout):
            calls.append(payload)
            return {'choices':[{'message':{'content':'{"type":"move","direction":"forward"}'}}]}
        with patch.object(chat, 'complete', fake), patch.object(sys, 'argv', ['chat', '--model','mock']), \
                patch.dict(os.environ, {'OPENAI_BASE_URL':'https://example.com/v1'}, clear=False), \
                patch.object(sys, 'stdin', io.StringIO('\n'.join(packets))), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(chat.main(), 0)
        return calls, packets

    def test_history_window_and_notes_permissions(self):
        full, packets = self._adapter_requests('full')
        window, _ = self._adapter_requests('window')
        notes, _ = self._adapter_requests('notes')
        first = json.loads(packets[0])['observation']['encounter']['request']
        first_person = json.loads(packets[0])['observation']['encounter']['person']
        self.assertIn(first, json.dumps(full[-1]['messages'], ensure_ascii=False))
        # Current and last previous turn only; older assistant actions do not
        # smuggle the initial observation into the last request.
        self.assertEqual(len(window[-1]['messages']), 4)
        self.assertNotIn(first_person, json.dumps(window[-1]['messages'], ensure_ascii=False))
        self.assertNotIn('tools', full[-1])
        self.assertNotIn('tools', window[-1])
        self.assertIn('tools', notes[-1])
        self.assertEqual(len(notes[-1]['messages']), 4)

    def test_note_mode_tool_call_is_visible_and_file_persists(self):
        packets = [json.dumps({'type':'observation', 'turn':0, 'observation':{'message':'MAIN'},
                               'instructions':'RULES', 'policy':'notes', 'window_turns':1,
                               'max_responses_per_turn':32}),
                   json.dumps({'type':'observation', 'turn':1, 'observation':{'message':'LATER'}}),
                   '{"type":"end"}']
        requests = []
        def fake(endpoint, key, payload, http_timeout):
            requests.append(payload)
            if len(requests) == 1:
                return {'choices':[{'message':{'content':None,'tool_calls':[{'id':'c1','type':'function',
                    'function':{'name':'write_file','arguments':json.dumps({'name':'memo.txt','text':'MAIN'})}}]}}]}
            return {'choices':[{'message':{'content':'{"type":"move","direction":"forward"}'}}]}
        with tempfile.TemporaryDirectory() as tmp:
            previous = Path.cwd(); os.chdir(tmp)
            try:
                with patch.object(chat, 'complete', fake), patch.object(sys, 'argv', ['chat','--model','mock']), \
                        patch.dict(os.environ, {'OPENAI_BASE_URL':'https://example.com/v1'}, clear=False), \
                        patch.object(sys, 'stdin', io.StringIO('\n'.join(packets))), \
                        contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(chat.main(), 0)
                self.assertEqual(Path('memo.txt').read_text(), 'MAIN')
                self.assertTrue(any(m.get('role') == 'tool' and m.get('content') == 'Сохранено.'
                                    for m in requests[1]['messages']))
                # Window of one does not retain the earlier dialogue: the only
                # persistent channel is the intentionally saved file.
                self.assertNotIn('MAIN', json.dumps(requests[2]['messages'], ensure_ascii=False))
            finally:
                os.chdir(previous)

    def test_note_file_tool_rejects_paths_and_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            previous = Path.cwd(); os.chdir(tmp)
            try:
                self.assertEqual(chat._file_tool({'name':'write_file','arguments':json.dumps({'name':'memo.txt','text':'hello'})}), 'Сохранено.')
                self.assertEqual(chat._file_tool({'name':'read_file','arguments':json.dumps({'name':'memo.txt'})}), 'hello')
                self.assertNotEqual(chat._file_tool({'name':'read_file','arguments':json.dumps({'name':'../memo.txt'})}), 'hello')
                Path('link').symlink_to('memo.txt')
                self.assertEqual(chat._file_tool({'name':'read_file','arguments':json.dumps({'name':'link'})}), 'Файл недоступен.')
            finally:
                os.chdir(previous)


if __name__ == '__main__':
    unittest.main()
