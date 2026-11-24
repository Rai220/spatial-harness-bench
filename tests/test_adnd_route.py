"""Route task invariants and isolated runner tests; no model calls."""
from collections import Counter, deque
import hashlib
import heapq
import json
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from spatial_harness_bench.adnd_route.__main__ import run
from spatial_harness_bench.adnd_route.task import check_response, generate_routes, public_probe
from spatial_harness_bench.gateway import GatewayError
from spatial_harness_bench.stats import rows as stats_rows


def oracle(public):
    graph = {room['room']: room['exits'] for room in public['context']['observed_map']}
    start, goal = re.findall(r'«([^»]+)»', public['question'])
    dist, ways, routes = {start: 0}, {start: 1}, {start: []}
    todo = [(0, start)]
    while todo:
        cost, node = heapq.heappop(todo)
        if cost != dist[node]:
            continue
        for direction, edge in graph[node].items():
            target, new = edge['room'], cost + edge['cost']
            if new < dist.get(target, float('inf')):
                dist[target], ways[target] = new, ways[node]
                routes[target] = routes[node] + [direction]
                heapq.heappush(todo, (new, target))
            elif new == dist.get(target):
                ways[target] += ways[node]
    hops, firsts = {start: 0}, {start: set()}
    todo = deque([start])
    while todo:
        node = todo.popleft()
        for direction, edge in graph[node].items():
            target = edge['room']
            first = {direction} if node == start else firsts[node]
            if target not in hops:
                hops[target], firsts[target] = hops[node] + 1, set(first)
                todo.append(target)
            elif hops[target] == hops[node] + 1:
                firsts[target].update(first)
    return graph, ways[goal], routes[goal], firsts[goal]


class AdndRouteTests(unittest.TestCase):
    def test_original_protocol_compatibility(self):
        routes = generate_routes('dev-route-compatibility-1')
        digest = hashlib.sha256(json.dumps(routes, ensure_ascii=False, sort_keys=True,
                                          separators=(',', ':')).encode()).hexdigest()
        self.assertEqual(digest, 'cda28136a9d0f3b41721c7c22e720cf34edc93e18060810cb0d1b5ea0bdf49e2')
        self.assertEqual(routes, generate_routes('dev-route-compatibility-1'))

    def test_routes_are_unique_and_require_weights(self):
        for seed in range(10):
            probes = generate_routes(f'dev-{seed}')
            self.assertEqual(Counter(p['answer'] for p in probes), Counter(north=1, south=1, east=1, west=1))
            ranks = []
            for probe in probes:
                public = public_probe(probe)
                self.assertNotIn('answer', public)
                graph, ways, route, unweighted_firsts = oracle(public)
                self.assertIn(len(graph), (25, 36))
                self.assertGreater(sum(len(e) for e in graph.values())//2, len(graph)-1)
                self.assertEqual(ways, 1)
                self.assertTrue(5 <= len(route) <= 8)
                self.assertEqual(route[0], probe['answer'])
                self.assertNotIn(route[0], unweighted_firsts)
                start = re.findall(r'«([^»]+)»', public['question'])[0]
                ranked = sorted(graph[start], key=lambda d: (graph[start][d]['cost'], d))
                self.assertEqual(len(ranked), 4)
                ranks.append(ranked.index(route[0]))
            self.assertEqual(sorted(ranks), [0, 1, 2, 3])

    def test_run_uses_observations_only_and_grades_fresh_sessions(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / 'out'
            private_seed = 'private-seed-9d453a'
            workspaces = []
            def solve(command, work, question, env):
                public = json.loads(question.split('\n', 1)[1])
                self.assertNotIn('answer', public)
                self.assertNotIn(private_seed, question)
                self.assertFalse(list(work.iterdir()))
                workspaces.append(str(work))
                _, _, route, _ = oracle(public)
                (work / 'answer.json').write_text(json.dumps({'id': public['id'], 'answer': route[0]}))
            with patch('spatial_harness_bench.adnd_route.__main__.run_stage', side_effect=solve):
                report = run(['fake-harness'], out, private_seed)
            self.assertEqual((report['status'], report['score'], report['max_score']), ('complete', 4, 4))
            self.assertEqual(len(set(workspaces)), 4)
            self.assertEqual((out / 'seed.txt').stat().st_mode & 0o777, 0o600)
            self.assertEqual((out / 'result.json').stat().st_mode & 0o777, 0o600)
            self.assertNotIn(private_seed, json.dumps(report))
            self.assertTrue(all(item['format_valid'] and item['points'] == 1 for item in report['items']))
            row = list(stats_rows(report))[0]
            self.assertEqual((row[0], row[2], row[3], row[4], row[5]), ('adnd-route-v1', 4, 4, 4, 4))

    def test_gateway_error_is_not_scored_as_failure(self):
        with TemporaryDirectory() as tmp:
            def stage(command, work, question, env):
                if 'route_1' in question:
                    raise GatewayError('failure')
                (work / 'answer.json').write_text('{"id":"wrong","answer":"north"}')
            with patch('spatial_harness_bench.adnd_route.__main__.run_stage', side_effect=stage):
                report = run(['fake'], Path(tmp) / 'out', 'dev-seed')
            self.assertEqual(report['status'], 'error')
            self.assertIsNone(report['score'])
            self.assertEqual(report['items'][0]['reason'], 'gateway')
            self.assertIsNone(report['items'][0]['points'])
            self.assertEqual([x['points'] for x in report['items'][1:]], [0, 0, 0])
            self.assertIsNone(list(stats_rows(report))[0][2])

    def test_duplicates_and_extra_json_fields_are_incorrect(self):
        probe = generate_routes('dev-seed')[0]
        answer = json.dumps({'id': probe['id'], 'answer': probe['answer'], 'extra': 1})
        self.assertFalse(check_response(probe, answer))
        duplicate = '{"id":' + json.dumps(probe['id']) + ',"answer":"north","answer":"south"}'
        self.assertFalse(check_response(probe, duplicate))

    def test_cli_runs_real_fresh_processes_and_reports_failures(self):
        with TemporaryDirectory() as tmp:
            code = 'from pathlib import Path; Path("answer.json").write_text("{}")'
            output = Path(tmp) / 'cli-output'
            proc = subprocess.run([sys.executable, '-m', 'spatial_harness_bench.adnd_route',
                                   '--seed', 'public-debug', '--output', str(output), '--',
                                   sys.executable, '-c', code], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0)
            result = json.loads((output / 'result.json').read_text())
            self.assertEqual((result['status'], result['score']), ('complete', 0))
            self.assertNotIn('public-debug', proc.stdout)


if __name__ == '__main__':
    unittest.main()
