import json
import math
import os
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from xml.etree import ElementTree as ET

from spatial_harness_bench.engineering_geometry.__main__ import main, run
from spatial_harness_bench.engineering_geometry.model import Cable, Circle
from spatial_harness_bench.engineering_geometry.reference import construct
from spatial_harness_bench.engineering_geometry.sandbox import SandboxResult, execute
from spatial_harness_bench.engineering_geometry.svg import SVG_NS, parse_svg, render_svg
from spatial_harness_bench.engineering_geometry.task import (
    FAMILIES, PILOT_SEEDS, Addition, Instance, Radii, Spacing, Thickness, make_instance, prompt_for,
)
from spatial_harness_bench.engineering_geometry.verifier import verify


SOLVED = b'{"status":"solved"}'
INFEASIBLE = b'{"status":"infeasible"}'


def change_element(data, name, edit):
    root = ET.fromstring(data)
    element = next(node for node in root if node.get("id") == name)
    edit(root, element)
    return ET.tostring(root)


def move_core(cable, name="core-a", **coords):
    return replace(cable, cores=tuple(
        (key, replace(c, **coords) if key == name else c) for key, c in cable.cores
    ))


def alternate_ring(c):
    return (f"M {c.x - c.r:.9f},{c.y:.9f} "
            f"a {c.r:.9f} {c.r:.9f} 45 1 0 {c.r:.9f} {-c.r:.9f} "
            f"a {c.r:.9f} {c.r:.9f} -12 0 0 {-c.r:.9f} {c.r:.9f} z")


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.case = make_instance("core-radii", 1)
        self.answer = construct(self.case)
        self.svg = render_svg(self.answer)

    def rejects(self, code, svg, status=SOLVED, instance=None):
        verdict = verify(instance or self.case, svg, status)
        self.assertFalse(verdict.passed, verdict)
        self.assertEqual(verdict.code, code, verdict)

    def test_pilot_references_and_determinism(self):
        self.assertEqual(len(FAMILIES) * len(PILOT_SEEDS), 30)
        # Expected reasons follow the five public prompt templates, not conflict().
        proof_codes = ("core_overlap", "core_outside_insulation", "core_overlap",
                       "core_overlap", "core_outside_insulation")
        for family in FAMILIES:
            pair_set = set()
            for seed in PILOT_SEEDS:
                with self.subTest(family=family, seed=seed):
                    instance = make_instance(family, seed)
                    source = render_svg(instance.original)
                    prompt = prompt_for(instance)
                    twin = make_instance(family, seed)
                    self.assertEqual(instance, twin)
                    self.assertEqual(source, render_svg(twin.original))
                    self.assertEqual(prompt, prompt_for(twin))
                    input_cores = list(parse_svg(source).cores.values())
                    self.assertEqual(len(input_cores), 3)
                    side = input_cores[0].distance(input_cores[1])
                    self.assertAlmostEqual(input_cores[0].distance(input_cores[2]), side, places=7)
                    self.assertAlmostEqual(input_cores[1].distance(input_cores[2]), side, places=7)
                    pair_set.add((source, prompt))
                    if family == "incompatible":
                        verdict = verify(instance, None, INFEASIBLE)
                        self.assertEqual(verdict.code, proof_codes[seed])
                    else:
                        verdict = verify(instance, render_svg(construct(instance)), SOLVED)
                    self.assertTrue(verdict.passed, verdict)
            self.assertEqual(len(pair_set), len(PILOT_SEEDS))

    def test_infeasible_witnesses_and_nearby_feasible_requests(self):
        original = Cable(Circle(0, 0, 22), Circle(0, 0, 18), (
            ("core-a", Circle(-6, -2 * math.sqrt(3), 2)),
            ("core-b", Circle(6, -2 * math.sqrt(3), 2)),
            ("core-c", Circle(0, 4 * math.sqrt(3), 2)),
        ))
        cases = [
            # Centres 12 mm apart, so 7+6 overlaps and 5+6 clears.
            (Radii((7, 6, 2)), Radii((5, 6, 2)), "core_overlap"),
            # A core extends to 4√3+2 ≈ 8.93 mm from the origin.
            (Thickness(15, "outer"), Thickness(12, "outer"), "core_outside_insulation"),
            # Two radii of 2 mm touch at spacing 4 mm, but not at 5 mm.
            (Spacing(4), Spacing(5), "core_overlap"),
            # Core-d would be centered on core-a instead of the free centroid.
            (Addition(1, "core-a", 100), Addition(1, "core-a", 0), "core_overlap"),
            # At 300% core-d extends to 12√3+1 > 18 mm; 120% still overlaps core-c.
            # The centroid is free and strictly inside the same shell.
            (Addition(1, "core-c", 300), Addition(1, "core-c", 0), "core_outside_insulation"),
        ]
        for bad, good, code in cases:
            with self.subTest(edit=bad):
                impossible = Instance("manual", 0, original, bad)
                verdict = verify(impossible, None, INFEASIBLE)
                self.assertTrue(verdict.passed, verdict)
                self.assertEqual(verdict.code, code)
                feasible = Instance("manual", 0, original, good)
                self.rejects("false_refusal", None, INFEASIBLE, instance=feasible)

    def test_hand_authored_solution_without_reference_or_svg_serializer(self):
        original = Cable(Circle(0, 0, 15), Circle(0, 0, 12), (
            ("core-a", Circle(-5, 0, 1)), ("core-b", Circle(5, 0, 1)),
            ("core-c", Circle(0, 6, 1)),
        ))
        case = Instance("core-radii", 0, original, Radii((2, 1, 1)))
        svg = b"""<svg xmlns="http://www.w3.org/2000/svg" width="100mm" height="100mm" viewBox="-50 -50 100 100">
          <path id="shell" fill="#687582" fill-rule="evenodd" d="
            M 15 0 A 15 15 0 0 1 -15 0 A 15 15 0 0 1 15 0 Z
            M 12 0 A 12 12 0 0 1 -12 0 A 12 12 0 0 1 12 0 Z"/>
          <path id="insulation" fill="#f1ddb2" fill-rule="evenodd" d="
            M 12 0 A 12 12 0 0 1 -12 0 A 12 12 0 0 1 12 0 Z
            M -3 0 A 2 2 0 0 1 -7 0 A 2 2 0 0 1 -3 0 Z
            M 6 0 A 1 1 0 0 1 4 0 A 1 1 0 0 1 6 0 Z
            M 1 6 A 1 1 0 0 1 -1 6 A 1 1 0 0 1 1 6 Z"/>
          <circle id="core-a" fill="#b66038" cx="-5" cy="0" r="2"/>
          <circle id="core-b" fill="#b66038" cx="5" cy="0" r="1"/>
          <circle id="core-c" fill="#b66038" cx="0" cy="6" r="1"/>
        </svg>"""
        verdict = verify(case, svg, SOLVED)
        self.assertTrue(verdict.passed, verdict)

    def test_equivalent_xml_paths_and_circles(self):
        root = ET.fromstring(self.svg)
        root.attrib.update(width="1e2mm", height="100.0mm", viewBox="-5e1, -50, 100, 1e2")
        insulation = root.find(f'{SVG_NS}path[@id="insulation"]')
        insulation.set("d", " ".join(alternate_ring(c) for c in (
            *(c for _, c in reversed(self.answer.cores)), self.answer.inner
        )))
        shell = root.find(f'{SVG_NS}path[@id="shell"]')
        shell.set("d", alternate_ring(self.answer.inner) + " " + alternate_ring(self.answer.outer))
        node = root.find(f'{SVG_NS}circle[@id="core-a"]')
        node.tag = SVG_NS + "path"
        for key in ("cx", "cy", "r"):
            del node.attrib[key]
        node.set("d", alternate_ring(self.answer.core_map()["core-a"]))
        root[:] = reversed(root[:])
        verdict = verify(self.case, ET.tostring(root, xml_declaration=True, encoding="utf-8"), SOLVED)
        self.assertTrue(verdict.passed, verdict)

    def test_equivalent_exact_semicircles_at_decimal_coordinates(self):
        c = parse_svg(self.svg).cores["core-a"]
        d = (f"M {c.x + c.r:.9f} {c.y:.9f} "
             f"a {c.r} {c.r} 0 0 1 {-2 * c.r} 0 "
             f"a {c.r} {c.r} 0 0 1 {2 * c.r} 0 z")

        def as_path(_, node):
            node.tag = f"{SVG_NS}path"
            for key in ("cx", "cy", "r"):
                del node.attrib[key]
            node.set("d", d)

        verdict = verify(self.case, change_element(self.svg, "core-a", as_path), SOLVED)
        self.assertTrue(verdict.passed, verdict)

    def test_published_examples_match_the_versioned_generator(self):
        examples = (Path(__file__).resolve().parents[1]
                    / "spatial_harness_bench/engineering_geometry/examples")
        for family, seed in (("add-core", 2), ("remove-core", 0)):
            folder = examples / f"{family}-{seed}"
            case = make_instance(family, seed)
            self.assertEqual((folder / "input.svg").read_bytes(), render_svg(case.original))
            verdict = verify(case, (folder / "output.svg").read_bytes(), (folder / "status.json").read_bytes())
            self.assertTrue(verdict.passed, verdict)

    def test_rotated_triangle_of_right_size_is_not_equivalent(self):
        case = make_instance("core-spacing", 3)
        solution = construct(case)
        x, y = solution.centroid()
        rotated = tuple(
            (name, Circle(x - (c.y - y), y + (c.x - x), c.r)) for name, c in solution.cores
        )
        self.rejects("changed_ray", render_svg(replace(solution, cores=rotated)), instance=case)

    def test_wrong_position_with_requested_radius_and_matched_hole(self):
        c = self.answer.core_map()["core-a"]
        self.rejects("changed_center", render_svg(move_core(self.answer, x=c.x + 0.05)))

    def test_added_core_at_wrong_geometric_position(self):
        case = make_instance("add-core", 0)
        solution = construct(case)
        c = solution.core_map()["core-d"]
        self.rejects("wrong_added_center", render_svg(move_core(solution, "core-d", y=c.y + 0.02)),
                     instance=case)

    def test_changed_unrelated_objects(self):
        self.rejects("changed_shell", render_svg(replace(
            self.answer, outer=replace(self.answer.outer, r=self.answer.outer.r + 0.1)
        )))
        case = make_instance("shell-thickness", 0)
        solution = construct(case)
        c = solution.core_map()["core-b"]
        self.rejects("changed_radius", render_svg(move_core(solution, "core-b", r=c.r + 0.1)),
                     instance=case)

    def test_core_overlap_and_cavity_escape(self):
        b = self.answer.core_map()["core-b"]
        self.rejects("core_overlap", render_svg(move_core(self.answer, x=b.x + 0.1, y=b.y)))
        self.rejects("core_outside_insulation", render_svg(move_core(self.answer, x=49.9, y=0)))

    def test_shell_exceeds_the_viewport(self):
        invalid = replace(self.answer, outer=replace(self.answer.outer, r=51))
        self.rejects("outside_view", render_svg(invalid))

    def test_microscale_gap_or_overlap_at_a_shared_boundary(self):
        c = self.answer.core_map()["core-a"]
        for delta in (0.000005, -0.000005):
            invalid = change_element(self.svg, "core-a", lambda _, el: el.set("r", f"{c.r + delta:.9f}"))
            self.rejects("shared_boundary", invalid)

    def test_insulation_does_not_cover_the_shell_seam(self):
        self.rejects("shared_boundary", change_element(
            self.svg, "insulation", lambda _, el: el.set("d", " ".join(map(alternate_ring, (
                replace(self.answer.inner, r=self.answer.inner.r - 0.000005),
                *(c for _, c in self.answer.cores),
            ))))
        ))

    def test_leftover_or_missing_hole_and_empty_id(self):
        case = make_instance("remove-core", 1)
        removed = change_element(render_svg(case.original), "core-b", lambda root, el: root.remove(el))
        self.rejects("shared_boundary", removed, instance=case)
        without_hole = change_element(self.svg, "insulation", lambda _, el: el.set("d", el.get("d").rsplit(" M ", 1)[0]))
        self.rejects("shared_boundary", without_hole)
        self.rejects("invalid_svg", change_element(render_svg(construct(case)), "insulation",
                                                   lambda root, _: ET.SubElement(root, f"{SVG_NS}g", id="core-b")),
                     instance=case)

    def test_wrong_core_identity_even_with_valid_geometry(self):
        swapped = change_element(self.svg, "core-a", lambda _, el: el.set("id", "core-d"))
        self.rejects("wrong_core_ids", swapped)

    def test_unchanged_input_is_not_a_solution(self):
        self.rejects("wrong_radius", render_svg(self.case.original))

    def test_syntax_and_active_content(self):
        malicious_or_unsupported = (
            self.svg.replace(b'fill-rule="evenodd"', b'fill-rule="nonzero"', 1),
            self.svg.replace(b'width="100mm"', b'width="100cm"'),
            self.svg.replace(b"<circle ", b'<circle transform="translate(1)" ', 1),
            self.svg.replace(b"<circle ", b'<circle style="display:none" ', 1),
            self.svg.replace(b" A ", b" Q ", 1),
            self.svg.replace(b" A ", b" A ,", 1),
            b'<!DOCTYPE svg [<!ENTITY k "x">]>\n' + self.svg,
            b'<?xml-stylesheet href="network.css"?>\n' + self.svg,
        )
        for svg in malicious_or_unsupported:
            self.rejects("invalid_svg", svg)

    def test_partial_elliptical_and_twice_wound_paths_rejected(self):
        c = self.answer.core_map()["core-a"]
        ring = alternate_ring(c)
        paths = [
            ring[:ring.rfind(" a ")] + " z",  # implicit chord, not a full circle
            ring.replace(f"{c.r:.9f} {c.r:.9f}", f"{c.r:.9f} {c.r * 1.01:.9f}", 1),
            ring[:-2] + ring[ring.index(" a "):],
            ring.replace(" 1 0 ", " 1 1 "),  # inconsistent direction
        ]
        for path in paths:
            def as_path(_, node):
                node.tag = f"{SVG_NS}path"
                for key in ("cx", "cy", "r"):
                    del node.attrib[key]
                node.set("d", path)
            self.rejects("invalid_svg", change_element(self.svg, "core-a", as_path))

    def test_status_protocol_and_false_claims(self):
        self.rejects("false_refusal", None, INFEASIBLE)
        self.rejects("unexpected_output", self.svg, INFEASIBLE)
        self.rejects("missing_svg", None)
        for data in (None, b"", b"[]", b'{"status":"solved","status":"solved"}',
                     b'{"status":27}', b'{"status":"solved","thickness":8}', b"\xff", b" " * 4100):
            self.rejects("invalid_status", self.svg, data)
        for seed in PILOT_SEEDS:
            case = make_instance("incompatible", seed)
            self.rejects("false_success", render_svg(case.original), instance=case)


class RunnerTests(unittest.TestCase):
    def test_series_writes_artifacts_and_separates_outcomes(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "results"
            outcomes = (SandboxResult(status_json=INFEASIBLE), SandboxResult(status_json=SOLVED),
                        SandboxResult(technical_error="provider unavailable"))
            with patch("spatial_harness_bench.engineering_geometry.__main__.execute",
                       side_effect=outcomes) as launch:
                report = run(["agent"], output, image="test", family="incompatible", seed=120, count=3)
            self.assertEqual((report["passed"], report["agent_errors"], report["technical_errors"]), (1, 1, 1))
            self.assertEqual((report["score"], report["max_score"]), (1, 2))
            self.assertIsNone(report["items"][2]["score"])
            self.assertEqual(json.loads((output / "result.json").read_bytes()), report)
            for i, call in enumerate(launch.call_args_list):
                folder = output / f"{i + 1:04d}"
                image, command, svg, prompt = call.args
                self.assertEqual((image, command), ("test", ["agent"]))
                self.assertNotIn("incompatible", prompt)
                self.assertNotIn("seed", prompt)
                self.assertEqual(call.kwargs["network"], "none")
                self.assertEqual((folder / "input.svg").read_bytes(), svg)
                self.assertEqual((folder / "prompt.txt").read_text(), prompt)
                self.assertTrue((folder / "stdout.log").is_file())
                self.assertTrue((folder / "result.json").is_file())
            self.assertEqual((output / "0001/status.json").read_bytes(), INFEASIBLE)
            self.assertFalse((output / "0003/status.json").exists())
            with self.assertRaises(FileExistsError):
                run(["agent"], output, image="test", family="all", seed=3)

    def test_all_families_and_cli_return_codes(self):
        with TemporaryDirectory() as tmp, patch(
            "spatial_harness_bench.engineering_geometry.__main__.execute", return_value=SandboxResult()
        ):
            output = Path(tmp) / "all"
            report = run(["agent"], output, image="test", family="all", seed=0, count=5)
            self.assertEqual(report["total"], 30)
            self.assertEqual({row["family"] for row in report["items"]}, set(FAMILIES))
            self.assertEqual((report["agent_errors"], report["technical_errors"]), (30, 0))
            with patch("sys.argv", ["program", "--image", "test", "--family", "remove-core",
                                    "--seed", "0", "--output", str(Path(tmp) / "cli"), "--", "agent"]):
                self.assertEqual(main(), 1)

    def test_invalid_seed_count_and_timeout_rejected_before_writes(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "invalid"
            for params in ({"seed": -1}, {"count": 0}, {"seed": 2**63 - 1, "count": 2}, {"timeout": math.nan}):
                with self.assertRaises(ValueError):
                    run(["agent"], output, image="test", family="all", **({"seed": 0} | params))
                self.assertFalse(output.exists())


@unittest.skipUnless(os.getenv("ENGINEERING_GEOMETRY_TEST_IMAGE"), "set ENGINEERING_GEOMETRY_TEST_IMAGE for real Docker smoke tests")
class DockerTests(unittest.TestCase):
    def launch(self, code, timeout=10):
        return execute(os.environ["ENGINEERING_GEOMETRY_TEST_IMAGE"], ["python3", "-c", code],
                       b"<svg/>", "test prompt", timeout=timeout)

    def test_docker_receive_only_inputs_and_collect_outputs(self):
        with patch.dict(os.environ, {"ENGINEERING_GEOMETRY_EVALUATOR_ONLY": "canary"}):
            result = self.launch(
                'import os,pathlib,sys; p=pathlib.Path; '
                'assert set(os.listdir(".")) == {"input.svg", "prompt.txt"}; '
                'assert os.getcwd() == "/task" and os.getuid() != 0; '
                'assert not p("/var/run/docker.sock").exists(); '
                'assert not p("/task/.git").exists(); '
                'assert "ENGINEERING_GEOMETRY_EVALUATOR_ONLY" not in os.environ; '
                'assert sys.stdin.read() == p("prompt.txt").read_text() == "test prompt"; '
                'p("output.svg").write_bytes(p("input.svg").read_bytes()); p("status.json").write_bytes(b"{}"); '
                'print("x"*100000)'
            )
        self.assertIsNone(result.technical_error, result.stderr)
        self.assertIsNone(result.agent_error)
        self.assertEqual(result.output_svg, b"<svg/>")
        self.assertEqual(result.status_json, b"{}")
        self.assertTrue(result.logs_truncated)

    def test_docker_no_follow_no_oversize_and_teardown(self):
        symlink = self.launch('import os; os.symlink("/etc/passwd", "output.svg")')
        self.assertIn("regular file", symlink.agent_error)
        self.assertIsNone(symlink.output_svg)
        oversize = self.launch('from pathlib import Path; Path("status.json").write_bytes(b"x"*100000)')
        self.assertIn("artifact limit", oversize.agent_error)
        timed_out = self.launch("import time; time.sleep(30)", timeout=0.5)
        self.assertIn("deadline", timed_out.technical_error)
        self.assertNotIn("remove container", timed_out.technical_error)
        nonzero = self.launch("raise SystemExit(75)")
        self.assertIn("exit 75", nonzero.technical_error)
        self.assertEqual(nonzero.returncode, 75)
