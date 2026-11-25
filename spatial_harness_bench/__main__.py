"""Command-line entry point for running a benchmark suite."""

import argparse
import sys

from . import suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run",))
    argv = sys.argv[1:]
    if argv[:1] == ["run"]:
        return suite.main(argv[1:])
    parser.parse_args(argv)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
