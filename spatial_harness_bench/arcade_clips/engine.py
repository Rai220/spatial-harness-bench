"""Deterministic two-player ring arena. Run python engine.py for a replay summary."""
import hashlib
import json
from pathlib import Path

PLAYERS = ("p1", "p2")
MOVE = {"left": -1, "idle": 0, "right": 1}


def frames(path=Path("replay.jsonl")):
    """Yield (byte offset, raw line, decoded frame), validating the input stream."""
    with path.open("rb") as stream:
        offset = 0
        for tick, raw in enumerate(stream):
            row = json.loads(raw)
            if set(row) != {"tick", *PLAYERS} or type(row["tick"]) is not int or row["tick"] != tick:
                raise ValueError(f"bad frame at byte {offset}")
            for player in PLAYERS:
                pad = row[player]
                if (set(pad) != {"direction", "jump", "fire"} or pad["direction"] not in MOVE
                        or any(type(pad[key]) is not bool for key in ("jump", "fire"))):
                    raise ValueError(f"bad {player} input at tick {tick}")
            yield offset, raw, row
            offset += len(raw)


def initial():
    # Each rider stores lane, battery charge and points.
    return [[0, 12, 0], [8, 12, 0]]


def advance(state, row):
    """Move on a ring, recharge by coasting, and fire at moving targets."""
    tick = row["tick"]
    for rider, player in zip(state, PLAYERS, strict=True):
        pad = row[player]
        rider[0] = (rider[0] + MOVE[pad["direction"]] * (1 + pad["jump"])) % 17
        if pad["fire"] and rider[1] >= 3:
            rider[1] -= 3
            distance = (rider[0] - tick // 7) % 17
            rider[2] += 4 if distance == 0 else int(distance in (1, 16))
        elif not pad["fire"]:
            rider[1] = min(12, rider[1] + 1)


def signature(rows):
    """Independent checkpoint of the canonical input frames in one block."""
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, separators=(",", ":"), sort_keys=True).encode() + b"\n")
    return digest.hexdigest()


if __name__ == "__main__":
    state = initial()
    ticks = 0
    for _, _, frame in frames():
        advance(state, frame)
        ticks += 1
    print(json.dumps({"ticks": ticks, "state": state}))
