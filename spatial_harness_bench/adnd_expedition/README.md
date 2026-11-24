# Short AD&D-inspired expedition

This is an exploratory, short version of the `expedition-0.1.0` game, exposed through a JSON-lines harness adapter. It vendors `expedition.py` and its independent difficulty modules from `nyddle/adndbench` commit `6012e8db0a6e41d89afb2d4f8fd0958bb0893b0a`; the engine's only change is its package import. It does not access a local `adndbench` checkout. The game and its rules are inspired by tabletop adventures and are not an official AD&D product.

The default is a **single 80-turn episode** in the default expedition profile with two additional encounters. The short limit is an operational budget, **not a calibrated difficulty**; time-limit endings may be common. `score` is 1 only for completion of the required quest, 0 for death, turn limit, or a harness that consumes its 32-response decision limit, and `null` for adapter, timeout or gateway errors. Report `completed`, `death`, `turn_limit`, `response_limit`, and `technical_error` separately. The `task` ID contains the protocol version and turn limit so `stats` cannot pool budgets accidentally. Users must still label different adapters and their model, generation, context and notes settings separately. This optional task does not alter the existing leaderboard or imply comparability with its scores.

## Run

Install this repository's package and its prerequisites, and use a new output directory:

```sh
python -m spatial_harness_bench.adnd_expedition --output ./private-expedition -- \
  python -m spatial_harness_bench.adnd_expedition.pi -- pi
```

`pi` is an example executable name; configure its provider and credentials independently. The bridge uses Pi RPC with `read,write` tools and disables sessions, extensions and skills as in the maze adapter. For a different harness, supply its JSON-lines adapter directly after `--`. Adjust `--max-turns` and per-action `--timeout` explicitly if required. Omit `--seed` to generate a new random seed; `--seed PUBLIC_DEVELOPMENT_SEED` allows a reproducible development run. Avoid putting private seeds into shell command lines or public logs.

An OpenAI-compatible chat adapter is also available. Set `OPENAI_BASE_URL` to the provider's `/v1` base and `OPENAI_API_KEY` through your existing secure environment; never put a credential into an output directory. For example, once the package is installed:

```sh
python -m spatial_harness_bench.adnd_expedition --output ./private-expedition -- \
  python -m spatial_harness_bench.adnd_expedition.chat --model YOUR_MODEL_ID --context-turns 20
```

By default it keeps the system instruction and the last 20 observed turns, without file tools. Pass `--notes` to allow the model to read, write and append its own files in the temporary working directory; this is a separate harness condition, not an automatic map derived by the engine. Provider tool support is required. `--temperature`, `--max-tokens` and `--effort` are optional provider settings; fix them consistently for comparisons. This adapter reuses the memory task's connection and retry handling. A provider response with `finish_reason=length` is reported as an unscored technical failure `reason=output_limit`, even if a non-empty or valid-looking prefix was returned. A missing text response is also an unscored inference failure.

## Protocol

The adapter receives one JSON line per turn:

```json
{"type":"observation","turn":1,"instructions":"...","observation":{"room":"...","zone":"...","exits":["north"],"turn":0}}
```

The `instructions` field appears only on the first turn. `observation` is the game engine's current public observation, not the world map or seed. The adapter returns exactly one line, for example:

```json
{"action":{"type":"move","direction":"north"}}
```

Other actions and allowed fields are specified in the first-turn instructions. Extra fields, duplicate JSON keys and malformed answers become invalid game actions and consume a turn. A transport failure may be indicated with `{"type":"gateway_error"}` and is unscored; `{"type":"output_limit"}` separately signals a truncated inference response, also unscored. Exhausting 32 harness responses is signaled by `{"type":"response_limit","responses":32}` and scored as zero with reason `response_limit`, separately from deaths and ordinary turn-limit outcomes. An invalid response-limit count is an unscored adapter error. After a terminal game the harness receives `{"type":"end","status":"..."}`; timeout, missing response and protocol-level failures are separately reported.

The supplied command executes in a fresh temporary working directory, which can be used for a map and notes during the episode and is removed at exit. **A temporary working directory is not a security sandbox.** A harness with arbitrary filesystem access may be able to read other local files; a protected evaluation must arrange actual isolation outside this runner.

## Private local evidence

The new output directory has mode 0700 and its files mode 0600. `replay.json` contains the private seed, rules and settings needed to reproduce the world; `trace.jsonl` contains public observations, canonical actions and resulting observations; `result.json` contains aggregate outcomes. Seed and hidden world state are never included in the JSON-lines messages or printed in the aggregate result. Treat all three files as private evaluation material; do not publish them or commit held-out seeds. Replay by generating the world from `replay.json`, checking each saved observation, and applying each saved action in order.
