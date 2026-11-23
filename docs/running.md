# Installing and running

Code layout, installation, and how to run each stage. Back to the [README](../README.md).

## Layout

Relevant code:

```
bin/                  Shell entry points for each pipeline stage
  hf-download.sh        1. Download leaderboard results from Hugging Face
  stan-prep.sh          2. Build Stan inputs (data, variables, chain inits) per subject
  stan-sample.sh        3. Run Stan sampling
  model-fit.sh          4. Model-fit diagnostics (R-hat/ESS, chain sync, floor/ceiling)
src/
  data/                 Collection: list, gather, reduce, download (rate-limit and gating aware)
  experiments/          Defines what gets fit (one experiment per MMLU subject)
  model/                aggregate-data, build-ids, to-stan / to-variables / to-init, model.stan, irt-model.py (PyMC)
  analysis/             Post-fit work: item response curves, Hub upload prep
    model-fit/          ArviZ-based diagnostics and cross-subject reports
mylib/                Shared library (types, logging, question database)
etc/db.sql            Schema for the question bank / model metadata database
tests/                pytest suite
```

## Requirements

- Python >= 3.13, managed with [uv](https://docs.astral.sh/uv/)
- [CmdStan](https://mc-stan.org/users/interfaces/cmdstan) 2.40.0, unpacked at
  the repo root as `cmdstan-2.40.0/` and built (`make build`)
- GNU userland: the scripts use `nproc`, `mktemp --tmpdir`, and long-form
  `rm` flags (on macOS: `brew install coreutils` and put the GNU versions
  first on `PATH`)
- [GNU parallel](https://www.gnu.org/software/parallel/) and `pigz`
- A Hugging Face token. The download and upload scripts read
  `HF_BEARER_TOKEN` from `~/.keys/hf`, and some leaderboard detail repos are
  gated, so access requests must be approved

## Installation

```bash
git clone https://github.com/jerome-white/leaderboard-item-response
cd leaderboard-item-response
uv sync                      # creates .venv; the bin/ scripts activate it
# build CmdStan in ./cmdstan-2.40.0, then compile the model:
(cd cmdstan-2.40.0 && make build)
(cd cmdstan-2.40.0 && make $PWD/../src/model/model)   # stan-sample.sh also does this
```

Run the tests with `uv run pytest tests/`.

## Running

This code has only ever been run on one machine, a 2024 Mac mini (M4 Pro, 64 GB).
Other machines and operating systems are untested.

The scripts are meant to be run in order. Each stage writes under `data/`, and
later stages find their inputs there. Pass `-h` to any script for its options.

```bash
# 1. Collect: list leaderboard result files, download per-question responses,
#    and record model/question metadata in a SQLite database. Safe to re-run:
#    already-downloaded files are skipped, and flagged models are excluded.
bin/hf-download.sh -o data/hugging-face

# 2. Prepare Stan inputs per subject (stan.json, variables.json, init.json).
bin/stan-prep.sh -d data/hugging-face

# 3. Sample. Slow: see the cost note below. Skips subjects that already have a summary.csv.
bin/stan-sample.sh -r data/opt -s 1000 -p 1000 -j 1

# 4. Diagnose the fits. Writes per-subject diagnostics and cross-subject reports.
bin/model-fit.sh -b mmlu -o data/fit
```

**Cost note.** At `num_warmup=1000`, one run of the full 13-subject MMLU suite
took 61.1 hours on a Mac mini (2024, M4 Pro, 64 GB) (the largest subject, math, took about 13.8 hours
alone). Runtime grows faster than linearly in the number of items (roughly
`N^1.56`). Run these in `tmux`/`nohup` with the machine kept awake. A
machine that sleeps stalls the job. Single-subject runs on 8 cores overnight
have been workable so far. Running on AWS has been planned (issues [#63](https://github.com/jerome-white/leaderboard-item-response/issues/63), [#67](https://github.com/jerome-white/leaderboard-item-response/issues/67))
and sketched in a draft PR ([#65](https://github.com/jerome-white/leaderboard-item-response/issues/65)), but it is unfinished and deprioritized;
don't treat it as a supported path.

**Chain initialization is not optional.** `stan-sample.sh` passes
`init=<subject>/init.json`. Without it, CmdStan's default random-per-chain
initialization fails to converge on this model (see the [journal](journal.md)).

**Do not write into `data/opt/` or `data/hugging-face/`.** They gate very
long-running processes (the Hugging Face download and full Stan sampling), so
treat them as read-only when experimenting. Write experiment output
elsewhere, for example `data/experiments/`.

**Refitting with items removed (experimental).** `bin/exclude-items.sh` was
built for issue [#71](https://github.com/jerome-white/leaderboard-item-response/issues/71). It works on one subject at a time, is not yet a sweep
over subjects, and isn't part of the main pipeline. It rebuilds
`stan.json`/`variables.json`/`init.json` with the given item ids dropped:

```bash
bin/exclude-items.sh -i data/opt/mmlu/history -o data/experiments/history-excluded -e <item id> [-e <item id> ...]
```

The list of item ids currently comes from a one-off filter on
`diagnostics.csv` (beta R-hat > 1.1), not from a reusable script.
