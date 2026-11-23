# leaderboard-item-response

Bayesian item response theory (IRT) applied to per-question results from the
Hugging Face [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard).

> **Status:** an exploration, not a finished method. Results below are
> working conclusions from a running investigation, tracked in the
> [issues](https://github.com/jerome-white/leaderboard-item-response/issues).

## 1. Overview

I started this repository to practice Bayesian data analysis on a domain I
care about: evaluating large language models. The Open LLM Leaderboard
published per-question results for thousands of models, which is exactly the
shape of data IRT was built for: many "test takers" (models), many items
(benchmark questions), and a binary correct/incorrect response for each pair.

An IRT model gives each model an *ability* (`theta`) and each question a
*difficulty* (`beta`) and a *discrimination* (`alpha`), rather than collapsing
everything into a single accuracy number. Here that is a two-parameter
logistic (2PL) model, fit with full Bayesian inference (Stan / MCMC):

```
P(model j answers question i correctly) = logistic( alpha_i * (theta_j - beta_i) )
```

**What this project is, and isn't.** There is no fixed target. It started as a
vehicle for learning, and the question being asked has been getting defined
along the way: first "can I get this data and fit this model at all?", then
"does the fit actually converge?", then "does the model fit the data?", and now
"do the problems I found change which models are ranked above which?" Each
answer has opened the next question, and the [journal](#5-journal) records
that path. Expect the scope to keep moving.

Two goals coexist:

1. A working, end-to-end pipeline: collect leaderboard data, fit IRT models,
   evaluate the results.
2. A personal learning project in Bayesian workflow (priors, identifiability,
   sampler diagnostics, posterior checks). Some choices in the code are
   deliberately exploratory rather than settled best practice. A PyMC
   implementation of the model (on the `pymc` branch) sits alongside the
   Stan one for that reason.

The current scope is the 13 MMLU-Pro subject groups from the leaderboard's v2
era (June 2024 to March 2025), each fit as its own model.

## 2. Code layout and installation

### Layout

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

### Requirements

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

### Installation

```bash
git clone https://github.com/jerome-white/leaderboard-item-response
cd leaderboard-item-response
uv sync                      # creates .venv; the bin/ scripts activate it
# build CmdStan in ./cmdstan-2.40.0, then compile the model:
(cd cmdstan-2.40.0 && make build)
(cd cmdstan-2.40.0 && make $PWD/../src/model/model)   # stan-sample.sh also does this
```

Run the tests with `uv run pytest tests/`.

## 3. How to run things

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
initialization fails to converge on this model (see the journal).

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

## 4. Related work

Applying IRT to LLM benchmark results is a small but real subfield
("NLP psychometrics"). The closest work I know of:

- **metabench** (Kipnis et al., ICLR 2025). IRT over thousands of models on the
  Open LLM Leaderboard v1 suite, used to find a small, maximally informative
  item subset. The closest sibling to this project.
- **tinyBenchmarks** (Maia Polo et al., ICML 2024,
  [arXiv:2402.14992](https://arxiv.org/abs/2402.14992)). IRT to compress MMLU,
  HELM, AlpacaEval and Open LLM Leaderboard data to about 100 items per
  benchmark while preserving rankings.
- **Rodriguez et al., "Evaluation Examples Are Not Equally Informative"** (ACL
  2021, [link](https://aclanthology.org/2021.acl-long.346/)). Bayesian IRT on
  leaderboard data to flag annotation errors and rank items by
  informativeness. Close to the question this repo keeps circling back to.
- **Vania et al., "Comparing Test Sets with Item Response Theory"** (ACL 2021,
  [link](https://aclanthology.org/2021.acl-long.92/)). IRT across 29 datasets
  to measure benchmark saturation.
- **John Lalor's work** on IRT for NLP, from
  [2016](https://arxiv.org/abs/1605.08889) through a 2024 EACL tutorial. He
  also maintains `py-irt` (variational inference).
- **"Lost in Benchmarks? Rethinking LLM Benchmarking with Item Response
  Theory"** ([arXiv:2505.15055](https://arxiv.org/abs/2505.15055)) and
  **Land, "Auditing LLM Benchmarks with IRT"** (arXiv:2605.30504). Both use
  IRT for benchmark quality forensics.
- **MMLU-Redux** (Gema et al., NAACL 2025,
  [link](https://aclanthology.org/2025.naacl-long.262/)). Manual
  re-annotation of MMLU, finding substantial label errors. Not IRT, but a
  human-labelled counterpart to what the item parameters here are meant to
  surface (planned comparison: issue [#69](https://github.com/jerome-white/leaderboard-item-response/issues/69)).

Where this project sits relative to those: most of that work *uses* IRT as a
tool (to shrink benchmarks, or to audit them). This repo spends its effort on
the fitting itself, asking whether full-Bayes 2PL on LLM data converges, why
not, and whether the pathologies matter downstream. That is a narrower and
less-explored angle.

Not yet checked: no one has been found doing the same multi-chain diagnostic
workflow on LLM-evaluation IRT, but the literature check was shallow. See
[`FUTURE.md`](FUTURE.md) for the longer notes (Elo vs. 2PL, MIRT and
embedding-based IRT, and a possible "2PL for pairwise comparisons").

## 5. Journal

Big-ticket conclusions, roughly in the order they were reached. Issue numbers
point to the evidence.

**Data collection is most of the work.** The leaderboard's per-question files
have gated repos, rate limits, canonical (no-author) model names, and
multiple result files per model. A large share of the early history is
making collection resilient ([#3](https://github.com/jerome-white/leaderboard-item-response/issues/3) to [#34](https://github.com/jerome-white/leaderboard-item-response/issues/34)). One finding that shaped the model
inputs: `doc_hash` is a hash of the raw document dict and drifts across
harness and dataset revisions, so question identity is keyed on `doc_id`
instead ([#21](https://github.com/jerome-white/leaderboard-item-response/issues/21), [#26](https://github.com/jerome-white/leaderboard-item-response/issues/26)). The leaderboard has two incompatible eras (v1, v2); this
repo uses v2 only.

**Per-subject models, not grouped MMLU.** Fitting MMLU super-categories
together was retired in favor of one model per subject ([#37](https://github.com/jerome-white/leaderboard-item-response/issues/37)).

**The 2PL model is not identified as written, and anchoring fixed that
([#45](https://github.com/jerome-white/leaderboard-item-response/issues/45), [#47](https://github.com/jerome-white/leaderboard-item-response/issues/47)).** `alpha`, `beta` and `theta` can trade off exactly: scaling
`alpha` by `c` and `theta`, `beta` by `1/c` leaves the likelihood unchanged,
and shifting `theta` and `beta` together does too. Pinning two abilities
(`theta[1]`, `theta[2]`) removes it. A first attempt, hard-standardizing
`theta` inside the model, was mathematically right but gave the sampler such
bad geometry that it effectively stalled.

**Allowing negative discrimination surfaces bad items ([#22](https://github.com/jerome-white/leaderboard-item-response/issues/22), [#35](https://github.com/jerome-white/leaderboard-item-response/issues/35)).** Dropping
the `alpha > 0` constraint lets mis-keyed or ambiguous questions show up as
negative discrimination instead of being squeezed into a positive fit.

**The big one: the multimodality was initialization, not the model
([#46](https://github.com/jerome-white/leaderboard-item-response/issues/46), [#60](https://github.com/jerome-white/leaderboard-item-response/issues/60), [#62](https://github.com/jerome-white/leaderboard-item-response/issues/62)).** After the changes above, essentially every parameter
(about 99% of `alpha`, `beta` and `theta`) had R-hat > 1.1 on real data.
Six model-side explanations were tested and none moved that number: a tighter
`alpha` prior ([#52](https://github.com/jerome-white/leaderboard-item-response/issues/52)), a different anchor choice ([#56](https://github.com/jerome-white/leaderboard-item-response/issues/56)), excluding merged models
([#57](https://github.com/jerome-white/leaderboard-item-response/issues/57)), group-varying `theta` by model type ([#58](https://github.com/jerome-white/leaderboard-item-response/issues/58)), group-level differential
item functioning ([#59](https://github.com/jerome-white/leaderboard-item-response/issues/59)), and fully hierarchical `alpha`/`beta` ([#53](https://github.com/jerome-white/leaderboard-item-response/issues/53)). What
did work, with the model untouched, was giving all chains one shared,
data-informed starting point instead of CmdStan's independent random
starts:

| `mmlu/computer-science`, R-hat > 1.1 | `alpha` | `beta` | `theta` |
|---|---|---|---|
| random init per chain (default) | 99.8% | 99.5% | 98.5% |
| shared, data-informed init | 0.2% | 26.6% | 0.0% |

The init is closed-form (`alpha = 1`, `beta = logit(1 - accuracy)`,
`theta = logit(accuracy)`; `src/model/to-init.py`). Effective sample sizes
confirmed real convergence rather than chains frozen at a shared start.
Chains started at random scattered into different basins before warmup could
pull them together. In hindsight the model-side hypotheses above failed because
every one of them was sampled with random init.

**Warmup of 1000 is enough, but cost grows fast ([#62](https://github.com/jerome-white/leaderboard-item-response/issues/62)).** Validated on three
subjects spanning 3.5x in item count. Runtime scales roughly as `N^1.56` in
the number of items.

**Whole-suite convergence ([#66](https://github.com/jerome-white/leaderboard-item-response/issues/66)).** Across all 13 MMLU subjects (61.1 hours),
`theta` converges everywhere (0% bad R-hat), `alpha` nearly everywhere, and
`beta` has a residual of 8 to 31% bad R-hat that varies by subject
independently of item count.

**Convergence is not fit: what the residual `beta` problems look like
([#66](https://github.com/jerome-white/leaderboard-item-response/issues/66), tooling in [#68](https://github.com/jerome-white/leaderboard-item-response/issues/68)).** Looking at the worst-converging items per subject
turned up two distinct mechanisms, often mixed in one subject:
- *isolated items* with `alpha` near zero. When discrimination is about zero,
  the likelihood barely depends on `beta`, so `beta` wanders freely. A
  textbook local non-identifiability, and plausibly a bad or ambiguous question.
- *shared-ridge clusters*, groups of `beta`s whose chains drift together
  (pairwise correlations of 0.9 to 0.98), pointing to a common confound
  rather than separate problems.

Further results:
- *Item information vs. convergence, whole item bank.* In 11 of 13 subjects,
  an item's classical information (`rate * (1 - rate)`, peaking at 50%
  correct) correlates positively with its `beta` R-hat (Spearman 0.22 to
  0.67; biology is the flat outlier at -0.05). The most informative items
  converge worst, not the floor/ceiling ones.
- *Shared-ridge membership vs. information: a null.* Checking whether the
  worst-k shared-ridge items are just the high-information ones found no
  stable relationship. The pooled correlation was an artifact of restricted
  range in several subjects, and the real ones disagreed in sign. (This is
  a different question from the whole-bank result above, so the two don't
  contradict each other.)
- *Sampler health.* Standard NUTS diagnostics are clean on every subject: 0
  divergences, BFMI between 0.58 and 1.01. The sampler isn't broken. The
  problem is individual items that don't converge.
- *Seed sensitivity.* For `history`, the same model, data and init gave 2 of
  381 bad-R-hat `beta`s with seed 7 and 115 of 381 (30.2%) with the
  production run's default seed. That is a swing of roughly 60x from the seed
  alone, big enough to swamp the effect of the prior and hierarchical
  variants I tried. It's one subject and two seeds, so treat it as a
  warning, not an estimate ([#66](https://github.com/jerome-white/leaderboard-item-response/issues/66), [#72](https://github.com/jerome-white/leaderboard-item-response/issues/72)).

**Models are not independent test takers.** A premise of IRT is independent
examinees. Many leaderboard "models" are fine-tunes, merges and distillations
of the same few bases. Corroborated merges are modestly more correlated with
each other, but excluding them ([#57](https://github.com/jerome-white/leaderboard-item-response/issues/57)) or letting `theta` vary by group ([#58](https://github.com/jerome-white/leaderboard-item-response/issues/58))
didn't change convergence. This remains an assumption violation to keep in
mind, not one that has been resolved.

**A take on framing.** Full Bayes with multiple independently started chains
gave me a diagnostic (cross-chain R-hat/ESS) that exposed a silent failure
mode which a single-start optimizer or mean-field variational fit would not
reveal. That is an advantage in *diagnosability*, not evidence that MCMC is
inherently "more honest". Chain-initialization sensitivity is not new in
itself. If there is something worth reporting, it is the combination:
a rigorous multi-chain workflow applied to LLM-evaluation data, where the usual
IRT assumptions (independent examinees, a single latent trait) are more
strained than in human testing.

**Open: does any of this change a leaderboard? ([#70](https://github.com/jerome-white/leaderboard-item-response/issues/70), in progress)**
Everything above is about the fit. The question that connects back to LLM
evaluation is whether the pathologies move `theta`, the thing a leaderboard
would read off. The planned checks, cheapest first:
- refit `history` with items removed and compare model rankings with the
  full fit by Spearman correlation ([#71](https://github.com/jerome-white/leaderboard-item-response/issues/71)). The data prep is done: it drops
  all 115 of history's 381 items whose `beta` has R-hat > 1.1, a deliberately
  broad first pass. A second pass with only the 8 mechanistically flagged
  items may follow ([#80](https://github.com/jerome-white/leaderboard-item-response/issues/80)). The refit ([#78](https://github.com/jerome-white/leaderboard-item-response/issues/78)) hasn't been run and the comparison
  script ([#79](https://github.com/jerome-white/leaderboard-item-response/issues/79)) doesn't exist yet. Whether having about 30% of items with
  elevated R-hat moves `theta` at all is the open question; it is not
  evidence that the leaderboard is wrong.
- how much seed variance alone moves `theta` ([#72](https://github.com/jerome-white/leaderboard-item-response/issues/72))
- if rankings do move, which mechanism drives it ([#73](https://github.com/jerome-white/leaderboard-item-response/issues/73)) and whether the
  ordering of models that matter flips ([#74](https://github.com/jerome-white/leaderboard-item-response/issues/74))
- whether the pathological items coincide with MMLU-Redux's human-flagged
  label errors ([#69](https://github.com/jerome-white/leaderboard-item-response/issues/69))

At the time of writing, none of these has produced a result yet.
