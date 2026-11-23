# Journal

> **A note on what this is.** This is a notebook: observations from my
> exploration that I want to save. The entries are not statements of certainty,
> and they aren't things I'm strongly committed to. Some will turn out to be
> wrong or incomplete, and several rest on a handful of runs. Dates are
> approximate (month, from commit and issue history), and issue numbers point
> to the evidence. Newest first. Back to the [README](../README.md).

## 2026-11 (open): does any of this change a leaderboard? ([#70](https://github.com/jerome-white/leaderboard-item-response/issues/70))

Everything below is about the fit. The question that connects back to LLM
evaluation is whether the fit problems move `theta`, the thing a leaderboard
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
- whether the problem items coincide with MMLU-Redux's human-flagged
  label errors ([#69](https://github.com/jerome-white/leaderboard-item-response/issues/69))

At the time of writing, none of these has produced a result yet.

## 2026-11: Convergence and fit are different questions ([#66](https://github.com/jerome-white/leaderboard-item-response/issues/66), tooling in [#68](https://github.com/jerome-white/leaderboard-item-response/issues/68))

Looking at the worst-converging items in each subject suggested two
mechanisms, often mixed within one subject:
- *Isolated items* with `alpha` near zero. When discrimination is about zero,
  the likelihood depends very little on `beta`, so `beta` is barely
  constrained. That is consistent with a local non-identifiability, and these
  might be bad or ambiguous questions, though I haven't checked individual
  items.
- *Clusters of items* whose chains drift together (pairwise correlations of
  0.9 to 0.98). That might point to a shared cause rather than separate
  problems, but I don't know what the cause is.

Other observations:
- *Item information vs. convergence, whole item bank.* In 11 of 13 subjects,
  an item's classical information (`rate * (1 - rate)`, peaking at 50%
  correct) correlated positively with its `beta` R-hat (Spearman 0.22 to
  0.67; biology is the flat outlier at -0.05). So more informative items
  seemed to converge worse, not the floor/ceiling ones. I don't have an
  explanation.
- *Cluster membership vs. information: no relationship found.* I checked
  whether the clustered items are just the high-information ones. The pooled
  correlation turned out to be an artifact of restricted range in several
  subjects, and the per-subject ones disagreed in sign. This is a different
  question from the whole-bank result above, so the two don't contradict each
  other.
- *Sampler health.* Standard NUTS diagnostics looked clean on every subject:
  0 divergences, BFMI between 0.58 and 1.01. The sampler doesn't look
  broken; the trouble seems to be at the level of individual items.
- *Seed sensitivity.* For `history`, the same model, data and init gave 2 of
  381 bad-R-hat `beta`s with seed 7 and 115 of 381 (30.2%) with the
  production run's default seed. That is roughly a 60x swing from the seed
  alone, large enough to swamp the effect of the prior and hierarchical
  variants I tried. It's one subject and two seeds, so it's a warning, not an
  estimate ([#66](https://github.com/jerome-white/leaderboard-item-response/issues/66), [#72](https://github.com/jerome-white/leaderboard-item-response/issues/72)).

## 2026-11: Whole-suite convergence ([#66](https://github.com/jerome-white/leaderboard-item-response/issues/66))

Across all 13 MMLU subjects (61.1 hours), `theta` converged everywhere (0% bad
R-hat) and `alpha` nearly everywhere. `beta` had a residual of 8 to 31% bad
R-hat that varied by subject, with no obvious relationship to item count.

## 2026-10: Warmup of 1000 seemed enough, but cost grows fast ([#62](https://github.com/jerome-white/leaderboard-item-response/issues/62))

Checked on three subjects spanning 3.5x in item count. Runtime scaled roughly
as `N^1.56` in the number of items, from two data points, so the exponent is
rough.

## 2026-10: Chain initialization, not the model, seemed to be the problem ([#46](https://github.com/jerome-white/leaderboard-item-response/issues/46), [#60](https://github.com/jerome-white/leaderboard-item-response/issues/60), [#62](https://github.com/jerome-white/leaderboard-item-response/issues/62))

After the changes below, essentially every parameter (about 99% of `alpha`,
`beta` and `theta`) had R-hat > 1.1 on real data. I tried six model-side
explanations and none moved that number: a tighter `alpha` prior ([#52](https://github.com/jerome-white/leaderboard-item-response/issues/52)), a
different anchor choice ([#56](https://github.com/jerome-white/leaderboard-item-response/issues/56)), excluding merged models ([#57](https://github.com/jerome-white/leaderboard-item-response/issues/57)), group-varying
`theta` by model type ([#58](https://github.com/jerome-white/leaderboard-item-response/issues/58)), group-level differential item functioning ([#59](https://github.com/jerome-white/leaderboard-item-response/issues/59)),
and fully hierarchical `alpha`/`beta` ([#53](https://github.com/jerome-white/leaderboard-item-response/issues/53)). What did help, with the model
untouched, was giving all chains one shared, data-informed starting point
instead of CmdStan's independent random starts:

| `mmlu/computer-science`, R-hat > 1.1 | `alpha` | `beta` | `theta` |
|---|---|---|---|
| random init per chain (default) | 99.8% | 99.5% | 98.5% |
| shared, data-informed init | 0.2% | 26.6% | 0.0% |

The init is closed-form (`alpha = 1`, `beta = logit(1 - accuracy)`,
`theta = logit(accuracy)`; `src/model/to-init.py`). Effective sample sizes
were consistent with real convergence rather than chains frozen at a shared
start. My reading is that random starts scattered the chains into different
modes before warmup could pull them together. A related reading, from a later
check ([#66](https://github.com/jerome-white/leaderboard-item-response/issues/66)), is that the six model-side tests may have failed because they
were all sampled with random init, so they probably couldn't have shown a
benefit. This is from one subject at full scale plus follow-ups, not a
systematic study.

## 2026-10: The 2PL model wasn't identified as written ([#45](https://github.com/jerome-white/leaderboard-item-response/issues/45), [#47](https://github.com/jerome-white/leaderboard-item-response/issues/47))

`alpha`, `beta` and `theta` can trade off exactly: scaling `alpha` by `c` and
`theta`, `beta` by `1/c` leaves the likelihood unchanged, and shifting `theta`
and `beta` together does too. Anchoring two abilities (`theta[1]`,
`theta[2]`) is meant to remove this. A first attempt, hard-standardizing
`theta` inside the model, was mathematically fine but the sampler made almost
no progress on real data.

## 2026-10: Allowing negative discrimination ([#22](https://github.com/jerome-white/leaderboard-item-response/issues/22), [#35](https://github.com/jerome-white/leaderboard-item-response/issues/35))

I dropped the `alpha > 0` constraint, hoping mis-keyed or ambiguous questions
would show up as negative discrimination instead of being squeezed into a
positive fit. I haven't looked closely at whether that happens.

## 2026-10: Per-subject models, not grouped MMLU ([#37](https://github.com/jerome-white/leaderboard-item-response/issues/37))

Fitting MMLU super-categories together was retired in favor of one model per
subject.

## 2026-09 to 2026-10: Data collection took a lot of the work

The leaderboard's per-question files have gated repos, rate limits, canonical
(no-author) model names, and multiple result files per model. A large share of
the early history is making collection resilient ([#3](https://github.com/jerome-white/leaderboard-item-response/issues/3) to [#34](https://github.com/jerome-white/leaderboard-item-response/issues/34)). One thing that
shaped the model inputs: `doc_hash` appears to drift across harness and
dataset revisions, so question identity is keyed on `doc_id` instead ([#21](https://github.com/jerome-white/leaderboard-item-response/issues/21),
[#26](https://github.com/jerome-white/leaderboard-item-response/issues/26)). The leaderboard has two eras (v1, v2) that look incompatible; this repo
uses v2 only.

## Ongoing: Models probably aren't independent test takers

IRT assumes independent examinees. Many leaderboard "models" seem to be
fine-tunes, merges and distillations of the same few bases. In one check,
corroborated merges were somewhat more correlated with each other, but
excluding them ([#57](https://github.com/jerome-white/leaderboard-item-response/issues/57)) or letting `theta` vary by group ([#58](https://github.com/jerome-white/leaderboard-item-response/issues/58)) didn't change
convergence. So I'm treating this as a possible assumption violation to keep
in mind, not as something I've characterized.

## Ongoing: Why has this been harder than it looked?

I wanted to understand why this exploration felt so much harder than what
others seemed to be doing. One idea, which came out of a conversation with an
AI model and which I'm keeping as a hypothesis: running several independently
started chains and checking that they agree (R-hat, ESS) may have made a
problem visible that other fitting approaches can hide. A maximum-likelihood
fit from a single start, or a mean-field variational fit, doesn't obviously
come with an equivalent check unless you add one, for example by restarting
from several initializations and comparing the results. I don't know whether
those approaches, as used on LLM evaluation data, run into the same problem.

Initialization sensitivity in MCMC is well known in general, so I don't think
the finding itself is new. What I'm curious about is the combination: a
multi-chain workflow on LLM evaluation data, where the usual IRT assumptions
(independent examinees, a single latent trait) may hold less well than in
human testing. The merge result above is a hint about that, not a
demonstration. Whether any of this is worth writing up is an open question.
