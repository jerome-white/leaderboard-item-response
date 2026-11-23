# Journal

Big-ticket conclusions, newest first. Dates are approximate (month, from
commit and issue history). Issue numbers point to the evidence. Back to the
[README](../README.md).

## 2026-11 (open): does any of this change a leaderboard? ([#70](https://github.com/jerome-white/leaderboard-item-response/issues/70))

Everything below is about the fit. The question that connects back to LLM
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

## 2026-11: Convergence is not fit: what the residual `beta` problems look like ([#66](https://github.com/jerome-white/leaderboard-item-response/issues/66), tooling in [#68](https://github.com/jerome-white/leaderboard-item-response/issues/68))

Looking at the worst-converging items per subject
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

## 2026-11: Whole-suite convergence ([#66](https://github.com/jerome-white/leaderboard-item-response/issues/66))

Across all 13 MMLU subjects (61.1 hours),
`theta` converges everywhere (0% bad R-hat), `alpha` nearly everywhere, and
`beta` has a residual of 8 to 31% bad R-hat that varies by subject
independently of item count.

## 2026-10: Warmup of 1000 is enough, but cost grows fast ([#62](https://github.com/jerome-white/leaderboard-item-response/issues/62))

Validated on three
subjects spanning 3.5x in item count. Runtime scales roughly as `N^1.56` in
the number of items.

## 2026-10: The big one: the multimodality was initialization, not the model ([#46](https://github.com/jerome-white/leaderboard-item-response/issues/46), [#60](https://github.com/jerome-white/leaderboard-item-response/issues/60), [#62](https://github.com/jerome-white/leaderboard-item-response/issues/62))

After the changes above, essentially every parameter
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

## 2026-10: The 2PL model is not identified as written, and anchoring fixed that ([#45](https://github.com/jerome-white/leaderboard-item-response/issues/45), [#47](https://github.com/jerome-white/leaderboard-item-response/issues/47))

`alpha`, `beta` and `theta` can trade off exactly: scaling
`alpha` by `c` and `theta`, `beta` by `1/c` leaves the likelihood unchanged,
and shifting `theta` and `beta` together does too. Pinning two abilities
(`theta[1]`, `theta[2]`) removes it. A first attempt, hard-standardizing
`theta` inside the model, was mathematically right but gave the sampler such
bad geometry that it effectively stalled.

## 2026-10: Allowing negative discrimination surfaces bad items ([#22](https://github.com/jerome-white/leaderboard-item-response/issues/22), [#35](https://github.com/jerome-white/leaderboard-item-response/issues/35))

the `alpha > 0` constraint lets mis-keyed or ambiguous questions show up as
negative discrimination instead of being squeezed into a positive fit.

## 2026-10: Per-subject models, not grouped MMLU ([#37](https://github.com/jerome-white/leaderboard-item-response/issues/37))

Fitting MMLU super-categories
together was retired in favor of one model per subject ([#37](https://github.com/jerome-white/leaderboard-item-response/issues/37)).

## 2026-09 to 2026-10: Data collection is most of the work

The leaderboard's per-question files
have gated repos, rate limits, canonical (no-author) model names, and
multiple result files per model. A large share of the early history is
making collection resilient ([#3](https://github.com/jerome-white/leaderboard-item-response/issues/3) to [#34](https://github.com/jerome-white/leaderboard-item-response/issues/34)). One finding that shaped the model
inputs: `doc_hash` is a hash of the raw document dict and drifts across
harness and dataset revisions, so question identity is keyed on `doc_id`
instead ([#21](https://github.com/jerome-white/leaderboard-item-response/issues/21), [#26](https://github.com/jerome-white/leaderboard-item-response/issues/26)). The leaderboard has two incompatible eras (v1, v2); this
repo uses v2 only.

## Ongoing: Models are not independent test takers

A premise of IRT is independent
examinees. Many leaderboard "models" are fine-tunes, merges and distillations
of the same few bases. Corroborated merges are modestly more correlated with
each other, but excluding them ([#57](https://github.com/jerome-white/leaderboard-item-response/issues/57)) or letting `theta` vary by group ([#58](https://github.com/jerome-white/leaderboard-item-response/issues/58))
didn't change convergence. This remains an assumption violation to keep in
mind, not one that has been resolved.

## Ongoing: A take on framing

Full Bayes with multiple independently started chains
gave me a diagnostic (cross-chain R-hat/ESS) that exposed a silent failure
mode which a single-start optimizer or mean-field variational fit would not
reveal. That is an advantage in *diagnosability*, not evidence that MCMC is
inherently "more honest". Chain-initialization sensitivity is not new in
itself. If there is something worth reporting, it is the combination:
a rigorous multi-chain workflow applied to LLM-evaluation data, where the usual
IRT assumptions (independent examinees, a single latent trait) are more
strained than in human testing.

