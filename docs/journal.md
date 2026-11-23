# Journal

> **A note on what this is.** This is a notebook: observations from an
> exploration, saved so they aren't lost. The entries are not statements of
> certainty, and they aren't positions I'm committed to. Some will turn out to
> be wrong or incomplete, and several rest on a handful of runs. This is a
> high-level summary; the issues and pull requests linked from each entry
> have the detail. Dates are approximate (month, from commit and issue
> history). Newest first. Back to the [README](../README.md).

## 2026-11 (open): does any of this change a leaderboard? ([#70](https://github.com/jerome-white/leaderboard-item-response/issues/70))

The entries below are about the fit. The open question is whether the fit
problems move `theta`, which is what a leaderboard would read off. Planned,
cheapest first: refit `history` with problem items removed and compare model
rankings by Spearman correlation ([#71](https://github.com/jerome-white/leaderboard-item-response/issues/71), [#78](https://github.com/jerome-white/leaderboard-item-response/issues/78), [#79](https://github.com/jerome-white/leaderboard-item-response/issues/79)); measure how much seed
variance alone moves `theta` ([#72](https://github.com/jerome-white/leaderboard-item-response/issues/72)); and check whether the problem items
coincide with MMLU-Redux's human-flagged errors ([#69](https://github.com/jerome-white/leaderboard-item-response/issues/69)). The data prep for the
first is done (it drops the 115 of `history`'s 381 items with `beta` R-hat >
1.1, a deliberately broad first pass), but none of these has produced a result.

## 2026-11: Converging and fitting are different questions ([#66](https://github.com/jerome-white/leaderboard-item-response/issues/66))

After the initialization change below, the problems that remained were examined. The
worst-converging items seemed to fall into two patterns: isolated items with
`alpha` near zero (where `beta` is barely constrained, consistent with a local
non-identifiability), and clusters of items whose chains drift together. In 11
of 13 subjects, more informative items (those near 50% correct) tended to
converge worse. A check of whether the clusters were just the high-information
items found no stable relationship. Standard NUTS diagnostics looked clean (no
divergences). One caution: for `history`, the same model, data and init gave 2
of 381 bad-R-hat `beta`s with one seed and 115 of 381 with another. That's one
subject and two seeds, but it is large enough that single-seed comparisons
between model variants may not mean much.

## 2026-10 to 2026-11: Full suite, warmup and cost ([#62](https://github.com/jerome-white/leaderboard-item-response/issues/62), [#66](https://github.com/jerome-white/leaderboard-item-response/issues/66))

Across all 13 MMLU subjects (61.1 hours), `theta` converged everywhere, `alpha`
nearly everywhere, and `beta` had a residual of 8 to 31% bad R-hat that varied
by subject. A warmup of 1000 seemed enough on the three subjects checked (3.5x
range in item count). Runtime grew faster than linearly in the number of
items, roughly `N^1.56` from two data points.

## 2026-10: Chain initialization, not the model, seemed to be the problem ([#46](https://github.com/jerome-white/leaderboard-item-response/issues/46), [#60](https://github.com/jerome-white/leaderboard-item-response/issues/60), [#62](https://github.com/jerome-white/leaderboard-item-response/issues/62))

Almost every parameter (about 99%) had R-hat > 1.1 on real data. Six
model-side changes ([#52](https://github.com/jerome-white/leaderboard-item-response/issues/52), [#53](https://github.com/jerome-white/leaderboard-item-response/issues/53), [#56](https://github.com/jerome-white/leaderboard-item-response/issues/56), [#57](https://github.com/jerome-white/leaderboard-item-response/issues/57), [#58](https://github.com/jerome-white/leaderboard-item-response/issues/58), [#59](https://github.com/jerome-white/leaderboard-item-response/issues/59)) didn't move that. Starting
all chains from one shared, data-informed point instead of CmdStan's
independent random starts did, with the model untouched:

| `mmlu/computer-science`, R-hat > 1.1 | `alpha` | `beta` | `theta` |
|---|---|---|---|
| random init per chain (default) | 99.8% | 99.5% | 98.5% |
| shared, data-informed init | 0.2% | 26.6% | 0.0% |

(One seed, 250 warmup and 250 samples; longer runs gave a similar picture.)
The init is closed-form (`src/model/to-init.py`). A possible reading is that
random starts scattered the chains into different modes before warmup could
pull them together. A related one is that the six model-side tests were all
run with random init, so they could not have shown a benefit. Both readings
come from one subject at full scale plus follow-ups, not a systematic study.

## 2026-10: Model setup decisions ([#22](https://github.com/jerome-white/leaderboard-item-response/issues/22), [#37](https://github.com/jerome-white/leaderboard-item-response/issues/37), [#45](https://github.com/jerome-white/leaderboard-item-response/issues/45), [#47](https://github.com/jerome-white/leaderboard-item-response/issues/47))

As written, the 2PL model isn't identified: scaling `alpha` by `c` and
`theta`, `beta` by `1/c` leaves the likelihood unchanged, as does shifting
`theta` and `beta` together. Anchoring two abilities ([#47](https://github.com/jerome-white/leaderboard-item-response/issues/47)) addresses this; a
first attempt, hard-standardizing `theta` inside the model, made the sampler
very slow. Elsewhere, the `alpha > 0` constraint was dropped ([#22](https://github.com/jerome-white/leaderboard-item-response/issues/22)) in case
mis-keyed questions show up as negative discrimination, and MMLU is fit one
subject at a time instead of by super-category ([#37](https://github.com/jerome-white/leaderboard-item-response/issues/37)).

## 2026-09 to 2026-10: Data collection took a lot of the work

The leaderboard's per-question files have gated repos, rate limits,
no-author model names, and multiple result files per model, and much of the
early history is making collection resilient ([#3](https://github.com/jerome-white/leaderboard-item-response/issues/3) to [#34](https://github.com/jerome-white/leaderboard-item-response/issues/34)). `doc_hash` appears
to drift across harness and dataset revisions, so question identity is keyed
on `doc_id` instead ([#21](https://github.com/jerome-white/leaderboard-item-response/issues/21), [#26](https://github.com/jerome-white/leaderboard-item-response/issues/26)). The leaderboard's v1 and v2 eras look
incompatible; this repo uses v2 only.

## Ongoing: Models probably aren't independent test takers

IRT assumes independent examinees, and many leaderboard "models" seem to be
fine-tunes, merges and distillations of the same few bases. In one check,
corroborated merges were somewhat more correlated with each other, but
excluding them ([#57](https://github.com/jerome-white/leaderboard-item-response/issues/57)) or letting `theta` vary by group ([#58](https://github.com/jerome-white/leaderboard-item-response/issues/58)) didn't change
convergence. A possible assumption violation, not yet characterized.

## Ongoing: Why has this been harder than it looked?

I wanted to understand why this exploration felt so much harder than what
others seemed to be doing. One idea, which came out of a conversation with an
AI model and which I'm keeping as a hypothesis: running several independently
started chains and checking that they agree (R-hat, ESS) may have made a
problem visible that other fitting approaches can hide. A maximum-likelihood
fit from a single start, or a mean-field variational fit, doesn't obviously
come with an equivalent check unless you add one, for example by restarting
from several initializations. Whether those approaches run into the same
problem on LLM evaluation data is an open question (see
[related work](related-work.md)).

Initialization sensitivity in MCMC is well known in general, so the finding
itself probably isn't new. The combination may be what's interesting: a
multi-chain workflow on LLM evaluation data, where the usual IRT assumptions
(independent examinees, a single latent trait) may hold less well than in
human testing. Whether any of this is worth writing up is open.
