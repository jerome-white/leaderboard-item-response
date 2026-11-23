# Related work

Back to the [README](../README.md).

## Papers that use IRT on LLM benchmarks
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
workflow on LLM-evaluation IRT, but the literature check was shallow. See "Further notes" below for the longer notes (Elo vs. 2PL, MIRT and
embedding-based IRT, and a possible "2PL for pairwise comparisons").

## Further notes

From a 2026-07 research pass on where this project sits and where it could go.

### Elo vs. 2PL

It's tempting to think of Elo as "the maximum-likelihood version of 2PL." That's half right.

Precisely: **Bradley-Terry** — the model most "Elo" systems are actually estimating — is the paired-comparison analog of the **Rasch model (1PL)**: one strength/ability parameter per player, no per-item discrimination. Classic sequential Elo is a noisy, order-dependent *online* approximation of the MLE for that Bradley-Terry model. Confirmed concretely: [Chiang et al.'s Chatbot Arena paper](https://arxiv.org/abs/2403.04132) explicitly moved Arena off online Elo onto a centralized regularized-logistic-regression fit of Bradley-Terry, precisely because online Elo's variance and order-dependence were a problem.

So: Elo ≈ approximate MLE of Bradley-Terry ≈ 1PL, not 2PL. The discrimination parameter α is what 2PL adds that neither Elo nor Arena's Bradley-Terry has, and it's not a minor add-on. α tells you which items are worth trusting: an item with high α cleanly separates strong from weak models near its difficulty; low α means the item is noisy, guessable, or mislabeled and shouldn't move the ranking much. That's exactly the lever metabench, tinyBenchmarks, and Land's auditing paper all use, and it's structurally absent from any pure Elo/Bradley-Terry leaderboard. For MCQ formats like MMLU-Pro specifically, 3PL (a guessing-floor asymptote near 1/n_choices) would likely add more than 2PL alone — Bradley-Terry has no equivalent concept at all.

One nuance worth tracking: [Arena's newer "style-controlled" ranking](https://arena.ai/blog/extended-arena/) adds per-prompt covariates into the Bradley-Terry log-odds — a step toward explanatory-IRT territory, but nobody has shipped a clean, named "2PL-for-pairwise-comparisons" model. That's an actual gap, not an oversight (see §3).

### Modern 2PL variants

#### For this repo's data (static item bank, many models)

- **Multidimensional IRT (MIRT)** — a vector θ instead of a scalar, since "LLM ability" clearly isn't one dimension (a model can be strong at math, weak at instruction-following). Probably the single highest-leverage upgrade to the current model.
- **Explanatory/covariate IRT** — item parameters as a function of item metadata (subject, benchmark, format) rather than free per-item parameters.
- **Embedding-based / neural IRT** — [JE-IRT](https://arxiv.org/abs/2509.22888), EmbedLLM, PSN-IRT: predict item difficulty/discrimination from the question's text embedding instead of requiring dense response coverage. Worth flagging specifically — this structurally sidesteps the missing-data problem this project ran into, rather than just tolerating it as missing-at-random.

#### For a pairwise/Arena-style extension

There's no established "2PL for paired comparisons" in the literature — which means building one (adding a per-prompt discrimination term to Bradley-Terry, so some battles count more toward the ranking than others) would be a genuinely novel contribution, not a rehash of Arena's own covariate approach. Given the discussion in §1 about whether this kind of work is worth pursuing, this is probably the more interesting and more original of the two paths available — live data, an actual modeling gap, and a natural extension of the 2PL machinery already built here.
