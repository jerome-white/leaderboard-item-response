# Related work

Back to the [README](../README.md).

Other papers that apply item response theory (IRT) to LLM or NLP evaluation.
The descriptions below were checked against each paper's abstract page (or
the venue's proceedings page). **The papers have not been read in full**, so
where a claim goes beyond the abstract it is posed as a question.

## Papers using IRT for LLM / NLP evaluation

- **metabench** (Kipnis, Voudouris, Schulze Buschoff, Schulz; ICLR 2025;
  [arXiv:2407.12844](https://arxiv.org/abs/2407.12844)). Uses item-level data
  from over 5,000 LLMs on six Open LLM Leaderboard benchmarks (ARC, GSM8K,
  HellaSwag, MMLU, TruthfulQA, WinoGrande) to select a sparse set of
  informative items, under 3% of the combined size, using IRT. The closest
  data overlap with this project among the papers listed here.
- **tinyBenchmarks: evaluating LLMs with fewer examples** (Maia Polo et al.,
  6 authors; ICML 2024; [arXiv:2402.14992](https://arxiv.org/abs/2402.14992)).
  Shows that, for example, MMLU performance can be estimated from 100 curated
  examples, and releases tiny versions of the Open LLM Leaderboard, MMLU, HELM
  and AlpacaEval 2.0. The abstract does not mention IRT; whether and how the
  method uses it is something to confirm from the paper.
- **Rodriguez et al., "Evaluation Examples are not Equally Informative: How
  should that change NLP Leaderboards?"** (ACL-IJCNLP 2021;
  [ACL Anthology](https://aclanthology.org/2021.acl-long.346/)). A Bayesian
  leaderboard model in which latent subject skill and latent item difficulty
  predict correct responses. The authors report it can guide what to annotate,
  identify annotation errors, detect overfitting, and identify informative
  examples.
- **Vania et al., "Comparing Test Sets with Item Response Theory"**
  (ACL-IJCNLP 2021; [ACL Anthology](https://aclanthology.org/2021.acl-long.92/)).
  Uses IRT on 29 datasets with predictions from 18 pretrained Transformer
  models, and concludes that many of the datasets are largely saturated.
- **Lalor, Wu, Yu, "Building an Evaluation Scale using Item Response
  Theory"** (EMNLP 2016; [arXiv:1605.08889](https://arxiv.org/abs/1605.08889)).
  Proposes IRT as an alternative way to build gold-standard test sets and
  evaluate NLP systems, demonstrated on Recognizing Textual Entailment.
- **py-irt** (Lalor and Rodriguez; [arXiv:2203.01282](https://arxiv.org/abs/2203.01282)).
  A Python IRT library built on PyTorch and Pyro, fitting models by
  variational inference. Its PyPI page points to an EACL 2024 tutorial on IRT
  for NLP; the tutorial itself hasn't been looked at.
- **Zhou et al., "Lost in Benchmarks? Rethinking Large Language Model
  Benchmarking with Item Response Theory"** (13 authors; AAAI 2026;
  [arXiv:2505.15055](https://arxiv.org/abs/2505.15055)). Proposes a
  neural IRT variant (PSN-IRT), analyzes 11 LLM benchmarks with 41,871 items,
  and reports "significant and varied shortcomings" in their measurement
  quality.
- **Land and Bikel, "Auditing LLM Benchmarks with Item Response Theory"**
  (EMNLP 2026; [arXiv:2605.30504](https://arxiv.org/abs/2605.30504)). An
  IRT-based indicator for surfacing likely mislabeled benchmark items, with a
  reported 95% precision in the top 200 examples, on seven benchmarks using
  responses from 114 models.
- **Yao et al., "JE-IRT: A Geometric Lens on LLM Abilities through Joint
  Embedding Item Response Theory"** (TMLR;
  [arXiv:2509.22888](https://arxiv.org/abs/2509.22888)). Embeds LLMs and
  questions in a shared space, with question direction encoding semantics and
  norm encoding difficulty; a new LLM is added by fitting a single embedding.
- **MMLU-Redux** (Gema et al., "Are We Done with MMLU?", 16 authors; NAACL
  2025; [ACL Anthology](https://aclanthology.org/2025.naacl-long.262/)).
  Not IRT. Manually re-annotates 5,700 MMLU questions across all 57 subjects
  and reports that 57% of the analysed questions in the Virology subset
  contain errors. A human-labelled counterpart to what this project's item
  parameters might surface (planned comparison: issue
  [#69](https://github.com/jerome-white/leaderboard-item-response/issues/69)).

## How this project relates, as open questions

- Several of these papers use IRT to *do something with* the item parameters:
  shrink a benchmark (metabench, tinyBenchmarks) or find bad items (Rodriguez,
  Land and Bikel). This repo spends most of its effort on whether the
  full-Bayes 2PL fit itself can be trusted. Is that actually less explored, or
  do these papers already address it?
- How does each fit its model (maximum likelihood, variational inference,
  MCMC)? Do any run several independently initialized chains and report R-hat
  or effective sample size, the way this project found it needed to? That
  hasn't been checked, so whether the initialization sensitivity described in
  the [journal](journal.md) is known in this literature is unknown.
- metabench uses the same family of leaderboard data (v1 benchmarks, over
  5,000 models) while this project uses the v2 leaderboard's MMLU-Pro. How
  do its item parameters compare on overlapping benchmarks?

## Further notes

Earlier notes on where this might go, kept as questions and starting points,
not conclusions.

### Elo, Bradley-Terry and 2PL

Bradley-Terry, the paired-comparison model, has one strength parameter per
player, which looks closely analogous to the one-parameter (Rasch, 1PL) IRT
model. Elo is commonly described as an online way of updating such strengths.
2PL adds a per-item discrimination `alpha` that neither has.

- The Chatbot Arena paper ([Chiang et al.,
  arXiv:2403.04132](https://arxiv.org/abs/2403.04132)) says earlier versions
  of the platform reported Elo scores rather than Bradley-Terry coefficients,
  and that they changed "because the BT coefficients are better for the
  purpose of statistical estimation". Is the online-Elo variance and order
  dependence the only reason, or the main one?
- If `alpha` marks which items are worth trusting (low `alpha` for noisy,
  guessable or mislabeled items), is it fair to say pairwise leaderboards
  lack the equivalent? Does anyone model per-comparison discrimination in
  Bradley-Terry? The Arena team's post
  ["Statistical Extensions of the Bradley-Terry and Elo Models"](https://arena.ai/blog/extended-arena/)
  extends the model with a feature vector per battle (for systems made of
  several subsystems). It also links to a separate post on style control,
  which hasn't been read, so how close it is remains unknown.
- For multiple-choice data like MMLU-Pro, would a guessing parameter (3PL,
  with a floor near 1/number of choices) fit better than 2PL?

### Variants of 2PL to consider

For a static item bank with many models, like this repo's:

- **Multidimensional IRT.** "LLM ability" probably isn't one number: would a
  vector `theta` fit better? Is it a higher-value change than the others
  below? This is an untested guess.
- **Explanatory IRT**, where item parameters depend on item metadata (subject,
  format) rather than being free per item.
- **Embedding-based or neural IRT.** JE-IRT above, and PSN-IRT in Zhou et al.
  EmbedLLM was also suggested in earlier notes but couldn't be verified. If
  item parameters can be predicted from the question text, could that help
  with the missing-data problem this project ran into? That's a question for
  the papers, not something I've confirmed.

For a pairwise, Arena-style extension, a per-comparison discrimination term in
Bradley-Terry is one possible direction. I haven't found an established model
that does exactly this, but the search wasn't thorough, so whether it is
novel is an open question.
