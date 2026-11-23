# leaderboard-item-response

Bayesian item response theory (IRT) applied to per-question results from the
Hugging Face [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard).

> **Status:** an exploration, not a finished method. It has no fixed target;
> the question being asked keeps getting defined by what the last answer
> turned up.

## What this is

I started this repository to practice Bayesian data analysis on a domain I
care about: evaluating large language models. The leaderboard published
per-question results for thousands of models, which is exactly the shape of
data IRT was built for: many "test takers" (models), many items (benchmark
questions), and a correct/incorrect response for each pair.

Each model gets an *ability* (`theta`) and each question a *difficulty*
(`beta`) and a *discrimination* (`alpha`), in a two-parameter logistic model
fit with full Bayesian inference (Stan / MCMC):

```
P(model j answers question i correctly) = logistic( alpha_i * (theta_j - beta_i) )
```

The path so far: "can I get this data and fit this model at all?", then
"does the fit converge?", then "does the model fit the data?", and now "do the
problems I found change which models are ranked above which?" Two goals
coexist: a working end-to-end pipeline, and a personal learning project in
Bayesian workflow, so some choices are deliberately exploratory. Current
scope: the 13 MMLU-Pro subject groups from the leaderboard's v2 era (June 2024
to March 2025), each fit as its own model.

## What I've seen so far

These are observations from an ongoing exploration, not firm conclusions; the
[journal](docs/journal.md) has the details and caveats.

- **The persistent non-convergence seemed to be initialization, not the
  model.** About 99% of parameters had R-hat > 1.1 on real data. Six
  model-side changes (priors, anchors, hierarchy, model-type groups) didn't
  move that. Giving all chains one shared, data-informed starting point instead
  of CmdStan's independent random starts did help, with the model untouched:

  | `mmlu/computer-science`, R-hat > 1.1 | `alpha` | `beta` | `theta` |
  |---|---|---|---|
  | random init per chain (default) | 99.8% | 99.5% | 98.5% |
  | shared, data-informed init | 0.2% | 26.6% | 0.0% |

- **Converging isn't the same as fitting.** Across all 13 subjects the
  sampler looked healthy (no divergences) and `theta` converged everywhere, but
  8 to 31% of `beta`s still didn't. The worst ones suggest at least two
  patterns: isolated near-zero-discrimination items, and clusters of items
  whose chains drift together.
- **Models may not be independent test takers**, which IRT assumes: many
  leaderboard models appear to be merges and fine-tunes of the same bases.
- **Open:** whether any of this changes a ranking. No result yet.

## More

- [Installing and running](docs/running.md): code layout, requirements, and how
  to run each stage. Only ever run on a 2024 Mac mini (M4 Pro, 64 GB); other
  machines are untested. Full runs take tens of hours.
- [Journal](docs/journal.md): my notebook of what I've seen and explored, newest first.
- [Related work](docs/related-work.md): other papers using IRT on LLM
  evaluation, plus notes on Elo vs. 2PL and directions to take next.
