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

## What has been found so far

- **The persistent non-convergence was initialization, not the model.** About
  99% of parameters had R-hat > 1.1 on real data. Six model-side fixes
  (priors, anchors, hierarchy, model-type groups) didn't move that. Giving all
  chains one shared, data-informed starting point instead of CmdStan's
  independent random starts did, with the model untouched:

  | `mmlu/computer-science`, R-hat > 1.1 | `alpha` | `beta` | `theta` |
  |---|---|---|---|
  | random init per chain (default) | 99.8% | 99.5% | 98.5% |
  | shared, data-informed init | 0.2% | 26.6% | 0.0% |

- **Convergence is not fit.** Across all 13 subjects the sampler is healthy (no
  divergences) and `theta` converges everywhere, but 8 to 31% of `beta`s still
  don't. They come from at least two mechanisms: isolated near-zero
  discrimination items, and clusters of items whose chains drift together.
- **Models aren't independent test takers**, which IRT assumes: many
  leaderboard models are merges and fine-tunes of the same bases.
- **Open:** whether any of this changes a ranking. No result yet.

The details, evidence and dates are in the [journal](docs/journal.md).

## More

- [Installing and running](docs/running.md): code layout, requirements, and how
  to run each stage. Only ever run on a 2024 Mac mini (M4 Pro, 64 GB); other
  machines are untested. Full runs take tens of hours.
- [Journal](docs/journal.md): what was learned and explored, newest first.
- [Related work](docs/related-work.md): other papers using IRT on LLM
  evaluation, plus notes on Elo vs. 2PL and directions to take next.
