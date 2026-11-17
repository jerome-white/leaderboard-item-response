#!/bin/bash

# Compares the current alpha prior (normal(1, 1)) against a tighter
# one (normal(1, 0.3)) on the same real, full-scale data. See #52:
# normal(1, 1) puts real mass near alpha = 0, and an item with
# discrimination near zero is a textbook local non-identifiability
# (beta barely affects the likelihood when alpha ~ 0) - a candidate
# explanation for #46's persistent, pervasive multimodality that
# anchor selection (ruled out, see #46) didn't touch. normal(1, 0.3)
# still gives real prior mass to negative discrimination (~2.3% vs
# ~15.9% under normal(1,1)) - #22's intent survives, just needs
# stronger likelihood evidence to win out.
#
# beta is deliberately left untouched, to isolate this one change.
#
# Originally run (and only ever run) against mmlu/computer-science,
# whose worst-k (by ESS/r_hat) is 0% isolated near-zero-alpha items,
# 100% shared-ridge correlated pairs - it has no isolated-item
# mechanism for this fix to touch at all. Repointed at mmlu/history
# for #66's chain-sync follow-up: history's worst-k is a mix (2
# isolated near-zero-alpha items, 6 shared-ridge, confirmed directly
# from chain-sync.csv/selected.csv), so this is the first time the
# tighter prior is tested on a subject that actually has the
# isolated-item mechanism in its worst-k, even though most of
# history's worst-k is still the untouched shared-ridge confound.
#
# IMPORTANT, found the hard way: this script never passed `init=`,
# so both variants were sampling with cmdstan's default
# independent-random-per-chain init - exactly the already-diagnosed,
# already-fixed root cause of #46's pervasive ~99% Rhat-bad signature
# (see #60/PR #62, [[project_issue46_investigation]]). That bug, not
# anything about the alpha prior, is almost certainly why an earlier
# run of this script showed ~99.7% bad Rhat in both variants
# identically. Fixed here to pass the same data-informed init.json
# bin/run.sh uses in production, and bumped num_samples/num_warmup to
# match production (1000/1000) now that proper init should make that
# budget tractable - at 250/250 with no init fix, neither variant was
# ever going to converge far enough to show a real difference.

ROOT=`git rev-parse --show-toplevel`
CMDSTAN=$ROOT/cmdstan-2.40.0

#
#
#
benchmark=mmlu/history
data=$ROOT/data/opt/$benchmark/stan.json
init=$ROOT/data/opt/$benchmark/init.json

output=$ROOT/tmp/prior-check/`date +%Y%m%d-%H%M%S`/$benchmark
mkdir --parents $output
echo "[ `date` ] $output"

# Matches production (bin/stan-sample.sh's defaults), now that the
# init fix below should make this budget actually informative.
num_samples=1000
num_warmup=1000
num_chains=4

#
# model is the working tree's current model.stan (alpha ~ normal(1, 1)).
# model-tight only differs in the alpha prior line.
#
cat > $output/model-tight.stan <<'EOF'
/**
 * Two-Parameter Logistic Item Response Model
 *
 * https://mc-stan.org/users/documentation/case-studies/tutorial_twopl.html
 **/

data {
  int<lower=1> I; // questions
  int<lower=1> J; // persons
  int<lower=1> N; // observations
  array[N] int<lower=1, upper=I> q_i; // question for n
  array[N] int<lower=1, upper=J> p_j; // person for n
  array[N] int<lower=0, upper=1> y;   // correctness for n
}

parameters {
  vector[I] alpha;          // discrimination for item i
  vector[I] beta;           // difficulty for item i
  vector[J - 2] theta_free; // ability for persons 3..J
}

transformed parameters {
  // Persons 1-2 anchor theta's location/scale to resolve #45's
  // non-identifiability - cheaper and far better-conditioned than
  // standardizing the whole vector every leapfrog step.
  vector[J] theta = append_row([0, 1]', theta_free);
}

model {
  vector[N] eta;

  alpha ~ normal(1, 0.3);
  beta  ~ normal(0, 3);
  theta_free ~ normal(0, 1);
  eta = alpha[q_i] .* (theta[p_j] - beta[q_i]);
  y ~ bernoulli_logit(eta);
}
EOF
cp $output/model-tight.stan $ROOT/src/model/model-tight.stan
(cd $CMDSTAN && make --jobs=`nproc` $ROOT/src/model/model-tight) || exit 1
(cd $CMDSTAN && make --jobs=`nproc` $ROOT/src/model/model) || exit 1
rm $ROOT/src/model/model-tight.stan

for variant in current tight; do
    if [ $variant = current ]; then
        model=$ROOT/src/model/model
    else
        model=$ROOT/src/model/model-tight
    fi

    out=$output/$variant
    mkdir --parents $out
    echo "[ `date` ] running $variant"

    $model \
        sample \
        num_samples=$num_samples \
        num_warmup=$num_warmup \
        num_chains=$num_chains \
        data \
        file=$data \
        init=$init \
        output \
        file=$out/chain.csv \
        num_threads=$num_chains \
        random seed=7 \
        && $CMDSTAN/bin/stansummary \
               --csv_filename=$out/summary.csv $out/*.csv > /dev/null
done

#
# Fraction of each parameter type with R_hat > 1.1, current vs tight.
#
for variant in current tight; do
    echo "--- $variant ---"
    awk -F',' '
        NR == 1 { next }
        $1 ~ /^"#/ { next }
        {
            gsub(/"/, "", $1)
            rhat = $NF + 0
            if ($1 ~ /^alpha\[/)  { a_n++; if (rhat > 1.1) a_bad++ }
            else if ($1 ~ /^beta\[/)  { b_n++; if (rhat > 1.1) b_bad++ }
            else if ($1 ~ /^theta\[/) { t_n++; if (rhat > 1.1) t_bad++ }
        }
        END {
            printf "alpha: %d/%d (%.1f%%) Rhat > 1.1\n", a_bad, a_n, 100*a_bad/a_n
            printf "beta:  %d/%d (%.1f%%) Rhat > 1.1\n", b_bad, b_n, 100*b_bad/b_n
            printf "theta: %d/%d (%.1f%%) Rhat > 1.1\n", t_bad, t_n, 100*t_bad/t_n
        }
    ' $output/$variant/summary.csv
done
