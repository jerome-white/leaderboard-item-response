#!/bin/bash

# Compares the baseline model (master, fixed alpha/beta priors)
# against fully hierarchical alpha/beta (#53) on the same real,
# full-scale data. #52/#56/#57/#58/#59 all failed identically to fix
# #46 by removing one specific confound - this is the last untested
# item from #46's original candidate list, letting the data pick
# alpha/beta's pooling strength instead of fixing it.
#
# No new data fields needed - both variants run against the same
# stan.json already on disk.
#
# The #53 hierarchical variant exists as a real commit (f975e25,
# "Make alpha/beta fully hierarchical (#53)") but only on the
# hierarchical-alpha-beta branch, which never merged here - this
# branch's model.stan is still the fixed-prior version (byte-
# identical to master). It's built here via heredoc, copied straight
# from f975e25's content, the same way prior-check.sh builds
# model-tight: compiled, then deleted, leaving only the binary.
#
# f975e25 already ran this against mmlu/computer-science
# (tmp/hierarchical-check/20261009-120702, tmp/log/
# hierarchical-check.log): a near-total null, matching #52/#56-59's
# pattern - alpha 409/410 bad both variants, beta 408/410 both,
# theta 4414/4480 -> 4406/4480. sigma_beta's fitted posterior mean
# landed at 41.3 (vs. the fixed prior's sd=3) - the data pushed
# pooling strength to effectively nothing, not a converged useful
# pool - and sigma_alpha/sigma_beta themselves had bad Rhat (1.91,
# 1.78). That's consistent with "wrong subject for this fix":
# computer-science's worst-k items are 100% shared-ridge-drift, 0%
# isolated near-zero-alpha, and hierarchical pooling targets the
# latter. Repointed at mmlu/history, whose worst-k is a mix (2
# isolated near-zero-alpha items, 6 shared-ridge, confirmed directly
# from chain-sync.csv/selected.csv - not the "100% isolated" claim
# an earlier pass through this repo got wrong) - it has real
# isolated items, so this is still the first test of this fix
# against a subject that actually has the mechanism it targets, even
# though most of history's worst-k is the untouched shared-ridge
# confound instead.
#
# IMPORTANT, found the hard way: this script never passed `init=`,
# so both variants were sampling with cmdstan's default
# independent-random-per-chain init - exactly the already-diagnosed,
# already-fixed root cause of #46's pervasive ~99% Rhat-bad signature
# (see #60/PR #62, [[project_issue46_investigation]]), not anything
# about alpha/beta's pooling. That bug is almost certainly why both
# the original f975e25 run and today's repeat of it came back as a
# near-total null identical to baseline. Fixed here to pass the same
# data-informed init.json bin/run.sh uses in production, and bumped
# num_samples/num_warmup to match production (1000/1000) now that
# proper init should make that budget tractable.

ROOT=`git rev-parse --show-toplevel`
CMDSTAN=$ROOT/cmdstan-2.40.0

#
#
#
benchmark=mmlu/history
data=$ROOT/data/opt/$benchmark/stan.json
init=$ROOT/data/opt/$benchmark/init.json

output=$ROOT/tmp/hierarchical-check/`date +%Y%m%d-%H%M%S`/$benchmark
mkdir --parents $output
echo "[ `date` ] $output"

# Matches production (bin/stan-sample.sh's defaults), now that the
# init fix below should make this budget actually informative.
num_samples=1000
num_warmup=1000
num_chains=4

#
# baseline is master's model.stan (fixed alpha/beta priors).
# hierarchical replaces alpha/beta's fixed priors with hyperpriors,
# letting sigma_alpha/sigma_beta - the pooling strength - be
# estimated from the data instead of fixed.
#
git show master:src/model/model.stan > $output/model-baseline.stan
cp $output/model-baseline.stan $ROOT/src/model/model-baseline.stan
(cd $CMDSTAN && make --jobs=`nproc` $ROOT/src/model/model-baseline) || exit 1

cat > $output/model-hierarchical.stan <<'EOF'
/**
 * Two-Parameter Logistic Item Response Model - hierarchical alpha/beta (#53)
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
  real mu_alpha;              // population mean discrimination
  real<lower=0> sigma_alpha;  // population sd discrimination (pooling strength)
  real mu_beta;                // population mean difficulty
  real<lower=0> sigma_beta;    // population sd difficulty (pooling strength)
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

  mu_alpha ~ normal(1, 1);
  sigma_alpha ~ normal(1, 0.5);
  mu_beta ~ normal(0, 3);
  sigma_beta ~ normal(3, 1.5);

  alpha ~ normal(mu_alpha, sigma_alpha);
  beta  ~ normal(mu_beta, sigma_beta);
  theta_free ~ normal(0, 1);
  eta = alpha[q_i] .* (theta[p_j] - beta[q_i]);
  y ~ bernoulli_logit(eta);
}
EOF
cp $output/model-hierarchical.stan $ROOT/src/model/model-hierarchical.stan
(cd $CMDSTAN && make --jobs=`nproc` $ROOT/src/model/model-hierarchical) || exit 1
rm $ROOT/src/model/model-baseline.stan $ROOT/src/model/model-hierarchical.stan

for variant in baseline hierarchical; do
    if [ $variant = baseline ]; then
        model=$ROOT/src/model/model-baseline
    else
        model=$ROOT/src/model/model-hierarchical
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
               --csv_filename=$out/summary.csv $out/chain_*.csv > /dev/null
done

#
# Fraction of each parameter type with R_hat > 1.1, baseline vs
# hierarchical. Also prints the fitted hyperparameters for the
# hierarchical variant.
#
for variant in baseline hierarchical; do
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
    grep -E '^"(mu_alpha|sigma_alpha|mu_beta|sigma_beta)"' $output/$variant/summary.csv
done
