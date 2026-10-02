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
  // discrimination for item i. Unconstrained: a negative value flags
  // an item where higher ability predicts *lower* correctness
  // (mis-keyed or corrupted), rather than just shrinking it toward
  // uninformative, per Rodriguez et al. 2021's IRT-disc.
  vector[I] alpha;
  vector[I] beta;           // difficulty for item i
  vector[J] theta;          // ability for person j
}

model {
  vector[N] eta;

  // Symmetric and centered on a positive value: most items are
  // expected to discriminate normally (mean 1, in the same range as
  // the old lognormal(0.5, 1)'s typical values), but with enough
  // mass below zero (~16%) that the likelihood can pull a genuinely
  // bad item's alpha negative instead of just toward 0. The positive
  // center also matters for identifiability: with alpha unconstrained,
  // flipping the sign of every alpha, beta, and theta at once leaves
  // eta unchanged, so a prior symmetric about 0 would leave that
  // global reflection equally probable. Centering on +1 breaks the
  // symmetry by making the all-negative mirror image far less likely
  // a priori, while still letting individual items go negative.
  alpha ~ normal(1, 1);
  beta  ~ normal(0, 3);
  theta ~ normal(0, 1);
  for (n in 1:N) {
    eta[n] = alpha[q_i[n]] * (theta[p_j[n]] - beta[q_i[n]]);
  }
  y ~ bernoulli_logit(eta);
}
