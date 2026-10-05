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

  alpha ~ normal(1, 1);
  beta  ~ normal(0, 3);
  theta_free ~ normal(0, 1);
  eta = alpha[q_i] .* (theta[p_j] - beta[q_i]);
  y ~ bernoulli_logit(eta);
}
