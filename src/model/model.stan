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
  vector<lower=0>[I] alpha; // discrimination for item i
  vector[I] beta;           // difficulty for item i
  vector[J] theta;          // ability for person j
}

model {
  vector[N] eta;
  
  alpha ~ lognormal(0.5, 1);
  beta  ~ normal(0, 10);
  theta ~ normal(0, 1);
  for (n in 1:N) {
    eta[n] = alpha[q_i[n]] * (theta[p_j[n]] - beta[q_i[n]]);
  }
  y ~ bernoulli_logit(eta);
}
