
// LBA probability density function for single accumulator
real lba_pdf(real t, real b, real A, real v, real s) {
  real b_A_tv_ts = (b - A - t * v) / (t * s);
  real b_tv_ts = (b - t * v) / (t * s);

  real term_1 = v * Phi_approx(b_A_tv_ts);
  real term_2 = s * exp(std_normal_lpdf(fabs(b_A_tv_ts)));
  real term_3 = v * Phi_approx(b_tv_ts);
  real term_4 = s * exp(std_normal_lpdf(fabs(b_tv_ts)));

  return (1.0 / A) * (-term_1 + term_2 + term_3 - term_4);
}

// LBA cumulative distribution function for single accumulator
real lba_cdf(real t, real b, real A, real v, real s) {
  real b_A_tv = b - A - t * v;
  real b_tv = b - t * v;
  real ts = t * s;

  real term_1 = (b_A_tv / A) * Phi_approx(b_A_tv / ts);
  real term_2 = (b_tv / A) * Phi_approx(b_tv / ts);
  real term_3 = (ts / A) * exp(std_normal_lpdf(fabs(b_A_tv / ts)));
  real term_4 = (ts / A) * exp(std_normal_lpdf(fabs(b_tv / ts)));

  return 1 + term_1 - term_2 + term_3 - term_4;
}

// Full trial-level log-likelihood
real lba_lpdf(real rt, int response, int condition,
              real theta, real lambda_iat,
              real d0, real A, real b, real t0, real s) {
  real drift_effect = lambda_iat * theta * condition;
  real v_correct = fmax(d0 + drift_effect, 0.001);
  real v_error = fmax(d0 - drift_effect, 0.001);

  real t = rt - t0;
  if (t <= 0) return negative_infinity();

  real v_winner = response == 1 ? v_correct : v_error;
  real v_loser = response == 1 ? v_error : v_correct;

  real pdf_winner = lba_pdf(t, b, A, v_winner, s);
  real cdf_loser = lba_cdf(t | b, A, v_loser, s);
  real surv_loser = 1 - cdf_loser;

  // Defective density correction
  real prob_neg = Phi_approx(-v_correct / s) * Phi_approx(-v_error / s);
  real prob = (pdf_winner * surv_loser) / (1 - prob_neg);

  return log(fmax(prob, 1e-10));
}

