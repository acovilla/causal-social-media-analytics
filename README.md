# causal-social-media-analytics
Application of staggered DiD and count data models to unstructured text data in Python

1. Robust Difference-in-Differences (Sun & Abraham, 2021)

This function implements the interaction-weighted estimator proposed by Sun and Abraham (2021). In settings where treatment adoption is staggered, static TWFE models can conflate treatment effects with time trends. This method corrects for such potential bias by estimating cohort-specific average treatment effects on the treated. It subsequently aggregates these effects to provide a consistent estimate of the causal impact.

2. Two-Stage Difference-in-Differences (Gardner, 2021)

As an alternative robust estimator, the Gardner (2021) two-stage approach is included. This method separates the identification of the counterfactual from the estimation of the treatment effect.

Stage 1: Identifies unit and time shocks using only untreated observations to remove fixed effects.

Stage 2: Estimates the treatment effect on the resulting residuals. This approach is particularly useful for verifying the robustness of results obtained through other DiD specifications.

3. Count Data Models (Negative Binomial Regression)

Given the non-negative, integer nature of social media engagement metrics, the code implements generalized linear models with a Negative Binomial family. This specification explicitly models the overdispersion inherent in viral data distributions. Therefore, it provides more accurate standard errors and coefficient estimates than linear approximations or Poisson models, which assume the mean equals the variance.

4. Baseline Two-Way Fixed Effects (TWFE)

Standard TWFE implementations are provided to establish baseline correlations. While potentially biased in staggered settings, these models serve as a necessary benchmark for comparing the magnitude and direction of effects against the robust estimators described above.
