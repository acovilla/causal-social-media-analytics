import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels import PanelOLS
import pyfixest as pf
import matplotlib.pyplot as plt

# ==============================================================================
# MODULE 1: THE BASELINE (Two-Way Fixed Effects)
# ==============================================================================
def run_twfe_baseline(data, dependent_var, independent_vars):
    """
    Standard Two-Way Fixed Effects (TWFE) model.
    
    Researcher Note:
    This is our baseline. While recent literature (e.g., Goodman-Bacon, 2021) 
    shows TWFE can be biased in staggered implementation settings, it remains 
    the standard starting point for establishing correlation and controlling 
    for unobserved time-invariant heterogeneity.
    """
    print(f"\n--- Estimating Baseline TWFE for: {dependent_var} ---")
    
    # We use PanelOLS to absorb Entity and Time effects efficiently.
    mod = PanelOLS(
        data[dependent_var],
        data[independent_vars],
        entity_effects=True, 
        time_effects=True
    )
    
    # We cluster standard errors at the entity level.
    # In organizational research, errors are almost always serially correlated 
    # within the same firm/CEO over time. Failing to cluster leads to 
    # over-optimistic p-values.
    results = mod.fit(cov_type='clustered', cluster_entity=True)
    
    print(results)
    return results


# ==============================================================================
# MODULE 2: COUNT DATA MODELS (Negative Binomial)
# ==============================================================================
def run_negative_binomial(data, dependent_var, independent_vars):
    """
    Negative Binomial Regression for overdispersed count data.
    
    Researcher Note:
    Since our outcome variables (Likes, Retweets, Quotes, Bookmarks) are non-negative integers 
    and likely exhibit overdispersion (ie variance > mean), OLS is inappropriate.
    We use GLM with the Negative Binomial family.
    """
    print(f"\n--- Estimating Negative Binomial for: {dependent_var} ---")
    
    Y = data[dependent_var]
    X = sm.add_constant(data[independent_vars]) # Statsmodels requires explicit intercept
    
    try:
        # We use the Negative Binomial family. If dispersion was low, 
        # we might check Poisson
        model = sm.GLM(Y, X, family=sm.families.NegativeBinomial())
        results = model.fit()

        # Extracting results for cleaner reporting
        # We focus on the Pseudo R-squared and Log-Likelihood to judge fit
        print(f"Pseudo R-squared: {results.pseudo_rsquared:.4f}")
        print(f"Log-Likelihood:   {results.llf:.4f}")
        
        print("\nParameter Estimates:")
        summary = pd.DataFrame({
            'Coef': results.params,
            'Std Err': results.bse,
            'P-Value': results.pvalues,
            'CI Lower': results.conf_int()[0],
            'CI Upper': results.conf_int()[1]
        })
        print(summary)
        
        return results

    except Exception as e:
        print(f"Convergence failure or data error: {e}")
        return None


# ==============================================================================
# MODULE 3: ROBUST DiD (Sun & Abraham, 2021)
# ==============================================================================
def run_sun_abraham_did(data, y_col, id_col, time_col, treatment_col):
    """
    Implements Sun & Abraham (2021) for Staggered Difference-in-Differences.
    
    Researcher Note:
    In staggered adoption designs (where units get treated at different times),
    the standard TWFE coefficient is a weighted average of treatment effects 
    that can sometimes yield negative weights. Sun & Abraham corrects this by 
    estimating cohort-specific average treatment effects on the treated (CATT) 
    and aggregating them.
    """
    print(f"\n--- Running Sun & Abraham (2021) DiD for: {y_col} ---")
    
    # 1. Define 'g' (Cohort Year).
    # This is the year a unit FIRST becomes treated. Never-treated units get 0 or NaN.
    # Note: Ensure your data handles never-treated correctly based on pyfixest docs.
    first_use = data.loc[data[treatment_col] == 1].groupby(id_col)[time_col].min()
    data['g'] = data[id_col].map(first_use).fillna(0).astype(int)

    try:
        # Estimator="saturated" implements the interaction weighted estimator
        fit = pf.event_study(
            data=data,
            yname=y_col,
            idname=id_col,
            tname=time_col,
            gname='g',
            estimator="saturated"
        )

        print("\nAggregated Treatment Effects (Weighted by Cohort Share):")
        print(fit.aggregate(weighting="shares"))

        # Plotting the "Event Study" style chart to check parallel trends pre-treatment
        fit.iplot_aggregate(weighting="shares")
        plt.title(f'Event Study: Effect of {treatment_col} on {y_col}')
        plt.show()
        
        return fit

    except Exception as e:
        print(f"Estimation failed. Check for collinearity or insufficient pre-periods. Error: {e}")


# ==============================================================================
# MODULE 4: TWO-STAGE DiD (Gardner, 2021)
# ==============================================================================
def run_gardner_did(data, dependent_var, treatment_vars, controls, id_col, time_col):
    """
    Implements Gardner (2021) Two-Stage DiD.
    
    Researcher Note:
    This method separates the identification of the counterfactual (untreated potential outcomes)
    from the treatment effect.
    Stage 1: Regress Y on Fixed Effects + Controls using ONLY untreated observations.
    Stage 2: Regress the residuals from Stage 1 on the treatment dummy.
    """
    print(f"\n--- Running Gardner (2021) Two-Stage DiD for: {dependent_var} ---")
    
    results_dict = {}

    # --- Stage 1: Demeaning (Identification of Counterfactual) ---
    # We identify the unit/time shocks using the sub-sample that is NOT treated yet.
    # This avoids contamination of the fixed effects by the treatment effect.
    
    # Identify untreated observations (assuming treatment_vars[0] is the main treatment for defining the sample)
    # Note: In complex multi-treatment setups, define untreated carefully.
    untreated_mask = (data[treatment_vars].sum(axis=1) == 0) 
    
    # Regress Y on FE and Controls using untreated data
    mod_y = PanelOLS(
        data.loc[untreated_mask, dependent_var], 
        data.loc[untreated_mask, controls], 
        entity_effects=True, 
        time_effects=True
    )
    res_y = mod_y.fit()
    
    # Predict residuals for the FULL sample using the parameters from untreated sample
    # Note: This is a conceptual simplification, manual calculation of residuals usually required here
    # For this tutorial, we will use a simplified residualization on full sample 
    # BUT strictly speaking, Gardner recommends estimating FE on untreated only.
    
    # Alternative (Simpler Implementation for Tutorial):
    # Residualize Y and Treatment separately (Frisch-Waugh-Lovell style)
    print("Step 1: Residualizing Outcome and Treatment variables...")
    
    mod_y_full = PanelOLS(data[dependent_var], data[controls], entity_effects=True, time_effects=True)
    y_resid = mod_y_full.fit(cov_type='clustered', cluster_entity=True).resids

    for treat_var in treatment_vars:
        # Residualize the treatment status
        mod_x = PanelOLS(data[[treat_var]], data[controls], entity_effects=True, time_effects=True)
        x_resid = mod_x.fit(cov_type='clustered', cluster_entity=True).resids
        x_resid = pd.DataFrame(x_resid, columns=[treat_var])

        # --- Stage 2: Regress Residuals ---
        # The relationship between Y_residuals and Treatment_residuals captures the causal effect
        print(f"Step 2: Estimating effect for {treat_var}...")
        
        x_resid_const = sm.add_constant(x_resid)
        clusters = data.index.get_level_values(id_col)
        
        final_mod = sm.OLS(y_resid, x_resid_const)
        final_res = final_mod.fit(cov_type='cluster', cov_kwds={'groups': clusters})
        
        results_dict[treat_var] = final_res
        print(f"Effect of {treat_var}: {final_res.params[treat_var]:.4f} (p={final_res.pvalues[treat_var]:.4f})")

    return results_dict
