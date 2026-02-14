import numpy as np
import pandas as pd
from scipy import stats

def generate_simulated_data(conversion_rate_A, conversion_rate_B, 
                           sample_size_A, sample_size_B, seed=42):
    np.random.seed(seed)
    
    successes_A = np.random.binomial(sample_size_A, conversion_rate_A)
    successes_B = np.random.binomial(sample_size_B, conversion_rate_B)
    
    return {
        'A': {'successes': successes_A, 'trials': sample_size_A},
        'B': {'successes': successes_B, 'trials': sample_size_B}
    }

def calculate_required_sample_size(mde, alpha=0.05, power=0.8, baseline_rate=0.1):
    from statsmodels.stats.power import NormalIndPower
    from statsmodels.stats.proportion import proportion_effectsize
    
    effect_size = proportion_effectsize(baseline_rate, baseline_rate * (1 + mde))
    
    power_analysis = NormalIndPower()
    sample_size = power_analysis.solve_power(
        effect_size=effect_size,
        alpha=alpha,
        power=power,
        alternative='two-sided'
    )
    
    return int(np.ceil(sample_size))

def format_results_for_display(risk_metrics):
    return {
        'Probability B > A': f"{risk_metrics['probability_B_beats_A']:.3f}",
        'Probability A > B': f"{risk_metrics['probability_A_beats_B']:.3f}",
        'Expected Uplift': f"{risk_metrics['expected_uplift']:.2f}%",
        'Uplift 95% CI': f"[{risk_metrics['uplift_ci'][0]:.2f}%, {risk_metrics['uplift_ci'][1]:.2f}%]",
        'Expected Loss (Choose A)': f"{risk_metrics['expected_loss_choose_A']:.4f}",
        'Expected Loss (Choose B)': f"{risk_metrics['expected_loss_choose_B']:.4f}",
        'Recommended Choice': risk_metrics['recommended_choice']
    }

def calculate_bayes_factor(bayesian_test, n_samples=100000):
    samples_A = bayesian_test.get_posterior_samples('A', n_samples)
    samples_B = bayesian_test.get_posterior_samples('B', n_samples)
    
    prob_diff = np.mean(np.abs(samples_B - samples_A) > 0.01)
    prob_same = 1 - prob_diff
    
    bayes_factor = prob_diff / prob_same if prob_same > 0 else np.inf
    
    interpretation = ""
    if bayes_factor > 100:
        interpretation = "Decisive evidence for H1"
    elif bayes_factor > 30:
        interpretation = "Very strong evidence for H1"
    elif bayes_factor > 10:
        interpretation = "Strong evidence for H1"
    elif bayes_factor > 3:
        interpretation = "Substantial evidence for H1"
    elif bayes_factor > 1:
        interpretation = "Anecdotal evidence for H1"
    else:
        interpretation = "Evidence supports H0"
    
    return {
        'bayes_factor': bayes_factor,
        'interpretation': interpretation
    }