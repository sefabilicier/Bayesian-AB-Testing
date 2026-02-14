import numpy as np
from scipy import stats
import pandas as pd

class BayesianABTest:
    def __init__(self, alpha_prior=1, beta_prior=1):
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.results = {}
        
    def update_posterior(self, successes, trials, group_name):
        alpha_posterior = self.alpha_prior + successes
        beta_posterior = self.beta_prior + (trials - successes)
        
        posterior = {
            'alpha': alpha_posterior,
            'beta': beta_posterior,
            'successes': successes,
            'trials': trials,
            'conversion_rate': successes / trials if trials > 0 else 0,
            'posterior_mean': alpha_posterior / (alpha_posterior + beta_posterior)
        }
        
        self.results[group_name] = posterior
        return posterior
    
    def get_posterior_samples(self, group_name, n_samples=100000):
        posterior = self.results[group_name]
        samples = np.random.beta(
            posterior['alpha'], 
            posterior['beta'], 
            n_samples
        )
        return samples
    
    def probability_B_beats_A(self, n_samples=100000):
        if 'A' not in self.results or 'B' not in self.results:
            raise ValueError("Both groups A and B must be updated first")
        
        samples_A = self.get_posterior_samples('A', n_samples)
        samples_B = self.get_posterior_samples('B', n_samples)
        
        prob_B_beats_A = np.mean(samples_B > samples_A)
        return prob_B_beats_A
    
    def expected_loss(self, n_samples=100000):
        samples_A = self.get_posterior_samples('A', n_samples)
        samples_B = self.get_posterior_samples('B', n_samples)
        
        loss_choose_B = np.mean(np.maximum(0, samples_A - samples_B))
        
        loss_choose_A = np.mean(np.maximum(0, samples_B - samples_A))
        
        return {
            'expected_loss_choose_A': loss_choose_A,
            'expected_loss_choose_B': loss_choose_B,
            'optimal_choice': 'B' if loss_choose_B < loss_choose_A else 'A'
        }
    
    def uplift_distribution(self, n_samples=100000):
        samples_A = self.get_posterior_samples('A', n_samples)
        samples_B = self.get_posterior_samples('B', n_samples)
        
        absolute_uplift = samples_B - samples_A
        
        relative_uplift = (samples_B - samples_A) / samples_A * 100
        
        return {
            'absolute_uplift': absolute_uplift,
            'relative_uplift': relative_uplift,
            'mean_absolute_uplift': np.mean(absolute_uplift),
            'mean_relative_uplift': np.mean(relative_uplift),
            'credible_interval_absolute': np.percentile(absolute_uplift, [2.5, 97.5]),
            'credible_interval_relative': np.percentile(relative_uplift, [2.5, 97.5])
        }
    
    def calculate_risk(self, n_samples=100000):
        prob_B_beats_A = self.probability_B_beats_A(n_samples)
        uplift_stats = self.uplift_distribution(n_samples)
        loss = self.expected_loss(n_samples)
        
        return {
            'probability_B_beats_A': prob_B_beats_A,
            'probability_A_beats_B': 1 - prob_B_beats_A,
            'expected_uplift': uplift_stats['mean_relative_uplift'],
            'uplift_ci': uplift_stats['credible_interval_relative'],
            'expected_loss_choose_A': loss['expected_loss_choose_A'],
            'expected_loss_choose_B': loss['expected_loss_choose_B'],
            'recommended_choice': loss['optimal_choice']
        }


class FrequentistABTest:
    @staticmethod
    def chi_squared_test(successes_A, trials_A, successes_B, trials_B):
        from scipy.stats import chi2_contingency
        
        # Create contingency table
        table = np.array([
            [successes_A, trials_A - successes_A],
            [successes_B, trials_B - successes_B]
        ])
        
        chi2, p_value, dof, expected = chi2_contingency(table)
        return {
            'chi2_statistic': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'significant_at_0.05': p_value < 0.05
        }
    
    @staticmethod
    def proportion_test(successes_A, trials_A, successes_B, trials_B):
        from statsmodels.stats.proportion import proportions_ztest
        
        count = np.array([successes_A, successes_B])
        nobs = np.array([trials_A, trials_B])
        
        z_stat, p_value = proportions_ztest(count, nobs)
        
        # Calculate confidence interval for the difference
        p_A = successes_A / trials_A
        p_B = successes_B / trials_B
        pooled_p = (successes_A + successes_B) / (trials_A + trials_B)
        se = np.sqrt(pooled_p * (1 - pooled_p) * (1/trials_A + 1/trials_B))
        
        ci_diff = (p_B - p_A) + np.array([-1, 1]) * 1.96 * se
        
        return {
            'z_statistic': z_stat,
            'p_value': p_value,
            'difference': p_B - p_A,
            'ci_difference': ci_diff,
            'significant_at_0.05': p_value < 0.05
        }


class SequentialBayesianTest:
    def __init__(self, alpha_prior=1, beta_prior=1):
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.history = {'A': [], 'B': []}
        
    def add_observation(self, group, success, trial):
        if not self.history[group]:
            current_alpha = self.alpha_prior
            current_beta = self.beta_prior
        else:
            current_alpha, current_beta = self.history[group][-1]['posterior_params']
        
        # Update with new data
        new_alpha = current_alpha + (success if success else 0)
        new_beta = current_beta + (trial - success if trial else 0)
        
        observation = {
            'step': len(self.history[group]) + 1,
            'successes': success,
            'trials': trial,
            'cumulative_successes': (self.history[group][-1]['cumulative_successes'] + success 
                                    if self.history[group] else success),
            'cumulative_trials': (self.history[group][-1]['cumulative_trials'] + trial 
                                if self.history[group] else trial),
            'posterior_params': (new_alpha, new_beta),
            'posterior_mean': new_alpha / (new_alpha + new_beta)
        }
        
        self.history[group].append(observation)
        return observation
    
    def get_current_probability(self, n_samples=10000):
        if not self.history['A'] or not self.history['B']:
            return 0.5
        
        params_A = self.history['A'][-1]['posterior_params']
        params_B = self.history['B'][-1]['posterior_params']
        
        samples_A = np.random.beta(params_A[0], params_A[1], n_samples)
        samples_B = np.random.beta(params_B[0], params_B[1], n_samples)
        
        return np.mean(samples_B > samples_A)
    
    def get_history_df(self):
        records = []
        for group in ['A', 'B']:
            for obs in self.history[group]:
                records.append({
                    'group': group,
                    'step': obs['step'],
                    'cumulative_trials': obs['cumulative_trials'],
                    'cumulative_successes': obs['cumulative_successes'],
                    'posterior_mean': obs['posterior_mean']
                })
        return pd.DataFrame(records)