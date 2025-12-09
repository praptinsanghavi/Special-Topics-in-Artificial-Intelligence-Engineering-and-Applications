import numpy as np

class AssortmentBandit:
    """
    Contextual Bandit using Thompson Sampling.
    Responsible for selecting the 'Assortment' (Which new item to put on the shelf).
    """
    def __init__(self, num_arms):
        self.num_arms = num_arms # Number of candidate products
        # Alpha and Beta for Beta Distribution (Conjugate prior for Bernoulli rewards)
        # Represents (Successes + 1, Failures + 1)
        self.alphas = np.ones(num_arms)
        self.betas = np.ones(num_arms)

    def select_arm(self):
        """
        Selects an item to place on the shelf using **Thompson Sampling**.
        
        Why Thompson Sampling?
        Unlike "Epsilon-Greedy" (which chooses randomly x% of the time), Thompson Sampling
        chooses based on *probability*. If Item A has 5 successes and 0 failures, its Beta
        distribution is sharp around 0.83. We sample from that distribution.
        
        This handles the "Exploration-Exploitation Dilemma" naturally:
        - High Uncertainty (few trials) -> Wide distribution -> Higher chance of being sampled randomly.
        - High Certainty (many trials) -> Narrow distribution -> Only sampled if actually good.
        
        Returns:
            int: The index of the selected item (arm).
        """
        # Sample one value from the Beta distribution for each arm
        samples = [np.random.beta(self.alphas[i], self.betas[i]) for i in range(self.num_arms)]
        # Pick the arm that produced the highest sample
        return np.argmax(samples)

    def update(self, arm, reward):
        """
        Update the posterior distribution (Belief State) based on observed reward.
        
        Bayesian Update Rule:
        - Prior: Beta(alpha, beta)
        - Likelihood: Bernoulli(reward)
        - Posterior: Beta(alpha + reward, beta + (1-reward))
        
        Args:
           arm (int): The item we tried.
           reward (int): 1 if sold, 0 if not sold.
        """
        # Edge Case: Ensure reward is binary or bounded for standard Beta updates
        reward = max(0, min(1, int(reward))) 
        
        if reward > 0:
            self.alphas[arm] += reward
        else:
            self.betas[arm] += 1  # Penalty for taking up shelf space without sales

    def get_preferred_items(self):
        """
        Return arms sorted by expected value (Mean of Beta dist: alpha / (alpha + beta))
        """
        expected_values = self.alphas / (self.alphas + self.betas)
        return np.argsort(expected_values)[::-1]
