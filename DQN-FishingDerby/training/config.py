# Experiment configurations from your code
experiments_config = [
        {
            'episodes': 5000,  # TA requirement: minimum 5000
            'learning_rate': 0.00025,
            'gamma': 0.8,  # Assignment baseline
            'epsilon_decay': 0.99,
            'batch_size': 32,
            'exploration_policy': 'epsilon_greedy',
            'experiment_name': 'Baseline_gamma08'
        },
        {
            'episodes': 5000,
            'learning_rate': 0.00025,
            'gamma': 0.99,  # Higher gamma test
            'epsilon_decay': 0.99,
            'batch_size': 32,
            'exploration_policy': 'epsilon_greedy',
            'experiment_name': 'HighGamma_099'
        },
        {
            'episodes': 5000,
            'learning_rate': 0.001,  # Higher learning rate
            'gamma': 0.8,
            'epsilon_decay': 0.99,
            'batch_size': 32,
            'exploration_policy': 'epsilon_greedy',
            'experiment_name': 'HighLR_001'
        },
        {
            'episodes': 5000,
            'learning_rate': 0.00025,
            'gamma': 0.8,
            'epsilon_decay': 0.99,
            'batch_size': 32,
            'exploration_policy': 'boltzmann',
            'experiment_name': 'Boltzmann'
        },
        {
            'episodes': 5000,
            'learning_rate': 0.00025,
            'gamma': 0.8,
            'epsilon_decay': 0.99,
            'batch_size': 32,
            'exploration_policy': 'ucb',
            'experiment_name': 'UCB'
        },
        {
            'episodes': 5000,
            'learning_rate': 0.00025,
            'gamma': 0.8,
            'epsilon_decay': 0.995,  # Slower decay
            'batch_size': 32,
            'exploration_policy': 'epsilon_greedy',
            'experiment_name': 'SlowDecay_0995'
        }
    ]
