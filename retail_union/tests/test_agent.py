import pytest
import numpy as np
import torch

def test_agent_initialization(agent):
    assert agent.state_dim == 3
    assert agent.action_dim == 3
    assert agent.epsilon == 1.0

def test_agent_action_selection(agent):
    state = np.array([20.0, 5.0, 0.0], dtype=np.float32)
    
    # Epsilon is 1.0, so random action
    action = agent.select_action(state)
    assert action in [0, 1, 2]
    
    # Test Greedy (Epsilon = 0)
    agent.epsilon = 0.0
    action_greedy = agent.select_action(state)
    assert action_greedy in [0, 1, 2]

def test_memory_storage(agent):
    state = np.zeros(3)
    next_state = np.zeros(3)
    agent.store_transition(state, 0, 1.0, next_state, False)
    
    assert len(agent.memory) == 1

def test_train_step(agent):
    # Fill memory
    for _ in range(agent.batch_size + 1):
        s = np.zeros(3)
        ns = np.zeros(3)
        agent.store_transition(s, 0, 1.0, ns, False)
        
    initial_epsilon = agent.epsilon
    agent.train_step()
    
    # Epsilon should decay
    assert agent.epsilon < initial_epsilon
