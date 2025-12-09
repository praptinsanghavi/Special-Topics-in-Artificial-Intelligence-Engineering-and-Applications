import pytest
import sys
import os

# Ensure 'src' is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.simulation.kirana_env import KiranaClusterEnv
from src.agents.procurement_agent import ProcurementAgent

@pytest.fixture
def env():
    """Returns a fresh KiranaClusterEnv instance."""
    return KiranaClusterEnv()

@pytest.fixture
def agent():
    """Returns a fresh ProcurementAgent instance."""
    return ProcurementAgent(state_dim=3, action_dim=3)
