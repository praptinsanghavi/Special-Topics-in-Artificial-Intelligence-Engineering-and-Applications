import pytest
import numpy as np

def test_env_reset(env):
    """Test environment reset state."""
    obs, info = env.reset()
    
    assert env.day == 0
    assert env.inv == 20.0
    assert env.cash == 5000.0
    assert len(env.pending_orders) == 0
    
    # Check Observation Shape
    assert obs.shape == (3,)
    assert obs[0] == 20.0  # Inv
    assert obs[1] == 5.0   # Cash/1000
    assert obs[2] == 0     # Pending

def test_action_hold(env):
    """Test Hold action (0)."""
    env.reset()
    initial_cash = env.cash
    initial_inv = env.inv
    
    # Mock demand to fixed value for deterministic test
    # We can't easily mock internal np.random without patching, 
    # so we'll check ranges or logic.
    # Actually, let's just run step and check consistency.
    
    obs, reward, term, trunc, info = env.step(0)
    
    assert env.day == 1
    # Cash should increase (sales) or stay same (no sales), but hold cost applies
    # Inv should decrease or stay same
    assert env.inv <= initial_inv
    assert env.cash >= initial_cash # Revenue likely > holding cost
    assert term is False

def test_action_manufacturer(env):
    """Test Manufacturer Order (1)."""
    env.reset()
    start_cash = env.cash
    
    obs, reward, term, trunc, info = env.step(1)
    
    # Cost = 15 * 70 = 1050
    expected_cost = 15 * 70
    
    # Cash should decrease by cost (ignoring sales revenue for a moment)
    # Actually, step() adds revenue too. 
    # But we can check if pending order was created.
    
    assert len(env.pending_orders) == 1
    assert env.pending_orders[0] == (3, 15) # 3 days, 15 qty

def test_action_distributor(env):
    """Test Distributor Order (2)."""
    env.reset()
    start_inv = env.inv
    
    obs, reward, term, trunc, info = env.step(2)
    
    # Distributor is instant (inv increases immediately in step logic)
    # But sales happen AFTER arrival in step() logic?
    # Logic: 
    # 1. Arrivals (Pending) -> Inv update
    # 2. Demand -> Sales
    # 3. Action -> New Orders (Distributor adds to Inv immediately? No, usually next step)
    # Let's check code: 
    # elif action == 2: inv = min(inv + qty...)
    # It adds immediately at end of step. So it's available for NEXT step.
    
    # So Inv should decrease (sales) from start_inv, THEN add 10?
    # Code: 
    # ... Sales ... inv -= sold ...
    # ... Action ... inv += 10
    
    # So final inv = start - sold + 10
    # sold is max start_inv
    
    # Min possible inv = start - start + 10 = 10
    # Max possible inv = start - 0 + 10 = 30
    
    assert env.inv >= 10.0
    
def test_bankruptcy(env):
    """Test game over on negative cash."""
    env.reset()
    env.cash = -1000.0 # Deep debt
    env.inv = 0.0 # No stock to sell, so no revenue to save us
    
    # Trigger check in step
    obs, reward, term, trunc, info = env.step(0)
    
    assert term is True
