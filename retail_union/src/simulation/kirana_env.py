import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any, List, Optional

class KiranaClusterEnv(gym.Env):
    """
    Simulates a single Retail Kirana Store environment for ERP optimization.

    This environment models the trade-offs between inventory holding costs,
    stockout penalties, and procurement lead times. The agent must decide
    daily whether to hold, order from a slow/cheap manufacturer, or a
    fast/expensive distributor.

    Attributes:
        day (int): Current simulation day (steps).
        inv (float): Current inventory level.
        cash (float): Current cash reserves.
        pending_orders (List[Tuple[int, int]]): List of (days_remaining, quantity) for incoming stock.

    Action Space:
        0: Hold (Do nothing).
        1: Manufacturer Order (Cost=70, LeadTime=3 days).
        2: Distributor Order (Cost=100, LeadTime=0/Instant).

    Observation Space:
        Box(3,) containing:
        - Inventory Level (0-1000)
        - Cash (Normalized by 1000)
        - Pending Stock (Total units)
    """
    metadata = {'render_modes': ['human']}

    def __init__(self) -> None:
        super().__init__()
        self.max_inventory: int = 100
        self.start_cash: float = 5000.0
        
        # Market Parameters
        self.COST_MFG: float = 70.0      # Cheap, Slow (3 Days)
        self.COST_DIST: float = 100.0    # Expensive, Fast (Instant)
        self.PRICE: float = 120.0
        self.HOLDING_COST: float = 1.0
        self.STOCKOUT_PENALTY: float = 50.0
        
        self.action_space: spaces.Discrete = spaces.Discrete(3)
        # State: [Inventory, Cash/1000, Pending_Orders]
        self.observation_space: spaces.Box = spaces.Box(low=0, high=1000, shape=(3,), dtype=np.float32)

        # State Variables
        self.day: int = 0
        self.inv: float = 20.0
        self.cash: float = self.start_cash
        self.pending_orders: List[Tuple[int, int]] = []

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resets the environment to the initial state.

        Args:
            seed (int, optional): Random seed.
            options (dict, optional): Configuration options.

        Returns:
            Tuple[np.ndarray, dict]: Initial observation and info dictionary.
        """
        super().reset(seed=seed)
        self.day = 0
        self.inv = 20.0
        self.cash = self.start_cash
        self.pending_orders = [] # List of tuples: (days_remaining, qty)
        
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        """
        Constructs the observation vector.

        Returns:
            np.ndarray: [Inventory, Cash/1000, Total Pending Stock]
        """
        # Sum of all pending stock
        pending_stock = sum([x[1] for x in self.pending_orders])
        return np.array([self.inv, self.cash/1000.0, pending_stock], dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Executes one time step within the environment.

        Args:
            action (int): The action to take (0=Hold, 1=Mfg, 2=Dist).

        Returns:
            Tuple:
                - observation (np.ndarray): New state.
                - reward (float): Reward gained/lost.
                - terminated (bool): Whether episode ended (Bankruptcy/Year end).
                - truncated (bool): Whether episode was truncated.
                - info (dict): Detailed metrics.
        """
        self.day += 1
        reward: float = 0.0
        
        # 1. Process Arrivals
        # Decrement days for all pending orders
        new_pending: List[Tuple[int, int]] = []
        arrived_qty: int = 0
        for days, qty in self.pending_orders:
            if days <= 1: # Arrives today
                arrived_qty += qty
            else:
                new_pending.append((days-1, qty))
        self.pending_orders = new_pending
        
        # Update Inventory with Arrivals
        self.inv = min(self.inv + arrived_qty, self.max_inventory)
        
        # 2. Demand & Sales
        demand: int = np.random.poisson(lam=5)
        sold: float = min(self.inv, demand)
        missed: float = demand - sold
        
        revenue: float = sold * self.PRICE
        self.cash += revenue
        self.inv -= sold
        
        # Rewards
        reward += revenue
        reward -= (self.inv * self.HOLDING_COST)
        reward -= (missed * self.STOCKOUT_PENALTY)
        
        # 3. Action Execution
        cost: float = 0.0
        
        if action == 1: # Manufacturer (3 Days)
            qty = 15
            cost = qty * self.COST_MFG
            if self.cash >= cost:
                self.cash -= cost
                reward -= cost
                self.pending_orders.append((3, qty)) # 3 Days Lead Time
            else:
                reward -= 10 # Invalid
                
        elif action == 2: # Distributor (Instant)
            qty = 10
            cost = qty * self.COST_DIST
            if self.cash >= cost:
                self.cash -= cost
                reward -= cost
                self.inv = min(self.inv + qty, self.max_inventory)
            else:
                reward -= 10
        
        terminated: bool = bool(self.day >= 365 or self.cash < 0)
        truncated: bool = False
        
        info: Dict[str, Any] = {
            "cash": self.cash,
            "sold": sold,
            "revenue": revenue,
            "cost": cost,
            "pending": sum([x[1] for x in self.pending_orders]),
            "arrived": arrived_qty
        }
        
        return self._get_obs(), reward, terminated, truncated, info
