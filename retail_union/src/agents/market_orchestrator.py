class MarketOrchestrator:
    """
    The Orchestrator agent (System Logic) that attempts to form Group Orders.
    It reads the 'signals' from the environment state and executes buys.
    """
    def __init__(self, bulk_threshold=50):
        self.bulk_threshold = bulk_threshold

    def process_market_clearing(self, agent_signals):
        """
        Input: Dict or Array of {agent_id: quantity_signaled}
        Output: Boolean (Success/Fail), Final Price Tier
        """
        # Sum up total volume
        total_volume = sum(agent_signals)
        
        # Determine success
        success = total_volume >= self.bulk_threshold
        
        # Determine implied price (for reporting/reward calculation)
        # Note: The actual cost deduction happens in the Env, but Orchestrator 
        # signals the 'event' success.
        return success, total_volume
