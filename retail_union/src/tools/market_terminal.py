class MarketTerminal:
    """
    A tool that agents can use to query current market prices and tiered discounts.
    Using this tool allows the agent to make informed decisions about joining group orders.
    """
    def __init__(self):
        self.spot_price = 100.0
        self.tiers = {
            10: 95.0,
            50: 80.0,  # The target bulk tier
            100: 70.0
        }
    
    def get_spot_price(self):
        return self.spot_price
    
    def get_bulk_price(self, quantity):
        best_price = self.spot_price
        for tier_qty, price in self.tiers.items():
            if quantity >= tier_qty:
                best_price = min(best_price, price)
        return best_price

    def get_savings_potential(self):
        return self.spot_price - self.tiers[50]
