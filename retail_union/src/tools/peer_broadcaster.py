class PeerBroadcaster:
    """
    A communication tool that allows agents to signal their intent to neighbors.
    This fulfills the 'Agent Orchestration' and 'Custom Tool' requirements.
    """
    def __init__(self, agent_id, env_state_reference):
        self.agent_id = agent_id
        # In a real distributed system, this would use a message queue (Redis/RabbitMQ).
        # In this simulation, we access a shared state list/dict managed by the Orchestrator.
        self.shared_state = env_state_reference

    def broadcast_buy_intent(self, quantity):
        """
        Signal intent to buy a specific quantity to the group.
        """
        self.shared_state['signals'][self.agent_id] = quantity
        return True

    def check_group_volume(self):
        """
        Check the total volume currently signaled by the group.
        Agencies use this to decide whether to 'Join' or 'Wait'.
        """
        return sum(self.shared_state['signals'].values())
    
    def clear_signal(self):
        self.shared_state['signals'][self.agent_id] = 0
