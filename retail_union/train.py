import numpy as np
import torch
import matplotlib.pyplot as plt
from src.simulation.kirana_env import KiranaClusterEnv
from src.agents.procurement_agent import ProcurementAgent

def run_erp_training(num_episodes=200):
    env = KiranaClusterEnv()
    
    # State: [Inv, Cash, Pending] (Dim=3)
    # Action: [Hold, Mfg, Dist] (Dim=3)
    agent = ProcurementAgent(state_dim=3, action_dim=3)
    
    rewards_history = []
    
    print("Starting Smart ERP Training...")
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, term, trunc, info = env.step(action)
            done = term or trunc
            
            agent.store_transition(state, action, reward, next_state, done)
            agent.train_step()
            episode_reward += reward
            state = next_state
            
        agent.update_target_network()
        rewards_history.append(episode_reward)
        
        if episode % 20 == 0:
            print(f"Episode {episode}: Profit {episode_reward:.1f}, Epsilon {agent.epsilon:.2f}")

    # Plot
    plt.figure(figsize=(10,5))
    plt.plot(rewards_history)
    plt.title("Smart ERP Training: Learning to Plan Ahead")
    plt.xlabel("Episode")
    plt.ylabel("Profit")
    plt.savefig('erp_results.png')
    print("Training Complete.")
    
    torch.save(agent.policy_net.state_dict(), "erp_brain.pth")

if __name__ == "__main__":
    run_erp_training()
