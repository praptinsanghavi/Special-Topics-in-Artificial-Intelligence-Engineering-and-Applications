def train_dqn_memory_efficient(
    episodes=5000,
    learning_rate=0.00025,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=0.995,
    batch_size=32,
    target_update_freq=50,
    exploration_policy="epsilon_greedy",
    experiment_name="Baseline",
    save_checkpoint_freq=500,
    memory_size=10000  # Reduced from 100k
):
    """
    Memory-efficient DQN training with checkpointing
    """

    print(f"\n{'='*60}")
    print(f"🧪 Experiment: {experiment_name}")
    print(f"   Episodes: {episodes} | LR: {learning_rate} | γ: {gamma}")
    print(f"   Memory: {memory_size} | Batch: {batch_size}")
    print(f"{'='*60}")

    # Check memory before starting
    check_memory()

    # Initialize environment
    env = gym.make("ALE/FishingDerby-v5")
    n_actions = env.action_space.n

    # Initialize networks
    q_network = DQN(n_actions).to(device)
    target_network = DQN(n_actions).to(device)
    target_network.load_state_dict(q_network.state_dict())

    # Optimizer
    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)

    # Replay buffer (REDUCED SIZE)
    memory = deque(maxlen=memory_size)

    # Exploration parameters
    epsilon = epsilon_start
    temperature = 1.0
    action_counts = np.ones(n_actions)
    total_selections = 0

    # Metrics (store less data)
    rewards_history = []
    avg_rewards = []  # Store only averages every 100 episodes
    best_reward = -float('inf')

    # Training loop with memory management
    progress = tqdm(range(episodes), desc=experiment_name)

    for episode in progress:
        obs, _ = env.reset()
        frame = preprocess_frame(obs)
        state_stack = np.stack([frame] * 4, axis=0)
        episode_reward = 0

        for step in range(99):  # Max 99 steps
            # Action selection
            if exploration_policy == "epsilon_greedy":
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(state_stack).unsqueeze(0).to(device)
                        q_values = q_network(state_tensor)
                        action = q_values.argmax().item()

            elif exploration_policy == "boltzmann":
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state_stack).unsqueeze(0).to(device)
                    q_values = q_network(state_tensor).squeeze()
                    temp = max(temperature * epsilon, 0.1)
                    probs = F.softmax(q_values / temp, dim=0).cpu().numpy()
                    action = np.random.choice(n_actions, p=probs)

            elif exploration_policy == "ucb":
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state_stack).unsqueeze(0).to(device)
                    q_values = q_network(state_tensor).squeeze().cpu().numpy()
                    if total_selections > 0:
                        c = 2.0
                        ucb_values = q_values + c * np.sqrt(
                            np.log(total_selections + 1) / (action_counts + 1)
                        )
                        action = ucb_values.argmax()
                    else:
                        action = env.action_space.sample()
                    action_counts[action] += 1
                    total_selections += 1

            # Environment step
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Update frame stack
            next_frame = preprocess_frame(next_obs)
            next_state_stack = np.concatenate([state_stack[1:], [next_frame]], axis=0)

            # Store experience
            memory.append(Experience(
                state_stack.copy(),
                action,
                np.clip(reward, -1, 1),
                next_state_stack.copy(),
                done
            ))

            # Training step
            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)

                states = torch.FloatTensor(np.array([e.state for e in batch])).to(device)
                actions = torch.LongTensor([e.action for e in batch]).to(device)
                rewards = torch.FloatTensor([e.reward for e in batch]).to(device)
                next_states = torch.FloatTensor(np.array([e.next_state for e in batch])).to(device)
                dones = torch.FloatTensor([e.done for e in batch]).to(device)

                current_q = q_network(states).gather(1, actions.unsqueeze(1))

                # Double DQN
                next_actions = q_network(next_states).argmax(1)
                next_q = target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
                target_q = rewards + gamma * next_q * (1 - dones)

                loss = F.smooth_l1_loss(current_q.squeeze(), target_q.detach())

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q_network.parameters(), 10)
                optimizer.step()

            state_stack = next_state_stack
            episode_reward += reward

            if done:
                break

        # Update target network
        if episode % target_update_freq == 0:
            target_network.load_state_dict(q_network.state_dict())

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Store metrics efficiently
        rewards_history.append(episode_reward)
        if episode_reward > best_reward:
            best_reward = episode_reward

        # Calculate and store averages every 100 episodes
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            avg_rewards.append(avg_reward)

            # Clear old reward history to save memory
            if len(rewards_history) > 1000:
                rewards_history = rewards_history[-500:]

        # Update progress bar
        current_avg = np.mean(rewards_history[-min(100, len(rewards_history)):])
        progress.set_postfix({
            'reward': f'{episode_reward:.1f}',
            'avg': f'{current_avg:.1f}',
            'ε': f'{epsilon:.3f}',
            'best': f'{best_reward:.1f}'
        })

        # Save checkpoint periodically
        if (episode + 1) % save_checkpoint_freq == 0:
            import os
            print(f"\nSaving checkpoint to: {os.getcwd()}") # Add this line
            checkpoint_path = f'checkpoint_{experiment_name}_{episode+1}.pt'
            torch.save({
                'episode': episode + 1,
                'model_state': q_network.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'epsilon': epsilon,
                'best_reward': best_reward,
                'avg_rewards': avg_rewards
            }, checkpoint_path)
            print(f"\n💾 Checkpoint saved: {checkpoint_path}")
            check_memory()

        # Memory cleanup every 500 episodes
        if (episode + 1) % 500 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    env.close()

    # Final save
    print(f"\nSaving final model to: {os.getcwd()}") # Add this line
    final_path = f'final_model_{experiment_name}.pt'
    torch.save(q_network.state_dict(), final_path)
    print(f"\n✅ Final model saved: {final_path}")

    # Create summary results
    results = {
        'experiment': experiment_name,
        'episodes': episodes,
        'learning_rate': learning_rate,
        'gamma': gamma,
        'exploration_policy': exploration_policy,
        'best_reward': best_reward,
        'final_avg_reward': current_avg,
        'avg_rewards_history': avg_rewards,
        'last_100_rewards': rewards_history[-100:]
    }

    # Save results to disk
    print(f"\nSaving results to: {os.getcwd()}") # Add this line
    with open(f'results_{experiment_name}.json', 'w') as f:
        json.dump(results, f)

    return q_network, results
