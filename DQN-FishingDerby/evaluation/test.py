# ============================================================================
# CELL 5: TESTING FUNCTION
# ============================================================================

def test_trained_model(model_path, model_name, test_episodes=100):
    """Test a trained model with 100 episodes"""

    print(f"\n🧪 Testing: {model_name}")
    print(f"   Episodes: {test_episodes}")

    # Load model
    env = gym.make("ALE/FishingDerby-v5")
    n_actions = env.action_space.n

    model = DQN(n_actions).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Test metrics
    test_rewards = []
    test_steps = []
    test_wins = []

    for episode in tqdm(range(test_episodes), desc=f"Testing {model_name}"):
        obs, _ = env.reset()
        frame = preprocess_frame(obs)
        state_stack = np.stack([frame] * 4, axis=0)

        episode_reward = 0
        episode_steps = 0

        done = False
        while not done and episode_steps < 99:
            # Greedy action selection (no exploration during testing)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state_stack).unsqueeze(0).to(device)
                q_values = model(state_tensor)
                action = q_values.argmax().item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_frame = preprocess_frame(next_obs)
            state_stack = np.concatenate([state_stack[1:], [next_frame]], axis=0)

            episode_reward += reward
            episode_steps += 1

        test_rewards.append(episode_reward)
        test_steps.append(episode_steps)
        test_wins.append(1 if episode_reward > 0 else 0)

    env.close()

    # Calculate statistics
    stats = {
        'model_name': model_name,
        'mean_reward': np.mean(test_rewards),
        'std_reward': np.std(test_rewards),
        'min_reward': np.min(test_rewards),
        'max_reward': np.max(test_rewards),
        'win_rate': np.mean(test_wins),
        'mean_steps': np.mean(test_steps)
    }

    # Clean up
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return stats, test_rewards
