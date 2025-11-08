# ============================================================================
# CELL 7: VIDEO RECORDING
# ============================================================================

def record_gameplay(model_path, model_name, num_episodes=3):
    """Record gameplay videos"""

    print(f"\n📹 Recording {num_episodes} videos for {model_name}")

    env = gym.make("ALE/FishingDerby-v5", render_mode='rgb_array')
    n_actions = env.action_space.n

    model = DQN(n_actions).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    for ep in range(num_episodes):
        frames = []
        obs, _ = env.reset()
        frame = preprocess_frame(obs)
        state_stack = np.stack([frame] * 4, axis=0)

        episode_reward = 0
        for _ in range(500):  # Max frames
            frames.append(env.render())

            with torch.no_grad():
                state_tensor = torch.FloatTensor(state_stack).unsqueeze(0).to(device)
                q_values = model(state_tensor)
                action = q_values.argmax().item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward

            if terminated or truncated:
                break

            next_frame = preprocess_frame(next_obs)
            state_stack = np.concatenate([state_stack[1:], [next_frame]], axis=0)

        # Save video
        video_name = f'{model_name}_episode_{ep+1}.mp4'
        imageio.mimsave(video_name, frames, fps=30)
        print(f"   Saved: {video_name} (Reward: {episode_reward:.1f})")

    env.close()
    del model
    gc.collect()
