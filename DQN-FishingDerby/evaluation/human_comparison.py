# ============================================================================
# CELL 6: HUMAN VS BOT COMPARISON
# ============================================================================

def create_human_vs_bot_comparison(test_results_list):
    """Create comprehensive human vs bot comparison graph"""

    print("\n📊 Creating Human vs Bot Comparison")

    # Human performance benchmarks for FishingDerby
    benchmarks = {
        'Random Agent': {'mean': -5.0, 'std': 8.0, 'color': 'red'},
        'Novice Human': {'mean': 10.0, 'std': 5.0, 'color': 'orange'},
        'Average Human': {'mean': 25.0, 'std': 7.0, 'color': 'yellow'},
        'Expert Human': {'mean': 50.0, 'std': 10.0, 'color': 'green'}
    }

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Bar comparison
    ax = axes[0]
    labels = list(benchmarks.keys()) + [r['model_name'] for r in test_results_list]
    means = [benchmarks[k]['mean'] for k in benchmarks.keys()] + \
            [r['mean_reward'] for r in test_results_list]
    stds = [benchmarks[k]['std'] for k in benchmarks.keys()] + \
           [r['std_reward'] for r in test_results_list]
    colors = [benchmarks[k]['color'] for k in benchmarks.keys()] + \
             ['blue'] * len(test_results_list)

    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, color=colors, alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Average Score')
    ax.set_title('Human vs DQN Agent Performance')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3)

    # Add value labels
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.1f}', ha='center', va='bottom' if height >= 0 else 'top')

    # Plot 2: Performance levels
    ax = axes[1]

    # Create performance level chart
    levels = ['Random\n(<0)', 'Beginner\n(0-10)', 'Novice\n(10-20)',
              'Intermediate\n(20-35)', 'Advanced\n(35-50)', 'Expert\n(50+)']
    level_colors = ['red', 'orange', 'yellow', 'lightgreen', 'green', 'darkgreen']

    # Count agents in each level
    level_counts = [0] * 6

    for r in test_results_list:
        score = r['mean_reward']
        if score < 0:
            level_counts[0] += 1
        elif score < 10:
            level_counts[1] += 1
        elif score < 20:
            level_counts[2] += 1
        elif score < 35:
            level_counts[3] += 1
        elif score < 50:
            level_counts[4] += 1
        else:
            level_counts[5] += 1

    # Create pie chart
    non_zero_counts = [(c, l, col) for c, l, col in zip(level_counts, levels, level_colors) if c > 0]
    if non_zero_counts:
        counts, labels, colors = zip(*non_zero_counts)
        ax.pie(counts, labels=labels, colors=colors, autopct='%1.0f%%', startangle=90)
        ax.set_title('DQN Agent Performance Distribution')

    plt.suptitle('🎮 Human vs Bot Performance Analysis - FishingDerby 🎣',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('human_vs_bot_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Print analysis
    best_agent = max(test_results_list, key=lambda x: x['mean_reward'])
    print(f"\n🏆 Best Agent: {best_agent['model_name']}")
    print(f"   Score: {best_agent['mean_reward']:.2f} ± {best_agent['std_reward']:.2f}")
    print(f"   Win Rate: {best_agent['win_rate']:.1%}")

    if best_agent['mean_reward'] < 0:
        print("   Level: ⭐ Below Random - Needs more training")
    elif best_agent['mean_reward'] < 10:
        print("   Level: ⭐⭐ Beginner")
    elif best_agent['mean_reward'] < 25:
        print("   Level: ⭐⭐⭐ Novice")
    elif best_agent['mean_reward'] < 50:
        print("   Level: ⭐⭐⭐⭐ Intermediate")
    else:
        print("   Level: ⭐⭐⭐⭐⭐ Expert")
