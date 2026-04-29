import pandas as pd
import matplotlib.pyplot as plt




for size in [200, 500, 750]:
    dir = f'graph_data_{size}'

    print(f"\nLoading benchmark results for {size} nodes...")
    benchmark_df = pd.read_csv(f'{dir}/benchmark_results.csv')

    # colors for each algorithm - keep them consistent across all charts
    algo_colors = ['#4A90D9', '#F5A623', '#7ED321']
    # Louvain = blue, Hybrid = orange, Girvan-Newman = green


    # CHART 1: Runtime (seconds)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(benchmark_df['algorithm'], benchmark_df['runtime'], color=algo_colors)
    ax.set_ylabel('Runtime (seconds)', fontsize=12)
    ax.set_title('Runtime Comparison', fontsize=14, pad=15)
    ax.set_yscale('log')
    # add value labels on top of each bar
    for i, value in enumerate(benchmark_df['runtime']):
        ax.text(i, value, f'{value:.2f}s', ha='center', va='bottom', fontsize=11)
    plt.tight_layout()
    plt.savefig(f'{dir}/chart_runtime.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved chart_runtime.png")


    # CHART 2: Modularity
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(benchmark_df['algorithm'], benchmark_df['modularity'], color=algo_colors)
    ax.set_ylabel('Modularity', fontsize=12)
    ax.set_title('Modularity Comparison', fontsize=14, pad=15)
    ax.set_ylim(0, 1)
    for i, value in enumerate(benchmark_df['modularity']):
        ax.text(i, value, f'{value:.3f}', ha='center', va='bottom', fontsize=11)
    plt.tight_layout()
    plt.savefig(f'{dir}/chart_modularity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved chart_modularity.png")


    # CHART 3: ARI (accuracy)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(benchmark_df['algorithm'], benchmark_df['ari'], color=algo_colors)
    ax.set_ylabel('ARI Score', fontsize=12)
    ax.set_title('Accuracy (Adjusted Rand Index)', fontsize=14, pad=15)
    ax.set_ylim(0, 1)
    for i, value in enumerate(benchmark_df['ari']):
        ax.text(i, value, f'{value:.3f}', ha='center', va='bottom', fontsize=11)
    plt.tight_layout()
    plt.savefig(f'{dir}/chart_ari.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved chart_ari.png")


    # CHART 4: Communities Found
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(benchmark_df['algorithm'], benchmark_df['communities'], color=algo_colors)
    ax.set_ylabel('Number of Communities', fontsize=12)
    ax.set_title('Granularity (Communities Found)', fontsize=14, pad=15)
    for i, value in enumerate(benchmark_df['communities']):
        ax.text(i, value, str(value), ha='center', va='bottom', fontsize=11)
    plt.tight_layout()
    plt.savefig(f'{dir}/chart_communities.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved chart_communities.png")


    # CHART 5: Community Size Distribution
    # load per-algorithm result CSVs to get the actual community sizes
    louvain_df = pd.read_csv(f'{dir}/louvain_results.csv')
    hybrid_df = pd.read_csv(f'{dir}/hybrid_results.csv')
    gn_df = pd.read_csv(f'{dir}/gn_results.csv')

    # count members in each community for each algorithm
    louvain_sizes = louvain_df.groupby('detected_group').size().sort_values(ascending=False).values
    hybrid_sizes = hybrid_df.groupby('detected_group').size().sort_values(ascending=False).values
    gn_sizes = gn_df.groupby('detected_group').size().sort_values(ascending=False).values

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].bar(range(len(louvain_sizes)), louvain_sizes, color=algo_colors[0])
    axes[0].set_title(f'Louvain ({len(louvain_sizes)} communities)', fontsize=13)
    axes[0].set_xlabel('Community (sorted by size)')
    axes[0].set_ylabel('Number of nodes')

    axes[1].bar(range(len(hybrid_sizes)), hybrid_sizes, color=algo_colors[1])
    axes[1].set_title(f'Hybrid ({len(hybrid_sizes)} communities)', fontsize=13)
    axes[1].set_xlabel('Community (sorted by size)')
    axes[1].set_ylabel('Number of nodes')

    axes[2].bar(range(len(gn_sizes)), gn_sizes, color=algo_colors[2])
    axes[2].set_title(f'Girvan-Newman ({len(gn_sizes)} communities)', fontsize=13)
    axes[2].set_xlabel('Community (sorted by size)')
    axes[2].set_ylabel('Number of nodes')

    plt.tight_layout()
    plt.savefig(f'{dir}/chart_size_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved chart_size_distribution.png")


    # CHART 6: All metrics in one combined dashboard image
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].bar(benchmark_df['algorithm'], benchmark_df['runtime'], color=algo_colors)
    axes[0, 0].set_title('Runtime (log scale)', fontsize=13)
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_ylabel('Seconds')

    axes[0, 1].bar(benchmark_df['algorithm'], benchmark_df['modularity'], color=algo_colors)
    axes[0, 1].set_title('Modularity', fontsize=13)
    axes[0, 1].set_ylim(0, 1)

    axes[1, 0].bar(benchmark_df['algorithm'], benchmark_df['ari'], color=algo_colors)
    axes[1, 0].set_title('ARI Accuracy', fontsize=13)
    axes[1, 0].set_ylim(0, 1)

    axes[1, 1].bar(benchmark_df['algorithm'], benchmark_df['communities'], color=algo_colors)
    axes[1, 1].set_title('Communities Found', fontsize=13)

    plt.suptitle('Algorithm Benchmark Comparison', fontsize=16, y=1.00)
    plt.tight_layout()
    plt.savefig(f'{dir}/chart_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved chart_dashboard.png")

print("\nAll charts saved.")