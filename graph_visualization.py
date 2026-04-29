import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random 

random.seed(42)

sizes = [200, 500, 750]

for size in sizes:
    print(f"\nLoading data for {size} nodes...")
    dir = f'graph_data_{size}'
    nodes_df = pd.read_csv(f'{dir}/nodes.csv')
    edges_df = pd.read_csv(f'{dir}/edges.csv')


    # load the results from each algorithm
    louvain_df = pd.read_csv(f'{dir}/louvain_results.csv')
    gn_df = pd.read_csv(f'{dir}/gn_results.csv')
    hybrid_df = pd.read_csv(f'{dir}/hybrid_results.csv')


    # build the graph using networkx
    G = nx.from_pandas_edgelist(edges_df, 'source', 'target')
    G.add_nodes_from(nodes_df['node_id'])

    positions = nx.spring_layout(G, k=2.5, iterations=200, seed=42)

    max_followers = nodes_df['followers'].max()

    node_sizes = []
    for node_id in G.nodes():
        row = nodes_df[nodes_df['node_id'] == node_id].iloc[0]
        size = 60 + 250 * (row['followers'] / max_followers)
        if row['is_influencer']:
            size = size + 600
        node_sizes.append(size)


    # ID infleuncer nodes 
    influencer_labels = {}
    for node_id in G.nodes():
        row = nodes_df[nodes_df['node_id'] == node_id].iloc[0]
        if row['is_influencer']:
            influencer_labels[node_id] = str(node_id)


    # function to pick a color for each node based on which community it belongs to
    def get_colors_and_mapping(df, column):
        node_to_community = {}
        for i in range(len(df)):
            node_id = df['node_id'].iloc[i]
            community = df[column].iloc[i]
            node_to_community[node_id] = community

        unique_communities = sorted(set(node_to_community.values()))

        colormap = plt.get_cmap('tab20')
        community_to_color = {}
        for i in range(len(unique_communities)):
            community = unique_communities[i]
            community_to_color[community] = colormap(i % 20)

        colors = []
        for node_id in G.nodes():
            community = node_to_community[node_id]
            colors.append(community_to_color[community])

        return colors, community_to_color, unique_communities


    # function that draws and saves a single full-resolution graph
    def draw_graph(title, colors, community_to_color, unique_communities, filename):
        fig, ax = plt.subplots(figsize=(24, 16))
        fig.patch.set_facecolor('#1a1a1a')
        ax.set_facecolor('#1a1a1a')

        # draw edges first so they sit behind the nodes
        nx.draw_networkx_edges(G, positions, ax=ax, alpha=0.08, width=0.4, edge_color='white')

        # draw nodes 
        nx.draw_networkx_nodes(G, positions, ax=ax, node_color=colors, node_size=node_sizes, alpha=0.95, edgecolors='#1a1a1a', linewidths=0.5)

        # id inflencers on graph 
        nx.draw_networkx_labels(G, positions, labels=influencer_labels, ax=ax,font_size=9, font_color='white', font_weight='bold')

        # build the legend (one entry per community)
        legend_patches = []
        for community in unique_communities:
            patch = mpatches.Patch(color=community_to_color[community], label=f"Community {community}")
            legend_patches.append(patch)

        ax.legend(handles=legend_patches, loc='lower left', fontsize=10,facecolor='#2a2a2a', labelcolor='white', framealpha=0.9,title="Communities", title_fontsize=11)

        ax.set_title(f"{title}\nNode size = followers  |  Labeled nodes = influencers", fontsize=16, color='white', pad=20)
        ax.axis('off')
        
        # add a bit of margin around the graph
        ax.margins(0.05)

        plt.tight_layout()
        plt.savefig(filename, dpi=200, bbox_inches='tight', facecolor='#1a1a1a')
        plt.close()
        print(f"Saved {filename}")


    # build colors for each of the 4 graphs
    original_colors, original_map, original_unique = get_colors_and_mapping(nodes_df, 'primary_group')
    louvain_colors, louvain_map, louvain_unique = get_colors_and_mapping(louvain_df, 'detected_group')
    gn_colors, gn_map, gn_unique = get_colors_and_mapping(gn_df, 'detected_group')
    hybrid_colors, hybrid_map, hybrid_unique = get_colors_and_mapping(hybrid_df, 'detected_group')


    # save 4 separate full-resolution images
    draw_graph("Original Graph", original_colors, original_map, original_unique, f'{dir}/graph_original.png')

    draw_graph("After Louvain", louvain_colors, louvain_map, louvain_unique, f'{dir}/graph_louvain.png')

    draw_graph("After Pure Girvan-Newman", gn_colors, gn_map, gn_unique, f'{dir}/graph_girvan_newman.png')

    draw_graph("After Hybrid GN-Louvain", hybrid_colors, hybrid_map, hybrid_unique, f'{dir}/graph_hybrid.png')

print("\nAll 4 graphs saved.")