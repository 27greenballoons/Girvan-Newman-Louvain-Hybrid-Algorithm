import pandas as pd
import networkx as nx
import community as community_louvain
from sklearn.metrics import adjusted_rand_score
import time
import random

random.seed(42)

# louvain 
def louvain(G, nodes_df):
    start_time = time.time()
    print("Running Louvain community detection...")
    
    # resolution=1.0 is standard
    partition = community_louvain.best_partition(G, resolution=1.0)
    
    # does modularity 
    modularity = community_louvain.modularity(partition, G)
    
    nodes_df['detected_group'] = nodes_df['node_id'].map(partition)
    
    runtime = time.time() - start_time
    print(f"Louvain finished in {runtime:.4f}s. Modularity: {modularity:.3f}")
    
    return partition, nodes_df, runtime, modularity

# girvan newman 
def girvan_newman(G, nodes_df, max_iterations=50, patience=5):
    start_time = time.time()
    print("Running Girvan-Newman (Takes Long Time).")
    
    G_copy = G.copy()
    comp_generator = nx.community.girvan_newman(G_copy)
    
   
    initial = [set(c) for c in nx.connected_components(G)]
    best_partition = initial
    best_modularity = nx.community.modularity(G, initial)
    no_improvement_streak = 0
    
    try:
        for i, partition in enumerate(comp_generator):
            partition = [set(c) for c in partition]
            current_count = len(partition)
            
            try:
                mod = nx.community.modularity(G, partition)
            except Exception:
                continue
            
            # Progress print for my sanity :)
            if i % 5 == 0:
                print(f"   Step {i}: {current_count} communities, modularity={mod:.4f}")
            
            if mod > best_modularity:
                best_modularity = mod
                best_partition = partition
                no_improvement_streak = 0
            else:
                no_improvement_streak += 1
            
            # 3 stopping conditions 
            if no_improvement_streak >= patience:
                print(f"Stopping: no modularity improvement in {patience} splits.")
                break
            
            # max iterations to prevent long 
            if i + 1 >= max_iterations:
                print(f"Stopping: hit max_iterations ({max_iterations}).")
                break
            
            # timeout scales with graph size: 15 min for 1000-node graphs, 5 min otherwise
            timeout = 900 if len(G.nodes()) >= 1000 else 300
            if time.time() - start_time > timeout:
                print(f"Girvan-Newman timed out at {timeout // 60} minutes.")
                break
    
    except StopIteration:
        print("GN exhausted the graph structure.")
    
    # Node mapping 
    partition_dict = {}
    for group_id, node_set in enumerate(best_partition):
        for node in node_set:
            partition_dict[node] = group_id
    
    nodes_df['detected_group'] = nodes_df['node_id'].map(partition_dict)
    
    runtime = time.time() - start_time
    print(f"Girvan-Newman finished in {runtime:.2f}s.")
    print(f"   Final: {len(best_partition)} communities, modularity={best_modularity:.4f}")
    
    return partition_dict, nodes_df, runtime, best_modularity

def GN_Louvain_Hybrid(G, nodes_df):
    start_time = time.time()
    print("Running GN-Louvain Hybrid..")

    print("Step 1: First pass with Louvain...")
    first_pass = community_louvain.best_partition(G, resolution=1.0)
    
    # final comm assignments 
    final_partition = {}
    next_group_id = 0

    print("Step 2: Subcommunity discovery in Louvain-discovered communities with Girvan-Newman...")
    for comm_id in set(first_pass.values()):
        # nodes in this community
        nodes_in_comm = []
        for node, cid in first_pass.items():
            if cid == comm_id:
                nodes_in_comm.append(node)
        
        # create a smaller networkx graph of just these nodes
        subgraph = G.subgraph(nodes_in_comm).copy() 
        
        # only refine the largest communities to avoid over-splitting
        threshold = max(60, len(G.nodes()) // 15)
        if len(nodes_in_comm) > threshold:
            # iterative GN: remove highest-betweenness edge, recompute, repeat
            # stop once the subgraph fractures into more pieces - that's the bridge
            initial_components = nx.number_connected_components(subgraph)
            for _ in range(min(10, len(nodes_in_comm) // 10)):
                if subgraph.number_of_edges() == 0:
                    break
                edge_scores = nx.edge_betweenness_centrality(subgraph)
                if not edge_scores:
                    break
                # finds the edge with the highest level of betweenness centrality, and removes it
                worst_edge = max(edge_scores, key=edge_scores.get)  # type: ignore
                subgraph.remove_edge(*worst_edge)
                # stop once we've fractured into more pieces - GN's job is done
                if nx.number_connected_components(subgraph) > initial_components:
                    break

            # second pass with louvain at lower resolution to bias toward fewer, larger subcommunities 
            second_pass = community_louvain.best_partition(subgraph, resolution=0.5)
            
            # merge tiny subcommunities (< 5 nodes) into the largest one to avoid singletons
            sub_sizes = {}
            for node, sid in second_pass.items():
                sub_sizes[sid] = sub_sizes.get(sid, 0) + 1
            
            largest_sid = max(sub_sizes, key=sub_sizes.get)  # type: ignore
            for node, sid in list(second_pass.items()):
                if sub_sizes[sid] < 5:
                    second_pass[node] = largest_sid

            for node, sub_comm_id in second_pass.items():
                final_partition[node] = next_group_id + sub_comm_id
            
            next_group_id += max(second_pass.values()) + 1
        else:
            for node in nodes_in_comm:
                final_partition[node] = next_group_id
            next_group_id += 1
            
    nodes_df['detected_group'] = nodes_df['node_id'].map(final_partition)

    final_modularity = community_louvain.modularity(final_partition, G)

    runtime = time.time() - start_time
    print(f"GN-Louvain Hybrid finished in {runtime:.2f}s.")
    print(f"   Final: {len(set(final_partition.values()))} communities, modularity={final_modularity:.4f}")
    
    return final_partition, nodes_df, runtime, final_modularity

def accuracy_check(nodes_df, algo_name, runtime, modularity):
    nodes_df['detected_group'] = nodes_df['detected_group'].fillna(-1)
    ari_score = adjusted_rand_score(nodes_df['primary_group'], nodes_df['detected_group'])
    
    total_groups = nodes_df['detected_group'].nunique()
    
    # community size distribution stats
    size_counts = nodes_df.groupby('detected_group').size()
    avg_size = size_counts.mean()
    largest = size_counts.max()
    smallest = size_counts.min()
    
    print(f"\n--- {algo_name} Detection Summary ---")
    print(f"Runtime: {runtime:.2f}s")
    print(f"Modularity: {modularity:.4f}")
    print(f"Global Accuracy (ARI): {ari_score:.4f}")
    print(f"Total Communities Found: {total_groups}")
    print(f"Avg / Largest / Smallest community size: {avg_size:.1f} / {largest} / {smallest}")
    
    return {
        'algorithm': algo_name,
        'runtime': runtime,
        'modularity': modularity,
        'ari': ari_score,
        'communities': total_groups,
        'avg_community_size': avg_size,
        'largest_community': largest,
        'smallest_community': smallest,
    }

if __name__ == "__main__":
    print("Loading data...")

    # define the sizes we want to benchmark
    sizes = [200, 500, 750]

    for size in sizes:
        print(f"\n{'#' * 60}")
        print(f"# BENCHMARKING SIZE: {size} NODES")
        print('#' * 60)
        
        # each size has its own subdirectory
        data_dir = f'graph_data_{size}'
        
        # load the nodes and edges for this size
        nodes_df = pd.read_csv(f'{data_dir}/nodes.csv')
        edges_df = pd.read_csv(f'{data_dir}/edges.csv')
        G = nx.from_pandas_edgelist(edges_df, 'source', 'target')
        G.add_nodes_from(nodes_df['node_id'])
        
        results = []
        
        print(f"\nRunning Louvain on size {size}...")
        louvain_partition, louvain_nodes, louvain_time, louvain_mod = louvain(G, nodes_df.copy())
        results.append(accuracy_check(louvain_nodes, "Louvain", louvain_time, louvain_mod))
        louvain_nodes.to_csv(f'{data_dir}/louvain_results.csv', index=False)
        
        print(f"\nRunning Hybrid GN-Louvain on size {size}...")
        hybrid_partition, hybrid_nodes, hybrid_time, hybrid_mod = GN_Louvain_Hybrid(G, nodes_df.copy())
        results.append(accuracy_check(hybrid_nodes, "Hybrid", hybrid_time, hybrid_mod))
        hybrid_nodes.to_csv(f'{data_dir}/hybrid_results.csv', index=False)
        
        print(f"\nRunning Girvan-Newman on size {size}...")
        gn_partition, gn_nodes, gn_time, gn_mod = girvan_newman(G, nodes_df.copy(), max_iterations=50, patience=5)
        results.append(accuracy_check(gn_nodes, "Girvan-Newman", gn_time, gn_mod))
        gn_nodes.to_csv(f'{data_dir}/gn_results.csv', index=False)
        
        # final comparison table for this size
        print(f"\n{'=' * 90}")
        print(f"BENCHMARK COMPARISON - {size} NODES")
        print('=' * 90)
        print(f"{'Algorithm':<18}{'Runtime (s)':>14}{'Modularity':>14}{'ARI':>10}{'Communities':>14}{'Avg Size':>12}")
        print('-' * 90)
        for r in results:
            print(f"{r['algorithm']:<18}{r['runtime']:>14.2f}{r['modularity']:>14.4f}{r['ari']:>10.4f}{r['communities']:>14}{r['avg_community_size']:>12.1f}")
        print('=' * 90)
        
        # save the benchmark CSV for this size
        benchmark_df = pd.DataFrame(results)
        benchmark_df.to_csv(f'{data_dir}/benchmark_results.csv', index=False)
        print(f"\nResults saved to {data_dir}/")

    print("\nAll sizes complete.")