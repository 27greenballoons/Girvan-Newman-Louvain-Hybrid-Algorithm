import pandas as pd
import networkx as nx
import random
import os

random.seed(42)

def generate_blended_social_data(num_nodes):
    output_dir = f"graph_data_{num_nodes}"
    os.makedirs(output_dir, exist_ok=True)

    # starting with 10 communities and small 200-400 userbase
    communities = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    nodes_data = []
    
    # NODE CREATION
    for i in range(num_nodes):

        # primary and secondary group to represent what type of following/algorithm one may have
        primary = random.choice(communities)

        if random.random() < 0.4:
            secondary = random.choice(communities)
        else:
            secondary = primary
            
        
        # 5% chance to be an influencer, used for brand deal identification
        is_influencer = random.random() < 0.05

        # follower count based on influencer status 
        if is_influencer:
            follower_cnt = random.randint(5000, 20000)
        else:            
            follower_cnt = random.randint(10, 500)

        # group mix shows blend if it exists 
        if primary != secondary:
            group_mix = f"{primary}-{secondary}"
        else:
            group_mix = f"{primary}"
        
        # attributes for node
        node_attr = {
            "node_id": i,
            "primary_group": primary,
            "secondary_group": secondary,
            "is_influencer": is_influencer,
            "followers": follower_cnt,
            "engagement_rate": round(random.uniform(0.01, 0.1), 3),
            "group_mix": group_mix
        }
        nodes_data.append(node_attr)

    df_nodes = pd.DataFrame(nodes_data)

    # EDGE CREATION
    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            n1 = nodes_data[i]
            n2 = nodes_data[j]
            
            # create two sets and use & to only choose the ones that appear in both
            shared_interests = {n1['primary_group'], n1['secondary_group']} & {n2['primary_group'], n2['secondary_group']}
            
            # if both interests are shared, higher probability of connection 
            if len(shared_interests) >= 2:
                prob = 0.40  # Increased for stronger community cores
            # one common interest = smaller chance of connection
            elif len(shared_interests) == 1:
                prob = 0.10  # Increased to define clear "blends"
            # very small chance of connection if theres nothing in common, but always possible
            else:
                prob = 0.0005 # Reduced significantly to clear out random noise
                
            # Influencers act as global bridges
            if n1["is_influencer"] or n2["is_influencer"]:
                prob += 0.005 # Reduced slightly so they don't accidentally merge all groups

            # roll a random number and see if it will connect or not
            if random.random() < prob:
                edges.append({"source": i, "target": j})

    # save to csvs 
    df_edges = pd.DataFrame(edges)
    df_nodes.to_csv(f"{output_dir}/nodes.csv", index=False)
    df_edges.to_csv(f"{output_dir}/edges.csv", index=False)
    
    # networkx debugging to 
    G = nx.from_pandas_edgelist(df_edges, source="source", target="target")
    G.add_nodes_from(df_nodes["node_id"])  # include isolated nodes
    print(f"Components: {nx.number_connected_components(G)}")
    print(f"Largest component: {len(max(nx.connected_components(G), key=len))} nodes")
    print(f"Isolated nodes: {sum(1 for n in G.nodes if G.degree(n) == 0)}")

if __name__ == "__main__":
    generate_blended_social_data(200)
    generate_blended_social_data(500)
    generate_blended_social_data(750)