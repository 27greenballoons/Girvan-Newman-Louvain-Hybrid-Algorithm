# Girvan-Newman-Louvain-Hybrid-Algorithm
CSCI 605 Final Project. This algorithm's purpose is to identify different subcommunities in a graph based on a combination of social connections, primary interests, and secondary interests.

# Setup
1. Use requirements.txt to set up a virtual enviroment
2. ```chmod +x run.sh```
3. ``./run.sh`` used to go through all four python scripts for the project

# Files
- **graph_data/ -** the data I gathered during my first parts of the project, not extremely relevant 
- **graph_data_{size}/ -** the data I gathered while testing graphs with 200, 500, and 750 nodes. Includes csvs from the python scripts (the original graph creation, along with the other graphs made after community detection algorithms ran through them). Also includes graphs for runtime, modularity, size distribution for community, communities made, and ARI (adjusted rand score).
- **benchmark_algos.py -** 3 defs for the three algorithms tested, along with CLI printing out results for each, and extraction of the results to CSVs in the different graph_data folders.
- **generate_network.py -** Script used to create nodes and edges for the rest of the scripts to analyze. Creates nodes with primary interest/group, occasionally it can have a secondary group of interest. A user could also be an influencer node and be shown with priority on the network map. These factors are used to make edge connections across the entire graph. 
- **graph_visualization.py -** Turns the nodes and edges from each graph created into a visual graph representation to show relations and communities in a more presentable way.
- **stats_graphs.py -** Turns CSVs into bar graphs that can be used for visual analysis of results.
- **run.sh -** Runs through every python script in order and recreates the simulation for node counts 200, 500, 750. These values are hardcoded and would need to be changed. 
