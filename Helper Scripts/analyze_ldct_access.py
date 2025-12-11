import pickle

import networkx as nx
import pandas as pd

UTAH_ZCTAS_CSV = "utah_zctas.csv"
GRAPH_PKL = "utah_graph.gpickle"
LDCT_ZCTAS_CSV = "LDCT_utah_hospitals_zcta.csv"
OUTPUT_ACCESS_CSV = "utah_ldct_access.csv"


def load_graph(path: str):
    with open(path, "rb") as f:
        G = pickle.load(f)
    if not isinstance(G, nx.Graph):
        raise TypeError("Loaded object is not a NetworkX Graph")
    return G


def main():
    # Load node metadata
    zctas = pd.read_csv(UTAH_ZCTAS_CSV)
    # Normalize
    if "GEOID" in zctas.columns and "zcta" not in zctas.columns:
        zctas = zctas.rename(columns={"GEOID": "zcta"})
    if "INTPTLAT" in zctas.columns and "lat" not in zctas.columns:
        zctas = zctas.rename(columns={"INTPTLAT": "lat"})
    if "INTPTLONG" in zctas.columns and "lon" not in zctas.columns:
        zctas = zctas.rename(columns={"INTPTLONG": "lon"})

    zctas["zcta"] = zctas["zcta"].astype(str).str.zfill(5)

    # Load graph
    print("Loading graph...")
    G = load_graph(GRAPH_PKL)
    print(f"Graph has {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Load LDCT ZCTAs
    ldct_df = pd.read_csv(LDCT_ZCTAS_CSV)
    ldct_df["zcta"] = ldct_df["zcta"].astype(str).str.zfill(5)

    # Restrict LDCT list to nodes actually present in graph
    ldct_nodes = [z for z in ldct_df["zcta"] if z in G.nodes]
    if not ldct_nodes:
        raise ValueError("No LDCT ZCTAs matched nodes in the graph.")

    print(f"Using {len(ldct_nodes)} LDCT source nodes")

    # Multi-source Dijkstra: shortest travel_time_min to any LDCT node
    # Returns dict: {node: distance}
    dist = nx.multi_source_dijkstra_path_length(
        G, sources=ldct_nodes, weight="travel_time_min"
    )

    # Build result frame
    zctas["travel_time_to_ldct_min"] = zctas["zcta"].map(dist)

    # Some ZCTAs might be disconnected; mark them as NaN
    # Compute a simple need score
    zctas["need_score"] = zctas["population"] * zctas["travel_time_to_ldct_min"]

    # Sort by need
    out = zctas.sort_values("need_score", ascending=False)

    out.to_csv(OUTPUT_ACCESS_CSV, index=False)
    print(f"Saved {OUTPUT_ACCESS_CSV}")
    print(out[["zcta", "population", "travel_time_to_ldct_min", "need_score"]].head(20))


if __name__ == "__main__":
    main()
