import pickle

import networkx as nx
import numpy as np
import pandas as pd
import requests

# from networkx.readwrite import gpickle
from sklearn.neighbors import BallTree

OSRM_BASE = "http://localhost:5001"

# === CONFIG ===
INPUT_CSV = "utah_zctas.csv"  # your file
OUTPUT_EDGES_CSV = "utah_edges.csv"
OUTPUT_GRAPH_PKL = "utah_graph.gpickle"
K_NEIGHBORS = 6  # neighbors per node; tweak if you like


def load_zctas(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Normalize column names if needed
    cols = {c.lower(): c for c in df.columns}

    # Handle both original and cleaned forms
    if "geoid" in cols:
        df.rename(columns={cols["geoid"]: "zcta"}, inplace=True)
    elif "zcta" not in df.columns:
        raise ValueError("Could not find GEOID or zcta column in CSV")

    if "intptlat" in cols:
        df.rename(columns={cols["intptlat"]: "lat"}, inplace=True)
    if "intptlong" in cols:
        df.rename(columns={cols["intptlong"]: "lon"}, inplace=True)

    # Ensure required columns
    for col in ["zcta", "lat", "lon", "population"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Keep only Utah ZCTAs (84xxx), just in case
    df["zcta"] = df["zcta"].astype(str)
    df = df[df["zcta"].str.startswith("84")].copy()

    df["lat"] = df["lat"].astype(float)
    df["lon"] = df["lon"].astype(float)
    df["population"] = (
        pd.to_numeric(df["population"], errors="coerce").fillna(0).astype(int)
    )

    df = df.reset_index(drop=True)
    return df


def build_knn_edges(df: pd.DataFrame, k: int) -> list[tuple[int, int]]:
    """Return list of undirected edges as index pairs (i, j)."""
    coords = np.radians(df[["lat", "lon"]].values)  # haversine expects radians
    tree = BallTree(coords, metric="haversine")

    # +1 because the nearest neighbor is itself
    distances, indices = tree.query(coords, k=k + 1)

    edge_set = set()

    n = len(df)
    for i in range(n):
        for j_idx in indices[i][1:]:  # skip itself
            j = int(j_idx)
            u, v = sorted((i, j))
            edge_set.add((u, v))

    edges = list(edge_set)
    print(f"KNN created {len(edges)} unique edges for {n} nodes")
    return edges


def osrm_travel_time_minutes(lat1, lon1, lat2, lon2) -> float:
    """Get driving time in minutes via OSRM."""
    url = f"{OSRM_BASE}/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=false"
    r = requests.get(url)
    r.raise_for_status()
    data = r.json()

    if data.get("code") != "Ok":
        raise RuntimeError(f"OSRM error: {data}")

    secs = data["routes"][0]["duration"]
    return secs / 60.0


def build_graph(df: pd.DataFrame, edges_idx: list[tuple[int, int]]) -> nx.Graph:
    G = nx.Graph()

    # Add nodes with attributes
    for i, row in df.iterrows():
        G.add_node(
            row["zcta"],
            lat=row["lat"],
            lon=row["lon"],
            population=int(row["population"]),
        )

    # Add edges with travel_time_min
    edge_rows = []
    for count, (i, j) in enumerate(edges_idx, start=1):
        a = df.iloc[i]
        b = df.iloc[j]

        try:
            t_min = osrm_travel_time_minutes(a["lat"], a["lon"], b["lat"], b["lon"])
        except Exception as e:
            print(f"OSRM failed for {a['zcta']} – {b['zcta']}: {e}")
            continue

        G.add_edge(a["zcta"], b["zcta"], travel_time_min=t_min)

        edge_rows.append(
            {
                "zcta_u": a["zcta"],
                "zcta_v": b["zcta"],
                "travel_time_min": t_min,
            }
        )

        if count % 50 == 0:
            print(f"Processed {count}/{len(edges_idx)} edges...")

        # Be nice to OSRM if you want:
        # time.sleep(0.005)

    edges_df = pd.DataFrame(edge_rows)
    return G, edges_df


def main():
    print("Loading ZCTAs...")
    df = load_zctas(INPUT_CSV)
    print(f"Loaded {len(df)} Utah ZCTAs")

    print("Building KNN edges...")
    edges_idx = build_knn_edges(df, K_NEIGHBORS)

    print("Querying OSRM and building graph...")
    G, edges_df = build_graph(df, edges_idx)

    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    print(f"Saving edges to {OUTPUT_EDGES_CSV} ...")
    edges_df.to_csv(OUTPUT_EDGES_CSV, index=False)

    print(f"Saving graph to {OUTPUT_GRAPH_PKL} ...")
    with open(OUTPUT_GRAPH_PKL, "wb") as f:
        pickle.dump(G, f)

    print("Done.")


if __name__ == "__main__":
    main()
