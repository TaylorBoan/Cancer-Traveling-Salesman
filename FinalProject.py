import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _():
    from __future__ import annotations
    import marimo as mo
    import networkx as nx
    import matplotlib.pyplot as plt
    import pandas as pd
    import requests, zipfile, io
    import geopandas as gpd
    import requests
    import argparse
    import csv
    import json
    import os
    import re
    import sys
    from typing import Dict, List, Optional, Tuple
    import pickle
    import numpy as np
    from sklearn.neighbors import BallTree
    from pyvis.network import Network
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    import folium
    from pathlib import Path
    return (
        BallTree,
        Path,
        folium,
        mo,
        np,
        nx,
        pd,
        pickle,
        pywrapcp,
        requests,
        routing_enums_pb2,
    )


@app.cell
def _(mo):
    mo.md(r"""
    To see the whole repository that made this notebook possible checkout:

    https://github.com/TaylorBoan/Cancer-Traveling-Salesman.git
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Bio 362 Final Project
    ## Taylor Rubalcava
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Cancer diagnosis and treatment are anything but trivial, requiring highly specialized equipment that many smaller hospitals simply don’t have on-site. This creates a serious problem for rural and less densely populated areas, where access to advanced imaging and diagnostic services is limited. Early detection of cancer is of paramount importance, yet many patients in these regions face long travel times, delayed screenings, and a reduced likelihood of receiving timely care.

    ### One solution used today is mobile cancer screening units, specialized vehicles equipped with mammography, ultrasound, CT, or lab capabilities that travel directly to underserved communities. These units bridge the geographic gap by bringing essential diagnostic technologies to patients instead of requiring patients to travel long distances to large medical centers.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Designing an effective route for these mobile screening units is far from straightforward. Rural populations are not only sparse—they are often geographically fragmented, separated by long distances and wide variations of local need and demand. An optimal route must balance clinical need, population density, travel time, and operational constraints of the mobile unit itself.

    ### Routing therefore becomes a logistics problem as much as a healthcare problem. How can we design a route that maximizes the number of at-risk individuals reached, minimizes idle travel, and ensures predictable, recurring access for communities that need it most?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ###Here we will explore how an optimal route can be calculated using available open-source software and data. To simplify our calculations we will make several assumptions:

    1. We will only consder populations and hospitals found in Utah
    2. We will calculate the optimal route for a mobile unit only serving Utah residents
    3. We will only consder the need for lung cancer screening and the neccesary Low-Dose-CT machine needed. (LDCT)

    ###These assumptions will make our computational requirements more manageable while making it easy to expand our calculations after a proof of concept.

    We are focusing on Utah because that is where I live. We are focusing on lung cancer screening because not only is it one of the most deadly and common types of cancer but it is also the most difficult to screen for. Mobile LDCT buses/trailers (and LDCT machines themsevles) exist but they are rare, expensive, and as such the most in need of an optimal route.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The most obvious model is an undirected graph with weighted nodes and edges. Each node will represent a populated area and each edge represents travel paths between cities. Each node's weight will be determined by it's population and access to local cancer screening facilities. Each edge's weight will be the time required to travel between those two cities. Once we have built our graph we can use it to solve our weighted traveling salesman problem (WTSP).

    ### In summary we need:
    1. Nodes (Locations we _could_ visit)
    2. Nodes' weights (How much those places are in need of mobile screening)
    3. Edges (Paths between locations)
    4. Edges' weights (How long it takes to get between locations, a fair approximation of cost)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The first thing we need are the nodes, or possible locations. ZIP Code Tabulation Areas (ZCTAs) are widely used in healthcare access modeling and so that is what we will use here. Each ZCTA in Utah will be a possible node.

    I was able to find this 2023 data containing a lat and lon point for each zip code in the United States:

    https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2023_Gazetteer/2023_Gaz_zcta_national.zip

    Here it is,  as well as the interpretations for each column:
    GEOID = Zip Code (Revelant to us)
    ALAND = Area Land
    AWATER = Area Water
    ALAND_SQMI = Area Land by Square Mile
    AWATER_SQMI = Area Water by Square Mile
    INTPTLAT = Internal Point Latitude
    INTPTLONG = Internal Point Longitude
    """)
    return


@app.cell
def _(pd):
    zcta_nation = pd.read_csv("./zcta_national.txt")
    zcta_nation.head(10)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Nodes ✅
    ### I also found this census data that has the population for each zip code
    https://api.census.gov/data/2023/acs/acs5?get=NAME,B01003_001E&for=zip%20code%20tabulation%20area:*
    ### We will eventually need this for our node weights calculation.
    ### Unfortunately, neither of these links lead to clean .csv data. The script that read, cleaned, and merged these two pieces of data is found in build_utah_zctas.py. This resulted in this csv file:
    """)
    return


@app.cell
def _(pd):
    merged_zcta = pd.read_csv("./zcta_merged.csv")
    merged_zcta.head(10)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Which I then manually filtered and cleaned to Utah only Zip Codes/GEOIDs while inspecting it in Numbers:
    """)
    return


@app.cell
def _(pd):
    utah_zcta = pd.read_csv("./utah_zctas.csv")
    utah_zcta.head(10)
    print(len(utah_zcta))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Now we have the nodes and their populations. This is enough to begin building the graph.

    ### We could simply connect every node to every other node, but that would be an inefficient N^2 graph with ~88k edges. Ideally, we only connect nodes/zip codes to their neighboring nodes/zipcodes. We will use the K-Nearest Neighbors (KNN) algorithm to determine edge locations.

    ### We also need to determine the weights for those edges. The weight for any given edge is how long it takes to drive between the two nodes/zip codes that edge connects. Unfortunately the Google Maps API is paid and requires API Keys. Instead I will use Open Street Maps and Open Street Maps and Open Source Routing Machine. OSM and OSRM respectively. To do this I had to download the OSM data for Utah and set up a local OSMR server.

    ### In summary we will use:
    1. Sklearn to find each nodes k nearest neighbors / edges
    2. OSMR to get the travel time/weights for each of those edges
    3. NetworkX to create the graph including the nodes, edges, and edge weights

    Note, that in order for the following code to work the OSMR server must be running on port 5001
    These are the commands that I used to download the neccesary OSM data and start the OSMR server:

    * For more detailsa about how to setup this server locally see the bottom of this notebook
    """)
    return


@app.cell
def _(BallTree, np, nx, pd, pickle, requests):
    OSRM_BASE = "http://localhost:5001"

    INPUT_CSV = "utah_zctas.csv"
    OUTPUT_EDGES_CSV = "utah_edges.csv"
    OUTPUT_GRAPH_PKL = "utah_graph.gpickle"
    K_NEIGHBORS = 6  # neighbors per node


    def load_zctas(path: str) -> pd.DataFrame:
        df = pd.read_csv(path)

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

        # Keep only Utah ZCTAs (84xxx)
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
        coords = np.radians(df[["lat", "lon"]].values)  # expects radians
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

            # Be nice to OSRM:
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
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### We now have the whole graph minus the node weights. To determine the node weights we will need to know the population of each zip code _and_ the distance to the nearest LCDT machine.

    ### We already have the population data. We need to retrieve the location of every hospital in Utah that has a LCDT machines and thus lung cancer screening capabilities. This information is tracked by the government via their "Centers for Medicare & Medicaid services" website:

    https://data.cms.gov/provider-summary-by-type-of-service/medicare-physician-other-practitioners

    ### Let's take a look at it:
    """)
    return


@app.cell
def _(pd):
    LCDT_hospitals = pd.read_csv("./LDCT_hospitals.csv")
    LCDT_hospitals.head(10)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### As you can see, this CSV contains a row for every hospital with LDCT machine in the US with its zip code and address.

    ### I pre-filtered the CSV file from the CMS website to only include hospitals with the 71271 code which is the Medicare/Medicaid code for LDCT screenings, according to their website:

    https://www.cms.gov/medicare-coverage-database/view/article.aspx?articleid=58641

    While inspecting the data I manually cleaned by removing everything except for UT hospitals and their zip codes this to get to the following dataframe
    """)
    return


@app.cell
def _(pd):
    LDCT_utah_hospitals_zcta = pd.read_csv("./LDCT_utah_hospitals_zcta.csv")
    LDCT_utah_hospitals_zcta.head(100)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### These are the zipcodes in Utah that have LDCT machines. We now have everything we need to determine the node weights.

    ### Node Weight = "The travel time from this node/zip code to the nearest node/zip code with a LDCT machine"

    ### This is computed using Dijkstra's Algorithm
    """)
    return


@app.cell
def _(nx, pd, pickle):
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
    # print(out[["zcta", "population", "travel_time_to_ldct_min", "need_score"]].head(20))
    return


@app.cell
def _(pd):
    utah_ldct_access = pd.read_csv("./utah_ldct_access.csv")
    utah_ldct_access.head(100)
    return (utah_ldct_access,)


@app.cell
def _(utah_ldct_access):
    # Top 20 by need_score
    top_need = utah_ldct_access.sort_values("need_score", ascending=False).head(20)
    print(top_need[["zcta", "population", "travel_time_to_ldct_min", "need_score"]])

    # How many ZCTAs are missing access (no path)?
    print("\n")
    print("NaN travel time:", utah_ldct_access["travel_time_to_ldct_min"].isna().sum())
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### These are the 20 nodes/zip code that would benefit the most from a mobile LDCT visit. Obviously the number of zip codes that could actually be visited would be dependent on the resource constraints on the particular mobile LDCT unit, but I thought that 20 would be sufficient to present a proof of concept.

    ### The last thing left to do is actually calculate the optimal path to these 20 zip codes, given the edges (and weights) of our graph.
    """)
    return


@app.cell
def _(nx, pd, pickle, pywrapcp, routing_enums_pb2):
    def _():
        def solve_tsp(distance_matrix, start_index):
            """Solve TSP using OR-Tools."""
            n = len(distance_matrix)

            # 1 vehicle, depot at start_index
            manager = pywrapcp.RoutingIndexManager(n, 1, start_index)
            routing = pywrapcp.RoutingModel(manager)

            def cost_callback(from_index, to_index):
                f = manager.IndexToNode(from_index)
                t = manager.IndexToNode(to_index)
                return distance_matrix[f][t]

            transit_callback_idx = routing.RegisterTransitCallback(cost_callback)
            routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_idx)

            # Search parameters
            search_parameters = pywrapcp.DefaultRoutingSearchParameters()
            search_parameters.first_solution_strategy = (
                routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
            )
            # Optional: add a bit of local search for better routes
            search_parameters.local_search_metaheuristic = (
                routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
            )
            search_parameters.time_limit.seconds = 30  # 30 seconds search time

            solution = routing.SolveWithParameters(search_parameters)
            if not solution:
                raise RuntimeError("No solution found")

            # Extract route as list of node indices (0..n-1)
            route = []
            index = routing.Start(0)
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                route.append(node)
                index = solution.Value(routing.NextVar(index))
            # Add final depot (so route starts and ends at SLC)
            route.append(manager.IndexToNode(index))

            return route


        GRAPH_PKL = "utah_graph.gpickle"
        NEED_SCORE_CSV = "utah_ldct_access.csv"
        START_ZCTA = "84101"  # Salt Lake City downtown (can change)

        # How many top sites to route (use your existing list)
        TOP_N = 21


        def load_graph():
            with open(GRAPH_PKL, "rb") as f:
                G = pickle.load(f)
            return G


        def build_distance_matrix(G, nodes):
            """Compute pairwise travel times (in minutes) for TSP."""
            n = len(nodes)
            dist = [[0] * n for _ in range(n)]

            for i, a in enumerate(nodes):
                for j, b in enumerate(nodes):
                    if i == j:
                        dist[i][j] = 0
                        continue
                    # Use your graph's edge weights (multi-hop shortest path)
                    t = nx.shortest_path_length(G, a, b, weight="travel_time_min")
                    dist[i][j] = int(t)  # OR-Tools requires integers
            return dist


        def solve_tsp(distance_matrix, start_index):
            """Solve TSP using OR-Tools."""
            n = len(distance_matrix)

            # 1 vehicle, depot at start_index
            manager = pywrapcp.RoutingIndexManager(n, 1, start_index)
            routing = pywrapcp.RoutingModel(manager)

            def cost_callback(from_index, to_index):
                f = manager.IndexToNode(from_index)
                t = manager.IndexToNode(to_index)
                return distance_matrix[f][t]

            transit_callback_idx = routing.RegisterTransitCallback(cost_callback)
            routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_idx)

            # Search parameters
            search_parameters = pywrapcp.DefaultRoutingSearchParameters()
            search_parameters.first_solution_strategy = (
                routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
            )
            # Add a bit of local search for better routes
            search_parameters.local_search_metaheuristic = (
                routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
            )
            search_parameters.time_limit.seconds = 30  # 30 seconds search time

            solution = routing.SolveWithParameters(search_parameters)
            if not solution:
                raise RuntimeError("No solution found")

            # Extract route as list of node indices (0..n-1)
            route = []
            index = routing.Start(0)
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                route.append(node)
                index = solution.Value(routing.NextVar(index))
            # Add final depot (so route starts and ends at SLC)
            route.append(manager.IndexToNode(index))

            return route



        print("Loading graph...")
        G = load_graph()

        print("Loading need-score data...")
        df = pd.read_csv(NEED_SCORE_CSV)
        df["zcta"] = df["zcta"].astype(str)

        # Select top N highest-need ZCTAs
        top = df.sort_values("need_score", ascending=False).head(TOP_N)
        targets = list(top["zcta"])

        # Make sure starting ZCTA is included
        if START_ZCTA not in targets:
            targets = [START_ZCTA] + targets

        print("Selected ZCTAs:", targets)

        # Build distance matrix
        print("Computing pairwise travel times...")
        dist_matrix = build_distance_matrix(G, targets)

        # Determine start index
        start_idx = targets.index(START_ZCTA)

        print("Solving TSP...")
        route_idx = solve_tsp(dist_matrix, start_idx)

        route = [targets[i] for i in route_idx]

        # print("\nOptimal Route:")
        # for z in route:
        #     print(z)

        # Save route to CSV
        pd.DataFrame({"zcta": route}).to_csv("optimal_route.csv", index=False)
        return print("\nSaved optimal_route.csv")

    _()
    return


@app.cell
def _(pd):
    optimal_route = pd.read_csv("./optimal_route.csv")
    optimal_route.head(20)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Starting in SLC (84101), and then returning to SLC in a loop, this is the optimal zip code visitation path.
    """)
    return


@app.cell
def _(folium, pd):
    # Here we simply display that optimal route in an interactive box
    def _():
        ZCTAS_CSV = "utah_zctas.csv"
        ROUTE_CSV = "optimal_route.csv"
        OUTPUT_HTML = "utah_ldct_route.html"

        # Load ZCTA centroids
        zctas = pd.read_csv(ZCTAS_CSV)
        # Normalize column names
        zctas.columns = [c.lower() for c in zctas.columns]

        if "geoid" in zctas.columns and "zcta" not in zctas.columns:
            zctas = zctas.rename(columns={"geoid": "zcta"})
        if "intptlat" in zctas.columns and "lat" not in zctas.columns:
            zctas = zctas.rename(columns={"intptlat": "lat"})
        if "intptlong" in zctas.columns and "lon" not in zctas.columns:
            zctas = zctas.rename(columns={"intptlong": "lon"})

        zctas["zcta"] = zctas["zcta"].astype(str).str.zfill(5)

        # Load route (ordered ZCTAs)
        route = pd.read_csv(ROUTE_CSV)
        route.columns = [c.lower() for c in route.columns]
        route["zcta"] = route["zcta"].astype(str).str.zfill(5)

        # Join route with coordinates
        route_coords = route.merge(zctas[["zcta", "lat", "lon"]], on="zcta", how="left")

        if route_coords["lat"].isna().any():
            missing = route_coords[route_coords["lat"].isna()]["zcta"].unique()
            raise ValueError(f"Missing coordinates for ZCTAs: {missing}")

        # Create list of (lat, lon) points in route order
        points = list(zip(route_coords["lat"], route_coords["lon"]))

        # Center map on the mean of all route points
        center_lat = route_coords["lat"].mean()
        center_lon = route_coords["lon"].mean()

        m = folium.Map(location=[center_lat, center_lon], zoom_start=7, tiles="OpenStreetMap")

        # Add straight-line polyline for the route
        folium.PolyLine(
            locations=points,
            weight=4,
            opacity=0.8,
        ).add_to(m)

        # Add markers for each stop
        for _, row in route_coords.iterrows():
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=4,
                fill=True,
                fill_opacity=0.9,
                popup=f"ZCTA {row['zcta']}",
            ).add_to(m)

        # Highlight the start/end (Salt Lake City)
        start = route_coords.iloc[0]
        folium.Marker(
            location=[start["lat"], start["lon"]],
            popup=f"Start (ZCTA {start['zcta']})",
            icon=folium.Icon(color="green", icon="play"),
        ).add_to(m)

        end = route_coords.iloc[-1]
        folium.Marker(
            location=[end["lat"], end["lon"]],
            popup=f"End (ZCTA {end['zcta']})",
            icon=folium.Icon(color="red", icon="stop"),
        ).add_to(m)

        # Save to HTML
        m.save(OUTPUT_HTML)
        return print(f"Saved map to {OUTPUT_HTML}")


    _()
    return


@app.cell
def _(Path, mo):
    html = Path("utah_ldct_route.html").read_text()
    mo.iframe(html, height="600px")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### As you can see the route matches intutition.

    ### There are of course several things you could do to tune this route builder
    ### First and foremost, depending on empircale data you might opt to make the need_score function more sophisticated than the simple product of population and distance
    ### On the whole though this is a robust process that can easily be adapted to any other state or even the entire US. Given the data already found in this repo.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## This is what you need to know to get the OSM data and the local OSMR server running.

    ## If you have cloned the repo all you should need is the Docker Image and to start the server. Otherwise, you'll need to download the OSM data.

    ### To get the environment set-up, first you will need to download the Utah OSM data:

    bash: curl -L -o utah.osm.pbf "https://download.geofabrik.de/north-america/us/utah-latest.osm.pbf"

    ### You will also need to install the OSRM (Open Street Routing Machine) docker image:

    bash: docker pull osrm/osrm-backend

    ### Then you will need to run some additional commands to get the OSRM server up and running locally:

    1) Extract for car profile
    docker run -t -v $(pwd):/data osrm/osrm-backend osrm-extract -p /opt/car.lua /data/utah-latest.osm.pbf

    3) Contract (build routing graph)
    docker run -t -v $(pwd):/data osrm/osrm-backend osrm-contract /data/utah-latest.osrm

    ### Finally, you can start actually running the server with this command:
    docker run --platform=linux/amd64 -t -i -p 5001:5000 \
      -v "$(pwd)":/data osrm/osrm-backend \
      osrm-routed --algorithm=CH /data/utah.osrm
    """)
    return


if __name__ == "__main__":
    app.run()
