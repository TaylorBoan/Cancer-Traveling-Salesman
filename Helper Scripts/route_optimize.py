import pickle

import networkx as nx
import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2


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
TOP_N = 20


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


def main():
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

    print("\nOptimal Route:")
    for z in route:
        print(z)

    # Save route to CSV
    pd.DataFrame({"zcta": route}).to_csv("optimal_route.csv", index=False)
    print("\nSaved optimal_route.csv")


if __name__ == "__main__":
    main()
