import numpy as np
from dm_alchemy.types import graphs as dm_graphs
from dm_alchemy.types import stones_and_potions # For Potion, LatentPotion
from dm_alchemy import event_tracker # For GameState

def modify_graph_cauldron_idx(graph: dm_graphs.Graph, new_potion_idx_for_cauldron: int):
    """
    Modifies the graph's edges in place.
    If an edge has a potion_idx of -1 (conventionally for cauldron),
    it's replaced with new_potion_idx_for_cauldron.

    Args:
        graph: The dm_alchemy.types.graphs.Graph object to modify.
        new_potion_idx_for_cauldron: The new potion index to use for edges
                                     that originally had -1.
    """
    if not graph or not hasattr(graph, 'edge_list') or not hasattr(graph.edge_list, 'edges'):
        return

    modified_edges = []
    for edge in graph.edge_list.edges:
        if hasattr(edge, 'potion_idx') and edge.potion_idx == -1:
            # Reconstruct the edge with the new potion_idx
            # Assuming Edge is a namedtuple or class constructible with these args
            # and original fields are start_node_idx, end_node_idx, potion_idx
            try:
                modified_edge = type(edge)(
                    start_node_idx=edge.start_node_idx,
                    end_node_idx=edge.end_node_idx,
                    potion_idx=new_potion_idx_for_cauldron
                )
                modified_edges.append(modified_edge)
            except TypeError:
                # Fallback if constructor is not as assumed, keep original
                # Or handle more gracefully depending on expected Edge structure
                modified_edges.append(edge)
        else:
            modified_edges.append(edge)
    
    graph.edge_list.edges = modified_edges

def update_graph_node_indices_with_potions(graph, existing_potions):
    """
    Updates nodes in the graph with idx=-1 to the correct potion instance index
    if their latent coordinates match a potion in existing_potions.

    Args:
        graph: The chemistry graph object (e.g., dm_alchemy.types.graphs.Graph).
               Expected to have a 'nodes' attribute which is a list of node objects.
               Each node object is expected to have 'idx' and 'latent_coords' attributes.
        existing_potions: A list of Potion objects (e.g., dm_alchemy.types.stones_and_potions.Potion).
                          Each Potion object is expected to have an 'idx' attribute (instance index)
                          and a 'latent_potion()' method returning an object with 'latent_coords'.

    Returns:
        The modified graph.
    """
    if not graph or not hasattr(graph, 'nodes') or not graph.nodes:
        print("Warning (update_graph_node_indices_with_potions): Graph is None, has no 'nodes' attribute, or 'nodes' is empty. Returning graph as is.")
        return graph

    if existing_potions is None:
        print("Warning (update_graph_node_indices_with_potions): existing_potions is None. Returning graph as is.")
        return graph
        
    potion_coord_to_idx_map = {}
    for potion_obj in existing_potions:
        if hasattr(potion_obj, 'latent_potion') and callable(potion_obj.latent_potion) and \
           hasattr(potion_obj, 'idx'):
            latent_potion = potion_obj.latent_potion()
            if hasattr(latent_potion, 'latent_coords'):
                try:
                    # Convert numpy array to tuple of its elements to be hashable
                    coords_tuple = tuple(np.array(latent_potion.latent_coords).flatten())
                    if coords_tuple not in potion_coord_to_idx_map:
                        potion_coord_to_idx_map[coords_tuple] = potion_obj.idx
                except Exception as e:
                    print(f"Warning (update_graph_node_indices_with_potions): Could not process latent_coords for potion_obj {potion_obj}. Error: {e}")
            else:
                print(f"Warning (update_graph_node_indices_with_potions): Potion object's latent_potion {latent_potion} missing 'latent_coords'.")
        else:
            print(f"Warning (update_graph_node_indices_with_potions): Potion object {potion_obj} missing 'latent_potion' method or 'idx' attribute.")

    if not potion_coord_to_idx_map:
        print("Warning (update_graph_node_indices_with_potions): Potion coordinate map is empty. No node indices will be updated.")


    for node in graph.nodes:
        if hasattr(node, 'idx') and node.idx == -1 and hasattr(node, 'latent_coords'):
            try:
                node_coords_tuple = tuple(np.array(node.latent_coords).flatten())
                if node_coords_tuple in potion_coord_to_idx_map:
                    node.idx = potion_coord_to_idx_map[node_coords_tuple]
            except Exception as e:
                print(f"Warning (update_graph_node_indices_with_potions): Could not process latent_coords for node {node}. Error: {e}")
        elif not (hasattr(node, 'idx') and hasattr(node, 'latent_coords')):
             print(f"Warning (update_graph_node_indices_with_potions): Node {node} missing 'idx' or 'latent_coords' attribute.")
            
    return graph

def resolve_graph_potion_indices(
    graph: dm_graphs.Graph,
    potions_in_trial: list[stones_and_potions.LatentPotion]
) -> dm_graphs.Graph:
    """
    Modifies the graph's node indices in-place.
    If a node has idx = -1, it attempts to find a matching potion
    in potions_in_trial based on latent_coords. If found,
    the node's idx is updated to the index of that potion in the list.

    Args:
        graph: The graph object to modify.
        potions_in_trial: A list of LatentPotion objects for the current trial.

    Returns:
        The modified graph.
    """
    for node in graph.nodes:
        if node.idx == -1:
            # Ensure node.latent_coords is not None
            if node.latent_coords is None:
                continue # Or handle as an error/warning

            for potion_idx, potion in enumerate(potions_in_trial):
                # Ensure potion.latent_coords is not None
                if potion.latent_coords is None:
                    continue

                if np.array_equal(node.latent_coords, potion.latent_coords):
                    node.idx = potion_idx
                    break  # Found a match, move to the next node
    return graph

def update_graph_potion_indices(graph: dm_graphs.Graph, game_state: event_tracker.GameState):
    """
    Modifies the graph's node indices in place.
    If a node in the graph has idx=-1, this function attempts to update it
    with the correct potion index by matching latent_coords with existing potions
    in the game_state.

    Args:
        graph: The dm_alchemy.types.graphs.Graph object to modify.
        game_state: The dm_alchemy.event_tracker.GameState object containing
                    current trial's potion information.

    Returns:
        The modified graph.
    """
    if not game_state or not hasattr(game_state, 'existing_potions') or not game_state.existing_potions():
        # Optionally, log a warning or return early if game_state is not as expected
        # print("Warning: game_state is None or has no existing_potions. Cannot update graph indices.")
        return graph

    # Create a mapping from tuple(latent_coords) to potion.idx for quick lookup
    potion_coord_to_idx_map = {}
    for potion in game_state.existing_potions():
        # Ensure potion object has the necessary attributes
        if hasattr(potion, 'latent_potion') and callable(potion.latent_potion) and \
           hasattr(potion.latent_potion(), 'latent_coords') and hasattr(potion, 'idx'):
            # Convert numpy array to tuple to be hashable for dict keys
            coords_tuple = tuple(potion.latent_potion().latent_coords)
            potion_coord_to_idx_map[coords_tuple] = potion.idx
        # else:
            # print(f"Warning: Potion object {potion} does not have expected attributes for coord mapping.")

    if not hasattr(graph, 'nodes'):
        # print("Warning: Graph object does not have 'nodes' attribute.")
        return graph

    for node in graph.nodes:
        # Ensure node object has the necessary attributes
        if hasattr(node, 'idx') and node.idx == -1 and hasattr(node, 'latent_coords'):
            node_coords_tuple = tuple(node.latent_coords)
            if node_coords_tuple in potion_coord_to_idx_map:
                # print(f"Updating node idx from -1 to {potion_coord_to_idx_map[node_coords_tuple]} for coords {node_coords_tuple}")
                node.idx = potion_coord_to_idx_map[node_coords_tuple]
            # else:
                # print(f"Node with idx=-1 and coords {node_coords_tuple} not found in existing potions.")
        # elif not (hasattr(node, 'idx') and node.idx == -1):
            # pass # Not a target node or already has a valid idx
        # else:
            # print(f"Warning: Node object {node} does not have expected attributes for update check.")
            
    return graph
