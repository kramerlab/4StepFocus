from collections import deque
import stark_qa
from stark_qa.skb import SKB


def print_neighbors(node: int, skb: stark_qa.skb.SKB):
    neighbors = skb.get_neighbor_nodes(node)
    print(f"Number of neighbors: {len(neighbors)}. Neighbors:")
    print(neighbors)

    skb.node_info[0]


def dfs(start: int, target: int, skb: stark_qa.skb.SKB, max_depth: int):
    return dfs_step(start, target, skb, set(), max_depth)


def dfs_step(start: int, end: int, skb: stark_qa.skb.SKB, visited, max_depth: int):
    visited.add(start)
    for neighbor in skb.get_neighbor_nodes(start):
        if neighbor == end:
            return [neighbor]
        if max_depth == 0:
            return None
        if neighbor not in visited:
            path_to_end = dfs_step(neighbor, end, skb, visited, max_depth - 1)
            if path_to_end is not None:
                path_to_end.append(neighbor)
                return path_to_end
    return None



def reduce_num_paths(paths: list[int], targets: list[int], limit: int):
    path_cnt2target = dict()
    for target in targets:
        path_cnt2target[target] = 0
    for path in paths:
        path_cnt2target[path[-1]] += 1

    targets_to_remove = set()
    for target in targets:
        if path_cnt2target[target] > limit:
            targets_to_remove.add(target)

    remaining_paths = []
    for path in paths:
        if path[-1] not in targets_to_remove:
            remaining_paths.append(path)
    return remaining_paths, targets_to_remove


def rels_to_unknown(start: int, skb: stark_qa.skb.SKB, type_of_unknown: str) -> (list, dict):
    paths_found = []

    for neighbor in skb.get_neighbor_nodes(start):
        if skb.node_info[neighbor]['type'] == type_of_unknown:
            paths_found.append([start, neighbor])
    return paths_found


def bfs_all_shortest_paths(start: int, targets: list[int], skb: stark_qa.skb.SKB, max_depth: int):
    # Queue for BFS that stores (node, path) tuples
    queue = deque([(start, [start])])
    visited = set()
    targets = targets.copy()
    paths_found = []

    # counters to track search depth
    depth = 0
    num_next_layer_nodes = 0
    targets_reached_at_this_depth = set()

    while queue:
        if depth > max_depth:
            print("Max depth reached. Terminating breadth-first search.")
            break

        # Dequeue a node and its path
        node, path = queue.popleft()

        # If the node is the target, save the path
        if node in targets:
            # check if this path has been found already
            if path not in paths_found:
                paths_found.append(path)
                visited.add(node)  # mark node as visited to not follow this path further and continue queue
                targets_reached_at_this_depth.add(node)

        # enqueue all adjacent nodes with the updated path and mark the node as visited, if last depth not reached yet
        if depth < max_depth:
            if node not in visited:
                visited.add(node)
                for neighbor in skb.get_neighbor_nodes(node):
                    if neighbor not in visited:
                        queue.append((neighbor, path + [neighbor]))
                        num_next_layer_nodes += 1

        # check if all targets are reached
        if len(queue) == num_next_layer_nodes:
            num_next_layer_nodes = 0
            for t in targets_reached_at_this_depth:
                targets.remove(t)
            if depth > max_depth:
                print(f"{depth=} completed.")
            else:
                print(f"{depth=} completed, queue length={len(queue)}.")
            if len(targets) == 0:
                print("all shortest path to all targets found. Terminating search.")
                return paths_found
            targets_reached_at_this_depth = set()
            depth += 1

    # If no target is reached but queue is empty
    print("End of search queue reached.")
    return paths_found



def bfs_all_shortest_paths_not_faster(start: int, targets: list[int], skb: stark_qa.skb.SKB, max_depth: int):
    # Queue for BFS that stores (node, path) tuples
    next_queue = deque([(start, [start])])
    visited = set()
    targets = targets.copy()
    paths_found = []

    for depth in range(max_depth):
        queue = next_queue
        next_queue = deque()
        targets_reached_at_this_depth = set()

        while queue:
            # Dequeue a node and its path
            node, path = queue.popleft()

            # If the node is the target, save the path
            if node in targets:
                paths_found.append(path)
                visited.add(node)  # mark node as visited to not follow this path further and continue queue
                targets_reached_at_this_depth.add(node)

            # enqueue all adjacent nodes with the updated path and mark the node as visited
            if node not in visited:
                visited.add(node)
                for neighbor in skb.get_neighbor_nodes(node):
                    if neighbor not in visited:
                        next_queue.append((neighbor, path + [neighbor]))

        # check if all targets are reached
        for t in targets_reached_at_this_depth:
            targets.remove(t)
        print(f"{depth=} completed, queue length={len(next_queue)}.")
        if len(targets) == 0:
            print("All shortest path to all targets found. Terminating search.")
            return paths_found

    # on max depth, don't add anything new to queue, just check for targets reached
    for depth in [max_depth]:
        queue = next_queue
        targets_reached_at_this_depth = set()
        while queue:
            # Dequeue a node and its path
            node, path = queue.popleft()

            # If the node is the target, save the path
            if node in targets:
                paths_found.append(path)
                visited.add(node)  # mark node as visited to not follow this path further and continue queue
                targets_reached_at_this_depth.add(node)

        # check if all targets are reached
        for t in targets_reached_at_this_depth:
            targets.remove(t)
        print(f"{depth=} completed.")
        if len(targets) == 0:
            print("All shortest path to all targets found. Terminating search.")
            return paths_found

    print("Max depth reached. Terminating breadth-first search.")
    return paths_found


def find_edge_type(u: int, v: int, skb: stark_qa.skb.SKB):
    for edge_type in skb.rel_type_lst():
        neighbors = skb.get_neighbor_nodes(u, edge_type)
        if v in neighbors:
            return edge_type
    return None


def get_target_neighbors_of_certain_type(node_ids: [int], max_path_to_unknowns: int, type_of_unknown: str, skb: SKB):
    paths = []
    truncated = False

    for i in range(len(node_ids)):
        paths += rels_to_unknown(node_ids[i], skb, type_of_unknown)

    if len(paths) > max_path_to_unknowns:
        truncated = True
        neighbor_frequency_counter = {}
        for path in paths:
            if path[-1] in neighbor_frequency_counter:
                neighbor_frequency_counter[path[-1]] += 1
            else:
                neighbor_frequency_counter[path[-1]] = 1
        max_connection_cnt = max(neighbor_frequency_counter.values())
        filtered_path = []
        for path in paths:
            if neighbor_frequency_counter[path[-1]] == max_connection_cnt:
                filtered_path.append(path)
        paths = filtered_path

    return paths, truncated
