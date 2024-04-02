import heapq
from collections import deque
# from Queue import PriorityQueue
from queue import PriorityQueue
from itertools import combinations

def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "astar": astar,
        "astar_corner": astar_corner,
        "astar_multi": astar_multi,
        "fast": fast,
    }.get(searchMethod)(maze)

# -------------------------------------------- PART I --------------------------------------------
"""pass all test data"""
# ==================== This is for BFS ====================
# This version can fit single dot, multi dots and corner problem
def bfs(maze):
    start = maze.getStart()
    objectives = set(maze.getObjectives())  # 获取所有目标点
    start_state = (start, frozenset())  # 初始状态包含起始位置和空的已达成目标集合
    
    queue = deque([start_state])
    visited = set([start_state])
    path = {}
    
    while queue:
        current_position, achieved_goals = queue.popleft()
        if objectives.issubset(achieved_goals):  # 如果所有目标都已达成
            return reconstruct_path_bfs(path, start_state, (current_position, achieved_goals))
        for neighbor in maze.getNeighbors(*current_position):
            new_achieved_goals = set(achieved_goals)
            if neighbor in objectives:
                new_achieved_goals.add(neighbor)
            next_state = (neighbor, frozenset(new_achieved_goals))
            if next_state not in visited:
                visited.add(next_state)
                queue.append(next_state)
                path[next_state] = (current_position, achieved_goals)
    return []

def reconstruct_path_bfs(path, start_state, goal_state):
    current_state = goal_state
    reverse_path = [current_state[0]]  # 仅记录位置信息
    while current_state != start_state:
        current_state = path[current_state]
        reverse_path.append(current_state[0])
    reverse_path.reverse()
    return reverse_path


# ==================== This is for A* ====================
# use manhattan distance as our first heuristic function
def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def euclidean_distance(a, b):
    return ( (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

def astar(maze):
    start = maze.getStart()
    goal = maze.getObjectives()[0]
    
    frontier = []
    heapq.heappush(frontier, (0, start))  # (priority, node)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0
    
    while not len(frontier) == 0:
        current = heapq.heappop(frontier)[1]
        
        if current == goal:
            break
        
        for next in maze.getNeighbors(*current):
            new_cost = cost_so_far[current] + 1  # 假设所有步骤的成本都是1
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + manhattan_distance(goal, next)
                heapq.heappush(frontier, (priority, next))
                came_from[next] = current
                
    return reconstruct_path(came_from, start, goal)

def reconstruct_path(came_from, start, goal):
    current = goal
    path = []
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)  # optional
    path.reverse()  # optional
    return path


# -------------------------------------------- PART II --------------------------------------------
"""pass all test data"""
def astar_corner(maze):
    start = maze.getStart()
    corners = maze.getObjectives()
    start_state = (start, tuple([start == corner for corner in corners]))
    
    frontier = []
    heapq.heappush(frontier, (0, start_state))
    came_from = {}
    cost_so_far = {}
    came_from[start_state] = None
    cost_so_far[start_state] = 0

    while frontier:
        current_state = heapq.heappop(frontier)[1]
        
        if all(current_state[1]):
            break
        
        for next_position in maze.getNeighbors(*current_state[0]):
            next_visited_corners = update_visited_corners(next_position, current_state[1], corners)
            next_state = (next_position, next_visited_corners)
            new_cost = cost_so_far[current_state] + 1
            
            if next_state not in cost_so_far or new_cost < cost_so_far[next_state]:
                cost_so_far[next_state] = new_cost
                priority = new_cost + heuristic_corner(next_position, next_visited_corners, corners)
                heapq.heappush(frontier, (priority, next_state))
                came_from[next_state] = current_state
    
    return reconstruct_path_corner(came_from, start_state, (current_state[0], current_state[1]))


def update_visited_corners(position, visited_corners, corners):
    new_visited_corners = list(visited_corners)
    for i, corner in enumerate(corners):
        if position == corner:
            new_visited_corners[i] = True
    return tuple(new_visited_corners)


def heuristic_corner(position, visited_corners, corners):
    # 更新启发式函数，以考虑到所有未访问角落的距离
    unvisited_corners = [corners[i] for i, visited in enumerate(visited_corners) if not visited]
    if not unvisited_corners:
        return 0
    # 可以选择使用更复杂的启发式，例如计算到每个未访问角落的距离的和
    # 这里为了简单，我们只计算到最近角落的距离
    return min(manhattan_distance(position, corner) for corner in unvisited_corners)


def reconstruct_path_corner(came_from, start, goal):
    current = goal
    path = []
    while current != start:
        path.append(current[0])  # Extract position from state
        current = came_from[current]
    path.append(start[0])  # Start position
    path.reverse()
    return path


# python3 hw1.py maps/corner/tinyCorners.txt --method astar_corner

# -------------------------------------------- PART III --------------------------------------------
# python3 hw1.py maps/multi/tinySearch.txt --method astar_multi
from functools import lru_cache

@lru_cache(maxsize=None)
def cached_mst_cost(objectives):
    return prim_mst_cost(objectives)

def heuristic(position, objectives):
    # Use a cached version of Prim's algorithm to calculate MST cost of remaining objectives
    # plus the distance to the closest objective.
    # This assumes that 'objectives' can be converted into a hashable type for caching.
    if not objectives:
        return 0
    objectives_tuple = tuple(objectives)  # Ensure it is hashable for caching.
    return cached_mst_cost(objectives_tuple) + min(manhattan_distance(position, obj) for obj in objectives)
    
def astar_multi(maze):
    start = maze.getStart()
    objectives = tuple(maze.getObjectives())
    start_state = (start, objectives)

    frontier = []
    heapq.heappush(frontier, (0, start_state))
    came_from = {start_state: None}
    cost_so_far = {start_state: 0}

    while frontier:
        _, current_state = heapq.heappop(frontier)
        current_position, current_objectives = current_state

        if not current_objectives:  # All objectives have been reached
            break

        for next_position in maze.getNeighbors(*current_position):
            # Directly construct the new state without intermediate variables if possible
            new_cost = cost_so_far[current_state] + 1  # Uniform cost assumed
            new_objectives = tuple(obj for obj in current_objectives if obj != next_position)

            next_state = (next_position, new_objectives)
            if next_state not in cost_so_far or new_cost < cost_so_far[next_state]:
                cost_so_far[next_state] = new_cost
                priority = new_cost + heuristic(next_position, new_objectives)
                heapq.heappush(frontier, (priority, next_state))
                came_from[next_state] = current_state

    return reconstruct_path_multi(came_from, start_state, (current_position, ()))


def reconstruct_path_multi(came_from, start, goal):
    current = goal
    path = []
    while current != start:
        path.append(current[0])
        current = came_from[current]
    path.append(start[0])
    path.reverse()
    return path

def prim_mst_cost(objectives):
    if not objectives:
        return 0

    mst_cost = 0
    connected = set([objectives[0]])
    # Edge candidates: Map from vertex to its closest edge (distance, vertex)
    edge_candidates = {obj: (manhattan_distance(objectives[0], obj), objectives[0]) for obj in objectives if obj != objectives[0]}
    heap = [(dist, start, end) for end, (dist, start) in edge_candidates.items()]
    heapq.heapify(heap)

    while len(connected) < len(objectives):
        cost, _, next_vertex = heapq.heappop(heap)
        if next_vertex in connected:
            continue
        mst_cost += cost
        connected.add(next_vertex)
        for obj in objectives:
            if obj not in connected:
                new_dist = manhattan_distance(next_vertex, obj)
                if new_dist < edge_candidates.get(obj, (float('inf'),))[0]:
                    edge_candidates[obj] = (new_dist, next_vertex)
                    heapq.heappush(heap, (new_dist, next_vertex, obj))

    return mst_cost

# -------------------------------------------- PART IV --------------------------------------------
'''pass all the test data'''
def greedy_best_first(maze, start, goal):
    # Open list to store the nodes to be checked, initialized with the start node
    open_list = [(manhattan_distance(start, goal), start)]
    # Dictionary to store the path taken to reach each visited node
    came_from = {start: None}

    while open_list:
        # Sort the open list to get the node with the lowest heuristic score
        open_list.sort(key=lambda x: x[0])
        current_distance, current_node = open_list.pop(0)

        # If the goal is reached, construct the path back to the start
        if current_node == goal:
            path = []
            while current_node:
                path.append(current_node)
                current_node = came_from[current_node]
            return path[::-1]  # Return reversed path

        # Get neighbors and add them to open list if they haven't been visited yet
        for neighbor in maze.getNeighbors(*current_node):
            if neighbor not in came_from:
                open_list.append((manhattan_distance(neighbor, goal), neighbor))
                came_from[neighbor] = current_node

    return None  # If there is no path to the goal

def fast(maze):
    start = maze.getStart()
    objectives = maze.getObjectives()
    path = []
    current = start

    while objectives:
        # Find the closest dot
        closest_dot = min(objectives, key=lambda dot: manhattan_distance(current, dot))
        
        # Find path to the closest dot using Greedy Best-First Search
        path_to_next_dot = greedy_best_first(maze, current, closest_dot)
        if path_to_next_dot is None:
            raise ValueError("No path to the objective was found.")
        
        # Skip the first node (current position) when extending the path
        path.extend(path_to_next_dot[1:])
        
        # Update the current position and remove the reached dot from the objectives
        current = closest_dot
        objectives.remove(closest_dot)

    return path



# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018
# Modified by Shang-Tse Chen (stchen@csie.ntu.edu.tw) on 03/03/2022

"""
This is the main entry point for HW1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)



# def bfs(maze):
#     """
#     Runs BFS for part 1 of the assignment.
#     @param maze: The maze to execute the search on.
#     @return path: a list of tuples containing the coordinates of each state in the computed path
#     """
#     start = maze.getStart()
#     goal = maze.getObjectives()[0]  # Assuming there's only one goal
    
#     queue = deque([start])
#     visited = set([start])
#     path = {} # Construct a dict to remember the path -> for reverse lookups.

#     while queue:
#         # since it's deque -> popleft()
#         current = queue.popleft()

#         # if == goal -> reverse the path
#         if current == goal:
#             return reconstruct_path_bfs(path, start, goal)
        
#         for neighbor in maze.getNeighbors(*current):  # *current unpacks the tuple
#             if neighbor not in visited:
#                 visited.add(neighbor)
#                 queue.append(neighbor) # add all the neighbors -> FIFO -> same layers would run first
#                 path[neighbor] = current

#     return []  # Return an empty list if no path is found


# def reconstruct_path_bfs(path, start, goal):
#     reverse_path = [goal]
#     while goal != start:
#         goal = path[goal]
#         reverse_path.append(goal)
#     return list(reversed(reverse_path)) # return the path 


# def astar_multi(maze):
#     """
#     Runs A* for part 3 of the assignment in the case where there are
#     multiple objectives.

#     @param maze: The maze to execute the search on.
#     @return path: a list of tuples containing the coordinates of each state in the computed path
#     """

#     def heuristic(position, objectives):
#         # Use Prim's algorithm to calculate MST cost of remaining objectives
#         # plus the distance to the closest objective
#         if not objectives:
#             return 0
#         return prim_mst_cost(objectives) + min(manhattan_distance(position, obj) for obj in objectives)

#     start = maze.getStart()
#     objectives = tuple(maze.getObjectives())
#     start_state = (start, objectives)

#     frontier = []
#     heapq.heappush(frontier, (0, start_state))
#     came_from = {start_state: None}
#     cost_so_far = {start_state: 0}

#     while frontier:
#         current_cost, current_state = heapq.heappop(frontier)
#         current_position, current_objectives = current_state

#         if not current_objectives:  # All objectives have been reached
#             break

#         for next_position in maze.getNeighbors(*current_position):
#             new_objectives = tuple(obj for obj in current_objectives if obj != next_position)
#             next_state = (next_position, new_objectives)
#             new_cost = cost_so_far[current_state] + 1  # Assume uniform cost

#             if next_state not in cost_so_far or new_cost < cost_so_far[next_state]:
#                 cost_so_far[next_state] = new_cost
#                 priority = new_cost + heuristic(next_position, new_objectives)
#                 heapq.heappush(frontier, (priority, next_state))
#                 came_from[next_state] = current_state

#     return reconstruct_path(came_from, start_state, (current_position, ()))

# def reconstruct_path(came_from, start, goal):
#     current = goal
#     path = []
#     while current != start:
#         path.append(current[0])
#         current = came_from[current]
#     path.append(start[0])
#     path.reverse()
#     return path


# def prim_mst_cost(objectives):
#     """Calculates the total weight of the MST using Prim's algorithm."""
#     if not objectives:
#         return 0

#     # Initialize the MST and its total cost
#     mst_cost = 0
#     vertices = list(objectives)
#     connected = {vertices[0]}  # Start with the first vertex in a set for quick lookup
#     edges = []

#     # Populate initial edges from the first connected vertex to all other vertices
#     for vertex in vertices[1:]:
#         heapq.heappush(edges, (manhattan_distance(vertices[0], vertex), vertex))

#     while len(connected) < len(vertices):
#         # Pop the shortest edge
#         cost, next_vertex = heapq.heappop(edges)
#         if next_vertex not in connected:
#             # Add the vertex to the MST
#             connected.add(next_vertex)
#             mst_cost += cost
#             # Add new edges from this vertex to all others not yet in the MST
#             for v in vertices:
#                 if v not in connected:
#                     heapq.heappush(edges, (manhattan_distance(next_vertex, v), v))

#     return mst_cost
