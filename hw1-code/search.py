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
def astar_multi(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    
    """did not pass the mediumSearch.txt yet"""

    start = maze.getStart()
    food = maze.getObjectives()  # 这应返回所有食物位置的列表或集合
    start_state = (start, tuple(sorted(food)))

    frontier = []
    heapq.heappush(frontier, (0, start_state))
    came_from = {}
    cost_so_far = {}
    came_from[start_state] = None
    cost_so_far[start_state] = 0

    while frontier:
        current = heapq.heappop(frontier)[1]

        if not current[1]:  # 检查是否所有食物都已被吃掉
            break

        for next_position in maze.getNeighbors(*current[0]):
            next_food = update_food(next_position, current[1])
            next_state = (next_position, next_food)
            new_cost = cost_so_far[current] + 1

            if next_state not in cost_so_far or new_cost < cost_so_far[next_state]:
                cost_so_far[next_state] = new_cost
                priority = new_cost + heuristic_food(next_position, next_food)
                heapq.heappush(frontier, (priority, next_state))
                came_from[next_state] = current

    return reconstruct_path_food(came_from, start_state, current)

def update_food(position, food):
    # 如果当前位置有食物，则从剩余食物中移除该位置
    return tuple(f for f in food if f != position)

def heuristic_food(position, food):
    # 估算从当前位置到剩余所有食物的距离
    # 这里简化为到最近食物的曼哈顿距离
    if not food:
        return 0
    return min(manhattan_distance(position, f) for f in food) # manhattan distance is faster
    # return min(euclidean_distance(position, f) for f in food)

def reconstruct_path_food(came_from, start, goal):
    # 重建路径
    current = goal
    path = []
    while current != start:
        path.append(current[0])
        current = came_from[current]
    path.append(start[0])  # 添加起始位置
    path.reverse()  # 反转路径
    return path

# python3 hw1.py maps/multi/tinySearch.txt --method astar_multi

# -------------------------------------------- PART IV --------------------------------------------
def fast(maze):
    """
    Runs suboptimal search algorithm for part 4.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    """did not pass the bigSearch"""

    # 初始化
    start = maze.getStart()
    food = maze.getObjectives()  # 获取所有食物的位置
    path = []  # 存储最终的路径

    while food:
        closest_food = None
        shortest_path = None
        min_distance = float("inf")

        # 找到最近的食物点
        for f in food:
            temp_path = bfs_fast(maze, start, f)
            if temp_path and len(temp_path) < min_distance:
                closest_food = f
                shortest_path = temp_path
                min_distance = len(temp_path)

        if shortest_path is None:
            break  # 如果没有路径可以到达任何食物点，则退出

        # 更新路径和起点
        path += shortest_path
        start = closest_food
        food.remove(closest_food)  # 移除已经到达的食物点

    return path

def bfs_fast(maze, start, goal):
    # 广度优先搜索找到从start到goal的路径
    queue = [(start, [])]  # 元素格式：(当前位置, 到达该位置的路径)
    visited = set()

    while queue:
        current_position, path = queue.pop(0)

        if current_position in visited:
            continue

        visited.add(current_position)

        if current_position == goal:
            return path

        for next_position in maze.getNeighbors(current_position[0], current_position[1]):
            if next_position not in visited:
                queue.append((next_position, path + [next_position]))

    return None  # 如果没有路径，则返回None



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