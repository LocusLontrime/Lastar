import heapq as hq
import time
import math
from enum import Enum
from functools import reduce

import numpy as np
# graphics:
import arcade
import arcade.gui
# data
import shelve
# queues:
from collections import deque
# abstract classes:
from abc import ABC, abstractmethod
# windows:
import pyglet
from typing_extensions import override

# screen sizes:
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1050


class Lastar(arcade.Window):  # 36 366 98 989 LL
    # initial directions prority for all algorithms:
    walk = [(1, 0), (0, -1), (-1, 0), (0, 1)]

    def __init__(self, width: int, height: int):
        super().__init__(width, height)
        arcade.set_background_color(arcade.color.DUTCH_WHITE)
        self.set_update_rate(1 / 60)
        # scaling:  TODO: add AI to calculate the sizes for every resolution possible:
        self.scale = 0
        self.scale_names = {0: 10, 1: 15, 2: 22, 3: 33, 4: 45, 5: 66, 6: 90,
                            7: 110}  # {0: 5, 1: 10, 2: 15, 3: 22, 4: 33, 5: 45, 6: 66, 7: 90, 8: 110, 9: 165, 10: 198}  # factors of 990 num
        # initial data and info:
        self.line_width = None
        self.tiles_q = None
        self.Y, self.X = SCREEN_HEIGHT - 26, SCREEN_WIDTH - 250
        self.tile_size, self.hor_tiles_q = self.get_pars()  # calculating
        # grid of nodes:
        self.grid = [[Node(j, i, 1, NodeType.EMPTY) for i in range(self.hor_tiles_q)] for j in range(self.tiles_q)]
        # modes, flags and pars needed for visualization:
        self.mode = 0  # 0 for building the walls and erasing them afterwards, 1 for a start and end nodes choosing and 2 for info getting for every node
        self.mode_names = {0: 'BUILDING/ERASING', 1: 'START&END_NODES_CHOOSING', 2: 'INFO_GETTING'}
        self.building_walls_flag = False
        self.build_or_erase = True  # True for building and False for erasing
        self.node_chosen = None
        self.path = None
        self.path_index = 0
        self.triangle_shape_list = arcade.ShapeElementList()  # <<-- for more comprehensive path visualization
        self.arrow_shape_list = arcade.ShapeElementList()  # <<-- for more comprehensive algorithm's visualization
        # important general settings:
        self.arrows_indices = []
        self.walk_index = 0
        self.arrows_vertices = None
        self.arrow_length, self.arrow_height = 36, 18  # directions priority
        self.choosing_arrows = True
        self.inter_types_arrows = [InterType.NONE for _ in range(4)]
        self.inter_type_reset_square = InterType.NONE
        # a_star important pars:
        self.start_node = None
        self.end_node = None
        self.heuristic = 0
        self.heuristic_names = {0: 'MANHATTAN', 1: 'EUCLIDIAN', 2: 'MAX_DELTA', 3: 'DIJKSTRA'}
        self.tiebreaker = None
        self.tiebreaker_names = {0: 'VECTOR_CROSS', 1: 'COORDINATES'}
        self.greedy_ind = None  # is algorithm greedy?
        self.nodes_to_be_visited = []

        self.nodes_visited = {}
        self.iterations = 0
        self.time_elapsed_ms = 0
        # interactive a_star pars:
        self.curr_node_dict = {}
        self.max_times_visited_dict = {0: 0}
        self.neighs_added_to_heap_dict = {}
        # lee_wave_spreading important pars:
        # Levi Gin area!
        # interactive pars (game logic):
        self.interactive_ind = None
        self.in_interaction = False
        self.guide_arrows_ind = None
        self.cycle_breaker_right = False
        self.cycle_breaker_left = False
        self.ticks_before = 0
        self.f_flag = False
        self.loading = False
        self.loading_ind = 0
        # lockers:
        self.in_interaction_mode_lock = False
        self.greedy_flag_lock = False
        self.heuristic_lock = False
        self.tiebreakers_lock = False
        self.scale_lock = False
        # self.arrows_lock = False  # is not needed, since it is an options that can be turned on/off during the interactive algo phase
        self.bfs_dfs_lock = False
        self.dict_something = {}
        # walls building/erasing dicts:
        self.walls_built_erased = [([], True)]  # TODO: swap to DICT!!!
        self.walls_index = 0
        self.walls = set()  # all walls located on the map at the time being
        # blocks:
        self.node_sprite_list = arcade.SpriteList()
        # grid_lines:
        self.grid_line_shapes = arcade.ShapeElementList()
        # UI ???:
        self.a_star_view = arcade.View()
        self.wave_lee_view = arcade.View()
        self.game_choosing_view = arcade.View()
        # GEAR WHEEL:
        self.cx, self.cy = 0, 0
        self.hole_rx, self.hole_ry = 0, 0
        self.upper_vertices_list = []
        self.multiplier = 1.5
        # BFS/DFS:
        self.bfs_dfs_cx, self.bfs_dfs_cy = 0, 0
        self.bfs_dfs_size = 0
        # A STAR LABEL:
        self.a_outer_points = []
        # RIGHT MENU LOGIC:
        self.inter_types = [InterType.NONE for _ in range(4)]  # <<-- GEAR WHEEL, BFS/DFS, A_STAR and WAVE LEE <<--
        self.incrementers = [0 for _ in range(4)]  # <<-- GEAR WHEEL, BFS/DFS, A_STAR and WAVE LEE <<--
        # WAVE_LEE (LEVI GIN):
        self.front_wave_lee = []
        self.next_wave_lee = []
        self.iterations_wave_lee = 0  # TODO: to be deleted!!!
        self.fronts_dict = {}
        # BFS/DFS:
        self.bfs_dfs_ind = 0
        self.queue = deque()
        # imported classes' objects:
        # self.cleaner = Cleaner(self)
        self.renderer = DrawLib()
        # ICONS:
        self.play_button = PlayButton(1785 + 6, 50, 32, 2)

        self.step_button_right = StepButton(1785 + 6 + 50, 50, 24, 16, 2)
        self.step_button_left = StepButton(1785 + 6 - 50, 50, 24, 16, 2, False)

        self.eraser = Eraser(1785 + 6, 125, 16, 32, 8, 2)

        self.undo = Undo(1785 + 6 - 75, 125, 24, 8, 12, 2)
        self.redo = Undo(1785 + 6 + 75, 125, 24, 8, 12, 2, True)
        # ALGOS:
        self.astar = Astar(self)
        self.wave_lee = WaveLee(self)
        self.bfs_dfs = BfsDfs(self)

    # INITIALIZATION AUX:
    # calculating grid visualization pars for vertical tiles number given:
    def get_pars(self):
        self.Y, self.X = SCREEN_HEIGHT - 60, SCREEN_WIDTH - 250
        self.tiles_q = self.scale_names[self.scale]
        self.line_width = int(math.sqrt(max(self.scale_names.values()) / self.tiles_q))
        tile_size = self.Y // self.tiles_q
        hor_tiles_q = self.X // tile_size
        self.Y, self.X = self.tiles_q * tile_size, hor_tiles_q * tile_size
        return tile_size, hor_tiles_q

    def get_hor_tiles(self, i):
        return (SCREEN_WIDTH - 250) // ((SCREEN_HEIGHT - 30) // self.scale_names[i])

    # A_STAR INTERACTIVE METHODS:
    # preparation for interactive a_star:
    def a_star_preparation(self):
        # heap:
        self.nodes_to_be_visited = [self.start_node]
        hq.heapify(self.nodes_to_be_visited)
        # heur/cost:
        self.start_node.g = 0
        # transmitting the greedy flag to the Node class: TODO: fix this strange doing <<--
        Node.IS_GREEDY = False if self.greedy_ind is None else True
        # important pars and dicts:
        self.iterations = 0
        self.neighs_added_to_heap_dict = {0: [self.start_node]}
        self.curr_node_dict = {0: None}
        self.max_times_visited_dict = {0: 0}
        # path nullifying:
        self.path = None
        # lockers locating:
        self.in_interaction_mode_lock = True
        self.greedy_flag_lock = True
        self.heuristic_lock = True
        self.tiebreakers_lock = True
        self.scale_lock = True
        self.bfs_dfs_lock = True
        # arrows list renewal:
        self.arrow_shape_list = arcade.ShapeElementList()

    # path-recovering process for interactive a_star:
    def recover_path(self):
        # start point of path restoration (here we begin from the end node of the shortest path found):
        node = self.end_node
        shortest_path = []
        # path restoring (here we get the reversed path):
        while node.previously_visited_node:
            shortest_path.append(node)
            node = node.previously_visited_node
        shortest_path.append(self.start_node)
        # returns the result:
        return shortest_path

    # a step forward in path restoring phase visualisation (interactive a_star mode):
    def path_up(self):
        if (path_node := self.path[self.path_index]).type not in [NodeType.START_NODE, NodeType.END_NODE]:
            path_node.type = NodeType.PATH_NODE
            path_node.update_sprite_colour()
        # arrows:
        p = -self.path[self.path_index + 1].x + self.path[self.path_index].x, \
            -self.path[self.path_index + 1].y + self.path[self.path_index].y
        p1, p2, p3 = self.get_triangle(self.path[self.path_index + 1], p)
        triangle_shape = arcade.create_triangles_filled_with_colors(
            [p1, p2, p3],
            [arcade.color.WHITE, arcade.color.RED, arcade.color.RED])
        self.triangle_shape_list.append(triangle_shape)
        # line arrows removing:
        node = self.path[self.path_index + 1]
        # if self.inter_types[2] == InterType.PRESSED:
        if node not in [self.start_node, self.end_node]:
            node.remove_arrow_from_shape_list(self)

    # a step back for path restoring phase visualisation (interactive a_star mode):
    def path_down(self):
        if (path_node := self.path[self.path_index]).type not in [NodeType.START_NODE, NodeType.END_NODE]:
            path_node.type = NodeType.VISITED_NODE
            path_node.update_sprite_colour()
        # arrows:
        self.triangle_shape_list.remove(self.triangle_shape_list[self.path_index - 1])
        # line arrows restoring:
        # if self.inter_types[2] == InterType.PRESSED:
        if path_node not in [self.start_node, self.end_node]:
            path_node.append_arrow(self)

    # a step forward in path-finding phase visualisation (interactive a_star mode)
    def a_star_step_up(self):
        if self.iterations == 0:
            self.nodes_to_be_visited = [self.start_node]
        self.neighs_added_to_heap_dict[self.iterations + 1] = []  # memoization
        # popping out the most priority node for a_star from the heap:
        self.curr_node_dict[self.iterations + 1] = hq.heappop(self.nodes_to_be_visited)  # + memoization
        curr_node = self.curr_node_dict[self.iterations + 1]
        if self.iterations > 0 and curr_node != self.end_node:
            curr_node.type = NodeType.CURRENT_NODE
            curr_node.update_sprite_colour()
        curr_node.times_visited += 1
        if self.iterations > 1:
            if (prev_node := self.curr_node_dict[self.iterations]).type not in [NodeType.END_NODE,
                                                                                NodeType.TWICE_VISITED]:
                prev_node.type = NodeType.VISITED_NODE
                prev_node.update_sprite_colour()
        self.max_times_visited_dict[self.iterations + 1] = max(self.max_times_visited_dict[self.iterations],
                                                               # memoization
                                                               curr_node.times_visited)
        # memoization for correct movement back:
        if curr_node in self.nodes_visited.keys():
            self.nodes_visited[curr_node] += 1
            curr_node.type = NodeType.TWICE_VISITED
            curr_node.update_sprite_colour()
        else:
            self.nodes_visited[curr_node] = 1
        # base case of finding the shortest path:
        if curr_node == self.end_node:
            self.path = self.recover_path()
        # next step:
        # we can search for neighs on the fly or use precalculated sets (outdated):
        for neigh in curr_node.get_neighs(self, [NodeType.WALL]):  # getting all the neighs 'on the fly;
            if neigh.g > curr_node.g + neigh.val:
                # memoization for further 'undoing':
                self.neighs_added_to_heap_dict[self.iterations + 1].append(
                    neigh.smart_copy(
                        [
                            'g',
                            'h',
                            'tiebreaker',
                            'type',
                            'previously_visited_node'
                        ]
                    )
                )
                # cost and heuristic computing:
                neigh.g = curr_node.g + neigh.val
                neigh.h = neigh.heuristics[self.heuristic](neigh, self.end_node)
                # tie-breaking:
                if self.tiebreaker is not None:
                    neigh.tiebreaker = self.start_node.tiebreakers[self.tiebreaker](
                        self.start_node, self.end_node, neigh)
                # previous visited node memoization for further path-recovering process:
                neigh.previously_visited_node = curr_node
                if neigh.type not in [NodeType.START_NODE, NodeType.END_NODE]:  # neigh not in self.nodes_visited and
                    neigh.type = NodeType.NEIGH
                    neigh.update_sprite_colour()
                    arrow = self.renderer.create_line_arrow(neigh, (neigh.x - curr_node.x, neigh.y - curr_node.y), self)
                    # here the arrow rotates (re-estimating of neigh g-cost):
                    if neigh.arrow_shape is not None:
                        neigh.remove_arrow(self)
                    neigh.arrow_shape = arrow
                    neigh.append_arrow(self)
                # adding all the valid neighs to the priority heap:
                hq.heappush(self.nodes_to_be_visited, neigh)
        # incrementation:
        self.iterations += 1

    # a step back in path-finding phase visualisation (interactive a_star mode)
    def a_star_step_down(self):
        # getting the previous current node from memo table:
        curr_node = self.curr_node_dict[self.iterations]
        if self.iterations > 1:  # stop condition for preventing the border case errors
            # times visited counter and colour 'backtracking':
            curr_node.times_visited -= 1
            if curr_node.times_visited == 0:
                if curr_node.type != NodeType.END_NODE:
                    curr_node.type = NodeType.NEIGH
                    curr_node.update_sprite_colour()
            elif curr_node.times_visited == 1:
                curr_node.type = NodeType.VISITED_NODE
                curr_node.update_sprite_colour()
            else:
                curr_node.type = NodeType.TWICE_VISITED
                curr_node.update_sprite_colour()
            if self.iterations > 2:
                self.curr_node_dict[self.iterations - 1].type = NodeType.CURRENT_NODE
                self.curr_node_dict[self.iterations - 1].update_sprite_colour()
        if self.iterations > 0:
            # removing the current node from nodes visited:
            if self.nodes_visited[curr_node] > 1:
                self.nodes_visited[curr_node] -= 1
            else:
                self.nodes_visited.pop(curr_node)
            # removing the neighs added from the heap:
            for neigh in self.neighs_added_to_heap_dict[self.iterations]:
                y, x = neigh.y, neigh.x
                node = self.grid[y][x]
                Node.aux_equal_flag = True
                self.remove_from_heapq(self.nodes_to_be_visited, self.nodes_to_be_visited.index(node))
                Node.aux_equal_flag = False
                node.smart_restore(
                    neigh,
                    [
                        'g',
                        'h',
                        'tiebreaker',
                        'type',
                        'previously_visited_node'
                    ]
                )
                if node.type not in [NodeType.START_NODE, NodeType.END_NODE]:
                    if node.arrow_shape is not None:
                        node.remove_arrow(self)
                if node.type == NodeType.NEIGH:
                    # here the arrow rotates backwards:
                    arrow = self.renderer.create_line_arrow(node, (
                        node.x - node.previously_visited_node.x, node.y - node.previously_visited_node.y), self)
                    node.arrow_shape = arrow
                    node.append_arrow(self)
            # adding current node (popped out at the current iteration) to the heap:
            hq.heappush(self.nodes_to_be_visited, curr_node)
            # iteration steps back:
            self.iterations -= 1

    # from stackoverflow, removing the element from the heap, keeping the heap invariant:
    @staticmethod
    def remove_from_heapq(heap, ind: int):
        heap[ind] = heap[-1]
        heap.pop()
        if ind < len(heap):
            # as far as it is known, possible to copy the source code from the heapq module... but how to do that?..
            Lastar.siftup(heap, ind)
            Lastar.siftdown(heap, 0, ind)

    # source code from: https://github.com/python/cpython/blob/main/Lib/heapq.py
    @staticmethod
    def siftdown(heap, start_pos, pos):
        new_item = heap[pos]
        # Follow the path to the root, moving parents down until finding a place
        # new item fits.
        while pos > start_pos:
            parent_pos = (pos - 1) >> 1
            parent = heap[parent_pos]
            if new_item < parent:
                heap[pos] = parent
                pos = parent_pos
                continue
            break
        heap[pos] = new_item

    # source code from: https://github.com/python/cpython/blob/main/Lib/heapq.py
    @staticmethod
    def siftup(heap, pos):
        end_pos = len(heap)
        start_pos = pos
        new_item = heap[pos]
        # Bubble up the smaller child until hitting a leaf.
        child_pos = 2 * pos + 1  # leftmost child position
        while child_pos < end_pos:
            # Set child pos to index of smaller child.
            right_pos = child_pos + 1
            if right_pos < end_pos and not heap[child_pos] < heap[right_pos]:
                child_pos = right_pos
            # Move the smaller child up.
            heap[pos] = heap[child_pos]
            pos = child_pos
            child_pos = 2 * pos + 1
        # The leaf at pos is empty now. Put new item there, and bubble it up
        # to its final resting place (by sifting its parents down).
        heap[pos] = new_item
        Lastar.siftdown(heap, start_pos, pos)

    # WAVE LEE LEVI GIN WORK (APPROVED):
    def wave_lee_preparation(self):
        # starting pars:
        self.front_wave_lee = []
        self.next_wave_lee = [self.start_node]
        self.start_node.val = 1  # ??? what about clearing?
        # variable ones:
        self.iterations_wave_lee = 0
        self.fronts_dict = {}

    def wave_lee_step_up(self):
        self.iterations_wave_lee += 1
        self.front_wave_lee = self.next_wave_lee[:]
        self.fronts_dict[self.iterations_wave_lee] = self.front_wave_lee
        self.next_wave_lee = []
        for curr_node in self.front_wave_lee:
            curr_node.val = self.iterations_wave_lee
            if curr_node not in [self.end_node, self.start_node]:
                curr_node.type = NodeType.VISITED_NODE
                curr_node.update_sprite_colour()
            if curr_node == self.end_node:
                self.end_node.val = self.iterations_wave_lee
                self.path = self.recover_path()
                break
            for neigh in curr_node.get_neighs(self, [NodeType.START_NODE, NodeType.WALL, NodeType.VISITED_NODE]):
                if neigh.val == 1:  # it is equivalent to if neigh.type == NodeType.EMPTY
                    if neigh not in self.next_wave_lee:
                        if neigh != self.end_node:
                            neigh.type = NodeType.NEIGH
                            neigh.update_sprite_colour()
                            arrow = self.renderer.create_line_arrow(neigh,
                                                                    (neigh.x - curr_node.x, neigh.y - curr_node.y),
                                                                    self)
                            neigh.arrow_shape = arrow
                            neigh.append_arrow(self)
                        self.next_wave_lee.append(neigh)
                        neigh.previously_visited_node = curr_node

    def wave_lee_step_down(self):
        # possibility check of wave_lee's stepping back:
        if self.iterations_wave_lee > 0:
            # decrementation:
            self.iterations_wave_lee -= 1
            # neighs have become EMPTY ones:
            for neigh in self.next_wave_lee:
                if neigh not in [self.start_node, self.end_node]:
                    neigh.type = NodeType.EMPTY
                    neigh.update_sprite_colour()
                    neigh.remove_arrow(self)
            if self.iterations_wave_lee != 0:
                # the front nodes have become NEIGHS:
                for node in self.front_wave_lee:
                    if node != self.start_node:
                        if node != self.end_node:
                            node.type = NodeType.NEIGH
                        node.update_sprite_colour()
                        node.val = 1
                # current and next fronts stepping back:
                self.next_wave_lee = self.front_wave_lee[:]
                self.front_wave_lee = self.fronts_dict[self.iterations_wave_lee]
            else:
                # the starting point:
                self.next_wave_lee = [self.start_node]
                self.front_wave_lee = []

    # WITCHDOCTOR'S BLOCK (IN PROCESS):

    # BFS:
    def bfs_preparation(self):
        self.queue = deque()
        self.queue.append(self.start_node)
        self.iterations = 0
        # dicts:
        self.curr_node_dict = {0: None}

    def bfs_step_up(self):
        # one bfs step up:
        self.iterations += 0
        curr_node = self.queue.pop()
        self.curr_node_dict[self.iterations + 1] = curr_node
        if self.iterations > 0:
            if curr_node.type != NodeType.END_NODE:
                curr_node.type = NodeType.CURRENT_NODE
                curr_node.update_sprite_colour()
        if self.iterations > 1:
            self.curr_node_dict[self.iterations].type = NodeType.VISITED_NODE
            self.curr_node_dict[self.iterations].update_sprite_colour()
        curr_node.times_visited += 1
        if curr_node == self.end_node:
            self.path = self.recover_path()
        self.neighs_added_to_heap_dict[self.iterations + 1] = set()
        for neigh in curr_node.get_neighs(self, [NodeType.START_NODE, NodeType.VISITED_NODE, NodeType.WALL] + (
                [NodeType.NEIGH] if self.bfs_dfs_ind == 0 else [])):
            if neigh.type != NodeType.END_NODE:
                # at first memoization for further 'undoing':
                self.neighs_added_to_heap_dict[self.iterations + 1].add(neigh.aux_copy())
                # then changing neigh's pars:
                neigh.type = NodeType.NEIGH
                neigh.update_sprite_colour()
                arrow = self.renderer.create_line_arrow(neigh, (neigh.x - curr_node.x, neigh.y - curr_node.y), self)
                if neigh.arrow_shape is not None:
                    neigh.remove_arrow(self)
                neigh.arrow_shape = arrow
                neigh.append_arrow(self)
            neigh.previously_visited_node = curr_node
            # BFS:
            if self.bfs_dfs_ind == 0:
                self.queue.appendleft(neigh)
            # DFS:
            else:
                self.queue.append(neigh)
        self.iterations += 1

    def bfs_step_down(self):
        if self.iterations > 0:
            # now the neighs of current node should become EMPTY ones:
            for neigh in self.neighs_added_to_heap_dict[self.iterations]:
                # TODO: neigh type restoring needed!!!
                y, x = neigh.y, neigh.x
                node = self.grid[y][x]
                node.restore(neigh)
                if node not in [self.start_node, self.end_node]:
                    node.remove_arrow(self)
                if node.type == NodeType.NEIGH:
                    # here the arrow rotates backwards:
                    arrow = self.renderer.create_line_arrow(
                        node,
                        (
                            node.x - self.curr_node_dict[self.iterations].x,
                            node.y - self.curr_node_dict[self.iterations].y
                        ),
                        self
                    )
                    node.arrow_shape = arrow
                    node.append_arrow(self)
                    # deque changing:
                    # BFS:
                    if self.bfs_dfs_ind == 0:
                        self.queue.popleft()
                    # DFS:
                    else:
                        self.queue.pop()
            # current node has become the NEIGH:
            curr_node = self.curr_node_dict[self.iterations]
            if curr_node not in [self.start_node, self.end_node]:
                curr_node.type = NodeType.NEIGH
                curr_node.update_sprite_colour()
            # adding back to the deque:
            # BFS & DFS:
            self.queue.append(curr_node)
            if self.iterations > 1:
                # previous step current node has become the current step current node:
                prev_node = self.curr_node_dict[self.iterations - 1]
                if prev_node not in [self.start_node, self.end_node]:
                    prev_node.type = NodeType.CURRENT_NODE
                    prev_node.update_sprite_colour()
            # step back:
            self.iterations -= 1

    # PRESETS:
    def setup(self):
        # game set up is located below:
        # sprites, shapes and etc...
        # blocks:
        self.get_sprites()
        # grid lines:
        self.make_grid_lines()

        # let us make the new mouse cursor:
        cursor = self.get_system_mouse_cursor(self.CURSOR_HAND)
        self.set_mouse_cursor(cursor)

        # let us change the app icon:
        self.set_icon(pyglet.image.load('C:\\Users\\langr\\PycharmProjects\\Lastar\\366.png'))

        # ICONS:
        self.eraser.setup()

    # shaping shape element list of grid lines:
    def make_grid_lines(self):
        for j in range(self.tiles_q + 1):
            self.grid_line_shapes.append(
                arcade.create_line(5, 5 + self.tile_size * j, 5 + self.X, 5 + self.tile_size * j, arcade.color.BLACK,
                                   self.line_width))

        for i in range(self.hor_tiles_q + 1):
            self.grid_line_shapes.append(
                arcade.create_line(5 + self.tile_size * i, 5, 5 + self.tile_size * i, 5 + self.Y, arcade.color.BLACK,
                                   self.line_width))

    # creates sprites for all the nodes:
    def get_sprites(self):  # batch -->
        for row in self.grid:
            for node in row:
                node.get_solid_colour_sprite(self)

    # DRAWING:
    # the main drawing method, that is called one times per frame:
    def on_draw(self):
        # renders this screen:
        arcade.start_render()
        # image's code:
        # grid:
        self.grid_line_shapes.draw()
        # blocks:
        self.node_sprite_list.draw()
        # arrows:
        if self.guide_arrows_ind is not None:
            if len(self.arrow_shape_list) > 0:
                self.arrow_shape_list.draw()
        # HINTS:
        arcade.Text(
            f'A* iters: {self.iterations}, path length: {len(self.path) if self.path else "No path found"}, nodes visited: {len(self.nodes_visited)}, time elapsed: {self.time_elapsed_ms} ms',
            365, SCREEN_HEIGHT - 35, arcade.color.BROWN, bold=True).draw()
        arcade.Text(f'Mode: {self.mode_names[self.mode]}', 25, SCREEN_HEIGHT - 35, arcade.color.BLACK, bold=True).draw()
        if self.mode == 2:
            if self.node_chosen:
                arcade.Text(
                    f"NODE'S INFO -->> pos: {self.node_chosen.y, self.node_chosen.x}, g: {self.node_chosen.g}, "
                    f"h: {self.node_chosen.h}, f=g+h: {self.node_chosen.g + self.node_chosen.h}, val: {self.node_chosen.val}, t: {self.node_chosen.tiebreaker}, times visited: {self.node_chosen.times_visited}, type: {self.node_chosen.type}",
                    1050, SCREEN_HEIGHT - 35, arcade.color.PURPLE, bold=True).draw()
        # SET-UPS FOR A_STAR:
        if self.inter_types[2] == InterType.PRESSED:
            bot_menu_x, bot_menu_y = SCREEN_WIDTH - 235, SCREEN_HEIGHT - 70 - 120
            # heuristics:
            self.renderer.draw_area(bot_menu_x, bot_menu_y, f'Heuristics', self.heuristic_names,
                                    self.heuristic, self.heuristic_lock)
            bot_menu_y -= 30 + (
                    len(self.heuristic_names) - 1) * self.renderer.get_delta() + 3 * self.renderer.get_sq_size()
            # tiebreakers:
            self.renderer.draw_area(bot_menu_x, bot_menu_y, f'Tiebreakers', self.tiebreaker_names,
                                    self.tiebreaker, self.tiebreakers_lock)
            bot_menu_y -= 30 + (
                    len(self.tiebreaker_names) - 1) * self.renderer.get_delta() + 3 * self.renderer.get_sq_size()
            # greedy flag:
            self.renderer.draw_area(bot_menu_x, bot_menu_y, f'Is greedy', {0: f'GREEDY_FLAG'}, self.greedy_ind,
                                    self.greedy_flag_lock)
            bot_menu_y -= 30 + 3 * self.renderer.get_sq_size()
            # guiding arrows:
            self.renderer.draw_area(bot_menu_x, bot_menu_y, f'Guide arrows', {0: f'ON/OFF'}, self.guide_arrows_ind,
                                    False)
            bot_menu_y -= 30 + 3 * self.renderer.get_sq_size()
            # a_star show mode:
            self.renderer.draw_area(bot_menu_x, bot_menu_y, f'Show mode', {0: f'IS_INTERACTIVE'}, self.interactive_ind,
                                    self.in_interaction_mode_lock)
        # BFS and DFS PARS:
        elif self.inter_types[1] == InterType.PRESSED:
            bot_menu_x, bot_menu_y = SCREEN_WIDTH - 235, SCREEN_HEIGHT - 70 - 120
            # algo choosing (bfs or dfs):
            self.renderer.draw_area(bot_menu_x, bot_menu_y, f'Core', {0: f'BFS', 1: f'DFS'}, self.bfs_dfs_ind,
                                    self.bfs_dfs_lock)
        # SETTINGS:
        elif self.inter_types[0] == InterType.PRESSED:
            bot_menu_x, bot_menu_y = SCREEN_WIDTH - 235, SCREEN_HEIGHT - 70 - 120
            # directions priority arrows (walk setting up):
            self.renderer.draw_arrows_menu(bot_menu_x, bot_menu_y, 36, 18, self)
            bot_menu_y -= 3 * 3 * self.renderer.get_sq_size()
            # scaling:
            self.renderer.draw_area(bot_menu_x, bot_menu_y, f'Sizes in tiles',
                                    {k: f'{v}x{self.get_hor_tiles(k)}' for k, v in self.scale_names.items()},
                                    self.scale, self.scale_lock)
        # NODE CHOSEN (should work for every algo) TODO: should be reworked!!!
        if self.node_chosen:
            arcade.draw_circle_filled(5 + self.node_chosen.x * self.tile_size + self.tile_size / 2,
                                      5 + self.node_chosen.y * self.tile_size + self.tile_size / 2, self.tile_size / 4,
                                      arcade.color.YELLOW)
        # CURRENT PATH NODE ON START OR END NODE (should work for every algo), TODO: should be implemented in a separated method:
        if self.triangle_shape_list:
            self.triangle_shape_list.draw()
        # ICONS OF INTERACTION:
        # GEAR WHEEL:
        self.renderer.draw_gear_wheel(1785 + 6, 1000, 24, 24, 6, game=self)
        # BFS and DFS icons:
        self.renderer.draw_bfs_dfs(1750 - 30 + 6, 1020 - 73 - 25, 54, 2, game=self)
        # A STAR LABEL:
        self.renderer.draw_a_star(1750 + 15 + 6, 1020 - 100 - 25, 22, 53, game=self)  # 36 366 98 989
        # LEE WAVES:
        self.renderer.draw_waves(1750 + 50 + 48 + 10 + 6, 1020 - 73 - 25, 32, 5, game=self)

        # ICONS:
        self.play_button.draw()

        # TESTS:
        self.step_button_right.draw()
        self.step_button_left.draw()

        self.eraser.draw()

        self.undo.draw()
        self.redo.draw()

    # triangle points getting:
    def get_triangle(self, node: 'Node', point: tuple[int, int]):
        scaled_point = point[0] * (self.tile_size // 2 - 2), point[1] * (self.tile_size // 2 - 2)
        deltas = (
            (
                scaled_point[0] - scaled_point[1],
                scaled_point[0] + scaled_point[1]
            ),
            (
                scaled_point[0] + scaled_point[1],
                -scaled_point[0] + scaled_point[1]
            )
        )
        cx, cy = 5 + node.x * self.tile_size + self.tile_size / 2, 5 + node.y * self.tile_size + self.tile_size / 2
        return (cx, cy), (cx + deltas[0][0], cy + deltas[0][1]), (cx + deltas[1][0], cy + deltas[1][1])

    # TODO: DECIDE IF IT IS NEEDED!!!
    # def draw_bfs(self, cx, cy, r, line_w):
    #     arcade.draw_circle_outline(cx, cy, r, arcade.color.BLACK, line_w)
    #     self.draw_line_arrow(cx + r + r / 3, cy, r, arrow_l=r / 4, is_left=False, line_w=line_w)
    #     self.draw_line_arrow(cx - r - r / 3, cy, r, arrow_l=r / 4, is_left=True, line_w=line_w)
    #     self.draw_dashed_line_circle(cx + 3 * r + 2 * r / 3, cy, r, r // 2, line_w)
    #     self.draw_dashed_line_circle(cx - 3 * r - 2 * r / 3, cy, r, r // 2, line_w)
    #
    # def draw_dfs(self):
    #     ...

    # colour gradient:
    @staticmethod
    def linear_gradi(c: tuple[int, int, int], i):
        return c[0] + 3 * i, c[1] - 5 * i, c[2] + i * 5

    # REBUILDING AND CLEARING/ERASING METHODS:
    # rebuilds grid lines and grid of nodes for a new vertical tiles number chosen:
    def rebuild_map(self):
        self.tile_size, self.hor_tiles_q = self.get_pars()
        # grid's renewing:
        self.grid = [[Node(j, i, 1, NodeType.EMPTY) for i in range(self.hor_tiles_q)] for j in range(self.tiles_q)]
        # pars resetting:
        self.aux_clear()
        self.start_node = None
        self.end_node = None
        self.node_chosen = None
        self.node_sprite_list = arcade.SpriteList()
        self.grid_line_shapes = arcade.ShapeElementList()
        self.walls = set()
        self.get_sprites()
        self.make_grid_lines()

    # clears all the nodes except start, end and walls
    def clear_empty_nodes(self):
        # clearing the every empty node:
        for row in self.grid:
            for node in row:
                if node.type not in [NodeType.WALL, NodeType.START_NODE, NodeType.END_NODE]:
                    node.clear()
                elif node.type in [NodeType.START_NODE, NodeType.END_NODE]:
                    node.heur_clear()
        # clearing the nodes-relating pars of the game:
        self.aux_clear()

    # entirely clears the grid:
    def clear_grid(self):
        # clearing the every node:
        for row in self.grid:
            for node in row:
                node.clear()
        # clearing the nodes-relating pars of the game:
        self.start_node, self.end_node = None, None
        self.aux_clear()
        self.walls = set()

    # auxiliary clearing methods for code simplifying:
    def aux_clear(self):
        self.nodes_visited = {}
        self.time_elapsed_ms = 0
        self.iterations = 0
        self.path = None
        self.path_index = 0
        self.curr_node_dict = {0: None}
        self.max_times_visited_dict = {0: 0}
        self.neighs_added_to_heap_dict = {}
        self.walls_built_erased = [([], True)]
        self.walls_index = 0
        self.in_interaction = False
        self.in_interaction_mode_lock = False
        self.greedy_flag_lock = False
        self.heuristic_lock = False
        self.tiebreakers_lock = False
        self.scale_lock = False
        self.bfs_dfs_lock = False
        # arrows and path triangles:
        self.arrow_shape_list = arcade.ShapeElementList()
        self.triangle_shape_list = arcade.ShapeElementList()

    def clear_inter_types(self, ind):
        for i, _ in enumerate(self.inter_types):
            if i != ind:
                self.inter_types[i] = InterType.NONE

    # erases all nodes, that are connected vertically, horizontally or diagonally to a chosen one,
    # then nodes connected to them the same way and so on recursively...
    def erase_all_linked_nodes(self, node: 'Node'):
        node.type = NodeType.EMPTY
        node.update_sprite_colour()
        self.walls.remove(self.number_repr(node))
        self.walls_built_erased[self.walls_index][0].append(self.number_repr(node))
        for neigh in node.get_extended_neighs(self):
            if neigh.type == NodeType.WALL:
                self.erase_all_linked_nodes(neigh)

    # UPDATING:
    # long press logic for mouse buttons:
    def update(self, delta_time: float):
        # icons spinning/moving:
        increments = [0.02, 0.15, 0.05, 0.25]
        for i, _ in enumerate(self.incrementers):
            if self.inter_types[i] == InterType.HOVERED:
                self.incrementers[i] += increments[i]
        self.play_button.update()

        self.step_button_right.update()
        self.step_button_left.update()

        self.eraser.update()

        # consecutive calls during key pressing:
        ticks_threshold = 12
        if self.cycle_breaker_right:
            self.ticks_before += 1
            if self.ticks_before >= ticks_threshold:
                if self.path is None:
                    if self.inter_types[1] == InterType.PRESSED:
                        self.bfs_step_up()
                    elif self.inter_types[2] == InterType.PRESSED:
                        self.a_star_step_up()
                    elif self.inter_types[3] == InterType.PRESSED:
                        self.wave_lee_step_up()
                else:
                    if self.path_index < len(self.path) - 1:
                        self.path_up()
                        self.path_index += 1
        if self.cycle_breaker_left:
            self.ticks_before += 1
            if self.ticks_before >= ticks_threshold:
                if self.path is None:
                    if self.inter_types[1] == InterType.PRESSED:
                        self.bfs_step_down()
                    elif self.inter_types[2] == InterType.PRESSED:
                        self.a_star_step_down()
                    elif self.inter_types[3] == InterType.PRESSED:
                        self.wave_lee_step_down()
                else:
                    if self.path_index > 0:
                        self.path_down()
                        self.path_index -= 1
                    else:
                        self.path = None
                        if self.inter_types[1] == InterType.PRESSED:
                            self.bfs_step_down()
                        elif self.inter_types[2] == InterType.PRESSED:
                            self.a_star_step_down()
                        elif self.inter_types[3] == InterType.PRESSED:
                            self.wave_lee_step_down()

    # KEYBOARD:
    def on_key_press(self, symbol: int, modifiers: int):
        # is called when user press the symbol key:
        match symbol:
            # a_star_call:
            case arcade.key.SPACE:
                if not self.loading:
                    if self.start_node and self.end_node:
                        # STEP BY STEP:
                        if self.interactive_ind is not None:
                            # game logic:
                            self.in_interaction = True
                            # prepare:
                            if self.inter_types[1] == InterType.PRESSED:
                                self.bfs_preparation()
                            elif self.inter_types[2] == InterType.PRESSED:
                                self.a_star_preparation()
                            elif self.inter_types[3] == InterType.PRESSED:
                                self.wave_lee_preparation()
                        # ONE SPACEBAR PRESS:
                        else:
                            start = time.time_ns()
                            # getting paths:
                            if self.inter_types[1] == InterType.PRESSED:
                                self.path = self.start_node.bfs(self.end_node, self)
                            elif self.inter_types[2] == InterType.PRESSED:
                                self.path = self.start_node.a_star(self.end_node, self)
                            elif self.inter_types[3] == InterType.PRESSED:
                                self.path = self.start_node.wave_lee(self.end_node, self)
                            finish = time.time_ns()
                            self.time_elapsed_ms = self.get_ms(start, finish)
                            # path's drawing for all three algos cores:
                            print(f"PATH'S LENGTH: {len(self.path)}")
                            for i, node in enumerate(self.path):
                                if node.type not in [NodeType.START_NODE, NodeType.END_NODE]:
                                    node.type = NodeType.PATH_NODE
                                    node.update_sprite_colour()
                                if i + 1 < len(self.path):
                                    p = -self.path[i + 1].x + self.path[i].x, \
                                        -self.path[i + 1].y + self.path[i].y
                                    p1, p2, p3 = self.get_triangle(self.path[i + 1], p)
                                    triangle_shape = arcade.create_triangles_filled_with_colors(
                                        [p1, p2, p3],
                                        [arcade.color.WHITE, arcade.color.RED, arcade.color.RED])
                                    self.triangle_shape_list.append(triangle_shape)
            # entirely grid clearing:
            case arcade.key.ENTER:
                self.clear_grid()
            # recall a_star :
            case arcade.key.BACKSPACE:
                self.clear_empty_nodes()
            # a_star interactive:
            case arcade.key.RIGHT:
                if self.in_interaction:
                    self.cycle_breaker_right = True
                    if self.path is None:
                        if self.inter_types[1] == InterType.PRESSED:
                            self.bfs_step_up()
                        elif self.inter_types[2] == InterType.PRESSED:
                            self.a_star_step_up()
                        elif self.inter_types[3] == InterType.PRESSED:
                            self.wave_lee_step_up()
                    else:
                        if self.path_index < len(self.path) - 1:
                            self.path_up()
                            self.path_index += 1
                elif self.loading:
                    self.change_nodes_type(NodeType.EMPTY, self.walls[f'ornament {self.loading_ind}'])
                    self.loading_ind = (self.loading_ind + 1) % len(self.walls)
                    self.change_nodes_type(NodeType.WALL, self.walls[f'ornament {self.loading_ind}'])
            case arcade.key.LEFT:
                if self.in_interaction:
                    self.cycle_breaker_left = True
                    if self.path is None:
                        if self.inter_types[1] == InterType.PRESSED:
                            self.bfs_step_down()
                        elif self.inter_types[2] == InterType.PRESSED:
                            self.a_star_step_down()
                        elif self.inter_types[3] == InterType.PRESSED:
                            self.wave_lee_step_down()
                    else:
                        if self.path_index > 0:
                            self.path_down()
                            self.path_index -= 1
                        else:
                            self.path = None
                            if self.inter_types[1] == InterType.PRESSED:
                                self.bfs_step_down()
                            elif self.inter_types[2] == InterType.PRESSED:
                                self.a_star_step_down()
                            elif self.inter_types[3] == InterType.PRESSED:
                                self.wave_lee_step_down()
                elif self.loading:
                    self.change_nodes_type(NodeType.EMPTY, self.walls[f'ornament {self.loading_ind}'])
                    self.loading_ind = (self.loading_ind - 1) % len(self.walls)
                    self.change_nodes_type(NodeType.WALL, self.walls[f'ornament {self.loading_ind}'])
            # heuristic showing for the every node::
            case arcade.key.F:
                self.f_flag = not self.f_flag
            # undoing and cancelling:
            case arcade.key.Z:  # undo
                if not (self.in_interaction or self.loading):
                    if self.walls_index > 0:
                        for num in (l := self.walls_built_erased[self.walls_index])[0]:
                            node = self.node(num)
                            node.type = NodeType.EMPTY if l[1] else NodeType.WALL
                            node.update_sprite_colour()
                            if l[1]:
                                self.walls.remove(self.number_repr(node))
                            else:
                                self.walls.add(self.number_repr(node))
                    self.walls_index -= 1
            case arcade.key.Y:  # cancels undo
                if not (self.in_interaction or self.loading):
                    if self.walls_index < len(self.walls_built_erased) - 1:
                        for num in (l := self.walls_built_erased[self.walls_index + 1])[0]:
                            node = self.node(num)
                            node.type = NodeType.WALL if l[1] else NodeType.EMPTY
                            node.update_sprite_colour()
                            if l[1]:
                                self.walls.add(self.number_repr(node))
                            else:
                                self.walls.remove(self.number_repr(node))
                        self.walls_index += 1
            # saving and loading:
            case arcade.key.S:  # TODO: AUX WINDOWS!!!
                print(f'walls: {self.walls}')
                with shelve.open(f'saved_walls {self.tiles_q}', 'c') as shelf:
                    index = len(shelf)
                    shelf[f'ornament {index}'] = self.walls
            case arcade.key.L:
                # cannot loading while in interaction:
                if not self.in_interaction:
                    if self.loading:
                        self.loading = False
                        self.walls = self.walls[f'ornament {self.loading_ind}']
                        self.walls_built_erased.append(([], False))
                        for num in self.walls:
                            self.walls_built_erased[self.walls_index][0].append(num)
                        self.walls_index += 1
                    elif len(self.walls) == 0:
                        with shelve.open(f'saved_walls {self.tiles_q}',
                                         'r') as shelf:  # TODO: flag to 'r' and check if the file exist!
                            index = len(shelf)
                            if index > 0:
                                self.loading = True
                                self.walls_built_erased.append(([], True))
                                for num in self.walls:
                                    self.walls_built_erased[self.walls_index][0].append(num)
                                self.walls_index += 1
                                self.walls = dict(shelf)

    def on_key_release(self, symbol: int, modifiers: int):
        match symbol:
            case arcade.key.RIGHT:
                self.cycle_breaker_right = False
                self.ticks_before = 0
            case arcade.key.LEFT:
                self.cycle_breaker_left = False
                self.ticks_before = 0

    # MOUSE:
    def on_mouse_motion(self, x, y, dx, dy):
        if self.building_walls_flag:
            if self.mode == 0:
                if self.build_or_erase is not None:
                    # now building the walls:
                    if self.build_or_erase:
                        n = self.get_node(x, y)
                        if n and n.type != NodeType.WALL:
                            n.type = NodeType.WALL
                            self.walls.add(self.number_repr(n))
                            n.update_sprite_colour()
                            if self.walls_index < len(self.walls_built_erased) - 1:
                                self.walls_built_erased = self.walls_built_erased[:self.walls_index + 1]
                            self.walls_built_erased.append(([self.number_repr(n)], self.build_or_erase))
                            self.walls_index += 1
                    # now erasing the walls:
                    else:
                        n = self.get_node(x, y)
                        if n and n.type == NodeType.WALL:
                            n.type = NodeType.EMPTY
                            self.walls.remove(self.number_repr(n))
                            n.update_sprite_colour()
                            if self.walls_index < len(self.walls_built_erased) - 1:
                                self.walls_built_erased = self.walls_built_erased[:self.walls_index + 1]
                            self.walls_built_erased.append(([self.number_repr(n)], self.build_or_erase))
                            self.walls_index += 1
        # GEAR WHEEL:
        if self.inter_types[0] != InterType.PRESSED:
            if arcade.is_point_in_polygon(x, y,
                                          self.upper_vertices_list):  # and (x - self.cx) ** 2 / self.hole_rx + (y - self.cy) / self.hole_ry ** 2 >= 1:
                self.inter_types[0] = InterType.HOVERED
            else:
                self.inter_types[0] = InterType.NONE
        # BFS/DFS:
        if self.inter_types[1] != InterType.PRESSED:
            if self.bfs_dfs_cx - self.bfs_dfs_size // 2 <= x <= self.bfs_dfs_cx + self.bfs_dfs_size // 2 \
                    and self.bfs_dfs_cy - self.bfs_dfs_size // 2 <= y <= self.bfs_dfs_cy + self.bfs_dfs_size // 2:
                self.inter_types[1] = InterType.HOVERED
            else:
                self.inter_types[1] = InterType.NONE
        # A_STAR:
        if self.inter_types[2] != InterType.PRESSED:
            if arcade.is_point_in_polygon(x, y, self.a_outer_points):
                self.inter_types[2] = InterType.HOVERED
            else:
                self.inter_types[2] = InterType.NONE
        # WAVE_LEE:
        if self.inter_types[3] != InterType.PRESSED:
            if (x - (1750 + 50 + 48 + 10)) ** 2 + (y - (1020 - 73 - 25)) ** 2 <= 32 ** 2:
                self.inter_types[3] = InterType.HOVERED
            else:
                self.inter_types[3] = InterType.NONE
        # MANAGEMENT MENU:
        # start icon:
        self.play_button.on_motion(x, y)
        # step buttons:
        self.step_button_right.on_motion(x, y)
        self.step_button_left.on_motion(x, y)
        # eraser:
        self.eraser.on_motion(x, y)
        # undo/redo:
        self.undo.on_motion(x, y)
        self.redo.on_motion(x, y)
        # SETTINGS:
        # arrows:
        if self.arrows_vertices is not None:
            for i in self.arrows_vertices.keys():
                if self.inter_types_arrows[i] != InterType.PRESSED:
                    if arcade.is_point_in_polygon(x, y, self.arrows_vertices[i]):
                        self.inter_types_arrows[i] = InterType.HOVERED
                    else:
                        self.inter_types_arrows[i] = InterType.NONE
        # arrows reset:
        if 1755 - self.arrow_height / 2 <= x <= 1755 + self.arrow_height / 2 and 785 - self.arrow_height / 2 <= y <= 785 + self.arrow_height / 2:
            self.inter_type_reset_square = InterType.HOVERED
        else:
            self.inter_type_reset_square = InterType.NONE

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int):
        # A_STAR BLOCK:
        if self.inter_types[2] == InterType.PRESSED:
            # setting_up heuristic and tiebreaker:
            if not self.heuristic_lock:
                for i in range(len(self.heuristic_names)):
                    if Icon.is_point_in_square(SCREEN_WIDTH - 225, SCREEN_HEIGHT - 100 - 120 - (
                            18 + 2 * 2 + 18) * i, 18, x, y):
                        self.heuristic = i
                        break
            if not self.tiebreakers_lock:
                for i in range(len(self.tiebreaker_names)):
                    if Icon.is_point_in_square(SCREEN_WIDTH - 225, SCREEN_HEIGHT - 100 - 120 - (
                            18 + 2 * 2 + 18) * (3 + i) - 18 * 3 - 30, 18, x, y):
                        self.tiebreaker = None if self.tiebreaker == i else i
                        break
            # setting up the greedy flag:
            if Icon.is_point_in_square(SCREEN_WIDTH - 225, SCREEN_HEIGHT - 130 - 120 - (
                    18 + 2 * 2 + 18) * 4 - 2 * 18 * 3 - 30, 18, x, y):
                if not self.greedy_flag_lock:
                    if self.greedy_ind is None:
                        self.greedy_ind = 0
                    else:
                        self.greedy_ind = None
            # setting up the arrows:
            if Icon.is_point_in_square(SCREEN_WIDTH - 225, SCREEN_HEIGHT - 160 - 120 - (
                    18 + 2 * 2 + 18) * 4 - 3 * 18 * 3 - 30, 18, x, y):
                if self.guide_arrows_ind is None:
                    self.guide_arrows_ind = 0
                else:
                    self.guide_arrows_ind = None
            # setting up the interactive a_star flag:
            if Icon.is_point_in_square(SCREEN_WIDTH - 225, SCREEN_HEIGHT - 190 - 120 - (
                    18 + 2 * 2 + 18) * 4 - 4 * 18 * 3 - 30, 18, x, y):
                if not self.in_interaction_mode_lock:
                    if self.interactive_ind is None:
                        self.interactive_ind = 0
                    else:
                        self.interactive_ind = None
        # BFS/DFS BLOCK:
        elif self.inter_types[1] == InterType.PRESSED:
            if Icon.is_point_in_square(SCREEN_WIDTH - 225, SCREEN_HEIGHT - 100 - 120, 18, x, y):
                self.bfs_dfs_ind = 0
            elif Icon.is_point_in_square(SCREEN_WIDTH - 225, SCREEN_HEIGHT - 100 - 120 - (
                    18 + 2 * 2 + 18), 18, x, y):
                self.bfs_dfs_ind = 1
        # WAVE LEE BLOCK:
        elif self.inter_types[3] == InterType.PRESSED:
            ...  # mhe 36 366 98 989 LL
        # SETTINGS BLOCK:
        elif self.inter_types[0] == InterType.PRESSED:
            # arrows-hints:
            if self.choosing_arrows:
                for i in self.arrows_vertices.keys():
                    if arcade.is_point_in_polygon(x, y, self.arrows_vertices[i]):
                        if self.inter_types_arrows[i] == InterType.HOVERED:
                            self.inter_types_arrows[i] = InterType.PRESSED
                            self.arrows_indices.append(i)
                            self.walk_index += 1
                            if self.walk_index == 4:
                                self.choosing_arrows = False
                                Node.walk = [self.walk[self.arrows_indices[_]] for _ in range(4)]
            # arrows reset:
            if Icon.is_point_in_square(1755, 785, self.arrow_height / 2, x, y):
                self.choosing_arrows = True
                self.walk_index = 0
                self.arrows_indices = []
                self.inter_types_arrows = [InterType.NONE for _ in range(4)]
            # mhe 36 366 98 989
            # choosing the scale factor:
            if not self.scale_lock:
                for i in range(len(self.scale_names)):
                    if Icon.is_point_in_square(SCREEN_WIDTH - 225, SCREEN_HEIGHT - 70 - 120 - (
                            18 + 2 * 2 + 18) * i - 3 * 18 * 3 - 30, 18, x, y):
                        self.scale = i
                        self.rebuild_map()
        # MODES OF DRAWING LOGIC:
        if self.mode == 0:
            self.building_walls_flag = True
            if button == arcade.MOUSE_BUTTON_LEFT:
                self.build_or_erase = True
            elif button == arcade.MOUSE_BUTTON_RIGHT:
                self.build_or_erase = False
            elif button == arcade.MOUSE_BUTTON_MIDDLE:
                self.build_or_erase = None
                n = self.get_node(x, y)
                if n:
                    if self.walls_index < len(self.walls_built_erased) - 1:
                        self.walls_built_erased = self.walls_built_erased[: self.walls_index + 1]
                    self.walls_built_erased.append(([], False))
                    self.walls_index += 1
                    self.erase_all_linked_nodes(n)
        elif self.mode == 1:
            if button == arcade.MOUSE_BUTTON_LEFT:
                sn = self.get_node(x, y)
                if sn:
                    if self.start_node:
                        self.start_node.type = NodeType.EMPTY
                        self.start_node.update_sprite_colour()
                    sn.type = NodeType.START_NODE
                    self.start_node = sn
                    self.start_node.update_sprite_colour()
            elif button == arcade.MOUSE_BUTTON_RIGHT:
                en = self.get_node(x, y)
                if en:
                    if self.end_node:
                        self.end_node.type = NodeType.EMPTY
                        self.end_node.update_sprite_colour()
                    en.type = NodeType.END_NODE
                    self.end_node = en
                    self.end_node.update_sprite_colour()
        elif self.mode == 2:  # a_star interactive -->> info getting:
            n = self.get_node(x, y)
            if n:
                if self.node_chosen == n:
                    self.node_chosen = None
                else:
                    self.a_star_choose_node(n)
        # GEAR_WHEEL:
        if arcade.is_point_in_polygon(x, y, self.upper_vertices_list):
            if self.inter_types[0] == InterType.HOVERED:
                self.inter_types[0] = InterType.PRESSED
                self.clear_inter_types(0)
            elif self.inter_types[0] == InterType.PRESSED:
                self.inter_types[0] = InterType.HOVERED
        # BFS/DFS:
        if self.bfs_dfs_cx - self.bfs_dfs_size / 2 <= x <= self.bfs_dfs_cx + self.bfs_dfs_size / 2 \
                and self.bfs_dfs_cy - self.bfs_dfs_size / 2 <= y <= self.bfs_dfs_cy + self.bfs_dfs_size / 2:
            if self.inter_types[1] == InterType.HOVERED:
                self.inter_types[1] = InterType.PRESSED
                self.clear_inter_types(1)
            elif self.inter_types[1] == InterType.PRESSED:
                self.inter_types[1] = InterType.HOVERED
        # A_STAR:
        if arcade.is_point_in_polygon(x, y, self.a_outer_points):
            if self.inter_types[2] == InterType.HOVERED:
                self.inter_types[2] = InterType.PRESSED
                self.clear_inter_types(2)
            elif self.inter_types[2] == InterType.PRESSED:
                self.inter_types[2] = InterType.HOVERED
        # WAVE_LEE:
        if Icon.is_point_in_circle(1750 + 50 + 48 + 10, 1020 - 73 - 25, 32, x, y):
            if self.inter_types[3] == InterType.HOVERED:
                self.inter_types[3] = InterType.PRESSED
                self.clear_inter_types(3)
            elif self.inter_types[3] == InterType.PRESSED:
                self.inter_types[3] = InterType.HOVERED
        # ICONS:
        # play button:
        self.play_button.on_press(x, y)
        # step buttons:
        self.step_button_right.on_press(x, y)
        self.step_button_left.on_press(x, y)
        # eraser:
        self.eraser.on_press(x, y)
        # undo/redo:
        self.undo.on_press(x, y)
        self.redo.on_press(x, y)

    # game mode switching by scrolling the mouse wheel:
    def on_mouse_scroll(self, x: int, y: int, scroll_x: int, scroll_y: int):
        self.mode = (self.mode + 1) % len(self.mode_names)

    def on_mouse_release(self, x: int, y: int, button: int, modifiers: int):
        self.building_walls_flag = False
        self.step_button_right.on_release(x, y)
        self.step_button_left.on_release(x, y)
        self.eraser.on_release(x, y)
        self.undo.on_release(x, y)
        self.redo.on_release(x, y)

    # SOME AUXILIARY METHODS:
    # changing the type of the node and then changes the node's sprite colour:
    def change_nodes_type(self, node_type: 'NodeType', walls_set: set or list):
        for node_num in walls_set:
            y, x = self.coords(node_num)
            self.grid[y][x].type = node_type
            self.grid[y][x].update_sprite_colour()

    # make a node the chosen one:
    def a_star_choose_node(self, node: 'Node'):
        self.node_chosen = node
        # draw a frame

    # gets the number representation for the node:
    def number_repr(self, node: 'Node'):
        return node.y * self.hor_tiles_q + node.x

    # gets node's coordinates for its number representation:
    def coords(self, number: int):
        return divmod(number, self.hor_tiles_q)

    # gets the node itself for its number representation:
    def node(self, num: int) -> 'Node':
        y, x = self.coords(num)
        return self.grid[y][x]

    # gets the node from the current mouse coordinates:
    def get_node(self, mouse_x, mouse_y):
        x_, y_ = mouse_x - 5, mouse_y - 5
        x, y = x_ // self.tile_size, y_ // self.tile_size
        return self.grid[y][x] if 0 <= x < self.hor_tiles_q and 0 <= y < self.tiles_q else None

    @staticmethod
    def get_ms(start, finish):
        return (finish - start) // 10 ** 6


class DrawLib:

    def __init__(self):
        self._sq_size = 18
        self._sq_line_w = 2
        self._delta = 2 * (self._sq_size + self._sq_line_w)

    def get_delta(self):
        return self._delta

    def get_sq_size(self):
        return self._sq_size

    def get_sq_line_w(self):
        return self._sq_line_w

    def draw_levi_gin(self):
        arcade.draw_circle_outline(1000, 500, 250, arcade.color.BLACK, 2)

    # draws the area (part) of right-sided menu:
    def draw_area(self, x, y, header_name: str, el_names: dict, el_ind: int, el_lock: bool):
        arcade.Text(f'{header_name}: ', x, y, arcade.color.BLACK, bold=True).draw()

        for i in range(len(el_names)):
            arcade.draw_rectangle_outline(x + 10, y - 30 - self._delta * i, self._sq_size, self._sq_size,
                                          arcade.color.BLACK, self._sq_line_w)
            arcade.Text(f'{el_names[i]}', x + 10 + self._sq_size + 2 * self._sq_line_w,
                        y - 30 - self._delta * i - 6, arcade.color.BLACK, bold=True).draw()

        if el_ind is not None:
            arcade.draw_rectangle_filled(x + 10, y - 30 - self._delta * el_ind,
                                         14, 14, arcade.color.BLACK)
        # heuristics lock:
        if el_lock:
            if el_ind is not None:
                self.draw_lock(x + 10, y - 30 - self._delta * el_ind)
            for i in range(len(el_names)):
                if el_ind is None or i != el_ind:
                    self.draw_cross(x + 10, y - 30 - self._delta * i)

    # draws a lock for right window part: 36 366 98 989
    @staticmethod
    def draw_lock(center_x: int, center_y: int):
        arcade.draw_rectangle_filled(center_x, center_y, 14, 14, arcade.color.RED)
        arcade.draw_rectangle_outline(center_x, center_y + 7, 8.4, 16.8, arcade.color.RED, border_width=2)

    # draws the cross of forbiddance:
    @staticmethod
    def draw_cross(center_x: int, center_y: int):
        arcade.draw_line(center_x - 9, center_y + 9, center_x + 9, center_y - 9, arcade.color.BLACK, line_width=2)
        arcade.draw_line(center_x + 9, center_y + 9, center_x - 9,
                         center_y - 9, arcade.color.BLACK, line_width=2)

    # draws directions priority choosing menu:
    def draw_arrows_menu(self, x, y, arrow_length, arrow_height, game: Lastar):
        arcade.draw_text(f'Directions priority: ', x, y, arcade.color.BLACK,
                         bold=True)

        for dx, dy in Lastar.walk:
            self.make_arrow(1755 + dx * arrow_length, 785 + dy * arrow_length, arrow_length,
                            arrow_height, 2,
                            (dx, dy), arcade.color.SANDY_BROWN, game)

        # ARROWS RESET:
        arcade.draw_rectangle_outline(1755, 785, arrow_height, arrow_height, arcade.color.BLACK,
                                      2 + (0 if game.inter_type_reset_square == InterType.NONE else 1))

    # draws left, right, up or down arrow:
    def make_arrow(self, cx, cy, arrow_length, arrow_height, line_w, point: tuple[int, int],
                   colour: tuple[int, int, int], game: Lastar):
        # index:
        ind = Lastar.walk.index(point)
        # center coords:
        cx_ = cx - point[0] * arrow_height / 2
        cy_ = cy - point[1] * arrow_height / 2

        w, h = abs(point[0]) * (arrow_length - arrow_height) + abs(point[1]) * arrow_height, abs(
            point[0]) * arrow_height + abs(point[1]) * (arrow_length - arrow_height)
        _dx, dx_ = arrow_length / 2 - arrow_height, arrow_length / 2
        h_ = arrow_height / 2

        if game.inter_types_arrows[ind] == InterType.PRESSED:
            arcade.draw_rectangle_filled(cx_, cy_, w, h, colour)

            arcade.draw_triangle_filled(cx + point[0] * _dx + point[1] * arrow_height,
                                        cy + point[1] * _dx + point[0] * arrow_height,
                                        cx + point[0] * _dx - point[1] * arrow_height,
                                        cy + point[1] * _dx - point[0] * arrow_height,
                                        cx + point[0] * arrow_length / 2, cy + point[1] * arrow_length / 2,
                                        colour)

            # numbers-hints:
            signed_delta = arrow_height / 2 - arrow_height / 12
            arcade.Text(f'{game.arrows_indices.index(ind) + 1}',
                        cx - arrow_height / 3 + arrow_height / 12 - point[0] * signed_delta,
                        cy - arrow_height / 3 - point[1] * signed_delta, arcade.color.BLACK, 2 * arrow_height / 3,
                        bold=True).draw()

        if game.arrows_vertices is None:
            game.arrows_vertices = {}

        if len(game.arrows_vertices) < 4:
            game.arrows_vertices[Lastar.walk.index(point)] = [
                [cx + point[0] * arrow_length / 2, cy + point[1] * arrow_length / 2],
                [cx + point[0] * _dx + point[1] * arrow_height,
                 cy + point[1] * _dx + point[0] * arrow_height],
                [cx + point[0] * _dx + point[1] * h_, cy + point[1] * _dx + point[0] * h_],
                [cx - point[0] * dx_ + point[1] * h_, cy - point[1] * dx_ + point[0] * h_],
                [cx - point[0] * dx_ - point[1] * h_, cy - point[1] * dx_ - point[0] * h_],
                [cx + point[0] * _dx - point[1] * h_, cy + point[1] * _dx - point[0] * h_],
                [cx + point[0] * _dx - point[1] * arrow_height,
                 cy + point[1] * _dx - point[0] * arrow_height]
            ]

        arcade.draw_polygon_outline(game.arrows_vertices[ind], arcade.color.BLACK,
                                    line_w if game.inter_types_arrows[ind] == InterType.NONE else line_w + 1)

    # by default the arrow to be drawn is left sided:
    def create_line_arrow(self, node: 'Node', deltas: tuple[int, int] = (-1, 0),
                          game: Lastar = None):  # left arrow by default
        cx, cy = 5 + node.x * game.tile_size + game.tile_size / 2, 5 + node.y * game.tile_size + game.tile_size / 2
        h = 2 * game.tile_size // 3
        _h, h_, dh = h / 6, h / 3, h / 2  # for 90 degrees triangle
        shape = arcade.create_triangles_filled_with_colors(
            (
                (cx + deltas[0] * h_, cy + deltas[1] * h_),
                (cx - (deltas[0] * _h + deltas[1] * dh), cy - (deltas[0] * dh + deltas[1] * _h)),
                (cx - (deltas[0] * _h - deltas[1] * dh), cy - (-deltas[0] * dh + deltas[1] * _h))
            ),
            (
                arcade.color.BLACK,
                arcade.color.BLACK,
                arcade.color.BLACK
            )
        )
        return shape

    # draws a spinning gear wheel:
    def draw_gear_wheel(self, cx, cy, rx=32, ry=32, cog_size=8, multiplier=1.5, line_w=2, shift=False, clockwise=True,
                        game: Lastar = None):
        game.cx, game.cy = cx, cy
        game.hole_rx, game.hole_ry = rx - multiplier * cog_size, ry - multiplier * cog_size
        circumference = math.pi * (rx + ry)  # approximately if cog_size << radius
        angular_size = (2 * math.pi) * cog_size / circumference
        max_cogs_fit_in_the_gear_wheel = int(circumference / cog_size)
        cogs_q = max_cogs_fit_in_the_gear_wheel // 2
        fit_angular_size = (2 * math.pi - cogs_q * angular_size) / cogs_q
        angle = (angular_size if shift else 0) + (
            game.incrementers[0] if clockwise else -game.incrementers[0])  # in radians
        game.upper_vertices_list = []
        for i in range(cogs_q):
            # aux pars:
            _a, a_ = (angle - angular_size / 2), (angle + angular_size / 2)
            _rx, _ry, rx_, ry_ = rx * math.cos(_a), ry * math.sin(_a), rx * math.cos(a_), ry * math.sin(a_)
            _dx, _dy = cog_size * math.cos(_a), cog_size * math.sin(_a)
            dx_, dy_ = cog_size * math.cos(a_), cog_size * math.sin(a_)
            # polygon's points:
            game.upper_vertices_list.append([cx + _rx, cy + _ry])
            game.upper_vertices_list.append([cx + _rx + _dx, cy + _ry + _dy])
            game.upper_vertices_list.append([cx + rx_ + dx_, cy + ry_ + dy_])
            game.upper_vertices_list.append([cx + rx_, cy + ry_])
            # angle incrementation:
            angle += angular_size + fit_angular_size
        # upper gear wheel:
        # arcade.draw_polygon_filled(upper_vertices_list, arcade.color.PASTEL_GRAY)
        if game.inter_types[0] == InterType.PRESSED:
            arcade.draw_polygon_filled(game.upper_vertices_list, arcade.color.RED)
        arcade.draw_polygon_outline(game.upper_vertices_list, arcade.color.BLACK,
                                    line_w + (0 if game.inter_types[0] == InterType.NONE else 1))
        # hole:
        arcade.draw_ellipse_filled(cx, cy, 2 * (rx - multiplier * cog_size), 2 * (ry - multiplier * cog_size),
                                   arcade.color.DUTCH_WHITE)
        arcade.draw_ellipse_outline(cx, cy, 2 * (rx - multiplier * cog_size), 2 * (ry - multiplier * cog_size),
                                    arcade.color.BLACK,
                                    line_w + (0 if game.inter_types[0] == InterType.NONE else 1))

    # draws round expending waves:
    def draw_waves(self, cx, cy, size=32, waves_q=5, line_w=2, game: Lastar = None):
        ds = size / waves_q
        s_list = sorted([(i * ds + game.incrementers[3]) % size for i in range(waves_q)], reverse=True)
        for i, curr_s in enumerate(s_list):
            if game.inter_types[3] == InterType.PRESSED:
                arcade.draw_circle_filled(cx, cy, curr_s, arcade.color.RED if i % 2 == 0 else arcade.color.DUTCH_WHITE)
            arcade.draw_circle_outline(cx, cy, curr_s, arcade.color.BLACK,
                                       line_w + (0 if game.inter_types[3] == InterType.NONE else 1))

    # draws an a_star label:
    def draw_a_star(self, cx, cy, size_w, size_h, line_w=2, clockwise=True, game: Lastar = None):
        # drawing A:
        self.draw_a(cx, cy, size_w, size_h, size_w / 3, line_w, game)
        # Star spinning around A:
        self.draw_star(cx + size_h / 2 + size_h / 3, cy + size_h, r=size_h / 4, line_w=line_w, clockwise=clockwise,
                       game=game)

    # draws a spinning star:
    def draw_star(self, cx, cy, vertices=5, r=32, line_w=2, clockwise=True, game: Lastar = None):
        delta_angle = 2 * math.pi / vertices
        d = vertices // 2
        angle = game.incrementers[2] if clockwise else -game.incrementers[2]  # in radians
        for i in range(vertices):
            da = d * delta_angle
            arcade.draw_line(cx + r * math.cos(angle),
                             cy + r * math.sin(angle),
                             cx + r * math.cos(angle + da),
                             cy + r * math.sin(angle + da),
                             arcade.color.BLACK, line_w)
            angle += da

    # draws 'A' letter:
    def draw_a(self, cx, cy, length, height, a_w, line_w, game: Lastar = None):
        upper_hypot = math.sqrt(length ** 2 + height ** 2)
        cos, sin = height / upper_hypot, length / upper_hypot
        line_w_hour_projection = a_w / cos
        dh = a_w / sin
        delta = (height - a_w - dh) / 2
        k, upper_k = (delta - a_w / 2) * length / height, (delta + a_w / 2) * length / height

        game.a_outer_points = [
            [cx, cy],
            [cx + length, cy + height],
            [cx + 2 * length, cy],
            [cx + 2 * length - line_w_hour_projection, cy],
            [cx + 2 * length - line_w_hour_projection - k, cy + delta - a_w / 2],
            [cx + k + line_w_hour_projection, cy + delta - a_w / 2],
            [cx + line_w_hour_projection, cy]
        ]

        a_inner_points = [
            [cx + length, cy + height - dh],
            [cx + 2 * length - line_w_hour_projection - upper_k, cy + delta + a_w / 2],
            [cx + line_w_hour_projection + upper_k, cy + delta + a_w / 2]
        ]

        if game.inter_types[2] == InterType.NONE:
            arcade.draw_polygon_outline(game.a_outer_points, arcade.color.BLACK, line_w)
            arcade.draw_polygon_outline(a_inner_points, arcade.color.BLACK, line_w)
        elif game.inter_types[2] == InterType.HOVERED:
            arcade.draw_polygon_outline(game.a_outer_points, arcade.color.BLACK, line_w + 1)
            arcade.draw_polygon_outline(a_inner_points, arcade.color.BLACK, line_w + 1)
        else:
            arcade.draw_polygon_filled(game.a_outer_points, arcade.color.RED)
            arcade.draw_polygon_outline(game.a_outer_points, arcade.color.BLACK, line_w + 1)
            arcade.draw_polygon_filled(a_inner_points, arcade.color.DUTCH_WHITE)
            arcade.draw_polygon_outline(a_inner_points, arcade.color.BLACK, line_w + 1)

    # simplified version, draws a living BFS/DFS label:
    def draw_bfs_dfs(self, cx, cy, size, line_w, game: Lastar = None):
        game.bfs_dfs_cx, game.bfs_dfs_cy = cx, cy
        game.bfs_dfs_size = size
        # filling:
        if game.inter_types[1] == InterType.PRESSED:
            arcade.draw_rectangle_filled(cx, cy, size, size, arcade.color.RED)
        # border:
        arcade.draw_rectangle_outline(cx, cy, size, size, arcade.color.BLACK,
                                      line_w + (0 if game.inter_types[1] == InterType.NONE else 1))
        # text:
        text_size = size / 4
        magnitude = size / 12
        arcade.Text('B', cx - size / 3, cy + text_size / 4 + magnitude * math.sin(game.incrementers[1]),
                    arcade.color.BLACK, text_size,
                    bold=True).draw()
        arcade.Text('D', cx - size / 3, cy - text_size - text_size / 4 + magnitude * math.sin(game.incrementers[1]),
                    arcade.color.BLACK,
                    text_size, bold=True).draw()

        arcade.Text('F', cx - size / 3 + size / 4,
                    cy + text_size / 4 + magnitude * math.sin(math.pi / 2 + game.incrementers[1]),
                    arcade.color.BLACK,
                    text_size,
                    bold=True).draw()
        arcade.Text('F', cx - size / 3 + size / 4,
                    cy - text_size - text_size / 4 + magnitude * math.sin(math.pi / 2 + game.incrementers[1]),
                    arcade.color.BLACK, text_size,
                    bold=True).draw()

        arcade.Text('S', cx - size / 3 + size / 4 + size / 4 - size / 32,
                    cy + text_size / 4 + magnitude * math.sin(math.pi + game.incrementers[1]),
                    arcade.color.BLACK, text_size,
                    bold=True).draw()
        arcade.Text('S', cx - size / 3 + size / 4 + size / 4 - size / 32,
                    cy - text_size - text_size / 4 + magnitude * math.sin(math.pi + game.incrementers[1]),
                    arcade.color.BLACK, text_size,
                    bold=True).draw()

    def draw_save(self):
        ...

    def draw_load(self):
        ...


class Grid:
    def __init__(self, hor_tiles_q, tiles_q):
        self._hor_tiles_q = hor_tiles_q
        self._tiles_q = tiles_q
        self._grid = None

    @property
    def grid(self):
        return self._grid

    def initialize(self):
        self._grid = [[Node(j, i, 1, NodeType.EMPTY) for i in range(self._hor_tiles_q)] for j in range(self._tiles_q)]

    def rebuild_map(self):
        ...

    def clear_empty_nodes(self):
        ...

    def clear_grid(self):
        ...

    def aux_clear(self):
        ...

    def get_sprites(self):
        ...

    def make_grid_lines(self):
        ...


class WallsManager:
    def __init__(self):
        ...

    def change_nodes_type(self, node_type: 'NodeType', walls_set: set or list):
        ...

    def erase_all_linked_nodes(self):
        ...

    def build_wall(self):
        ...

    def erase_wall(self):
        ...

    def undo(self):
        ...

    def redo(self):
        ...

    def save(self):
        ...

    def load(self):
        ...


# class for a node representation:
class Node:
    # horizontal and vertical up and down moves:
    walk = [(dy, dx) for dx in range(-1, 2) for dy in range(-1, 2) if dy * dx == 0 and (dy, dx) != (0, 0)]
    extended_walk = [(dy, dx) for dx in range(-1, 2) for dy in range(-1, 2) if (dy, dx) != (0, 0)]
    IS_GREEDY = False
    # for a_star (accurate removing from heap):
    aux_equal_flag = False

    def __init__(self, y, x, val, node_type: 'NodeType'):
        # type and sprite:
        self.type = node_type
        self.sprite = None
        # arrow shape:
        self.arrow_shape = None  # for more comprehensive visualization, consist of three line shapes
        # important pars:
        self.y, self.x = y, x
        self.val = val
        self.previously_visited_node = None  # for building the shortest path of Nodes from the starting point to the ending one
        self.times_visited = 0
        # cost and heuristic vars:
        self.g = np.Infinity  # aggregated cost of moving from start to the current Node, Infinity chosen for convenience and algorithm's logic
        self.h = 0  # approximated cost evaluated by heuristic for path starting from the current node and ending at the exit Node
        self.tiebreaker = None  # recommended for many shortest paths situations
        # f = h + g or total cost of the current Node is not needed here
        # heur dict, TODO: (it should be implemented in Astar class instead of node one) (medium, easy):
        self.heuristics = {0: self.manhattan_distance, 1: self.euclidian_distance, 2: self.max_delta,
                           3: self.no_heuristic}
        self.tiebreakers = {0: self.vector_cross_product_deviation, 1: self.coordinates_pair}

    # COPYING/RESTORING:
    # makes an auxiliary copy for a nde, it is needed for a_star interactive:
    def aux_copy(self):
        copied_node = Node(self.y, self.x, self.type, self.val)
        copied_node.g = self.g
        copied_node.h = self.h
        copied_node.tiebreaker = self.tiebreaker
        copied_node.type = self.type
        copied_node.previously_visited_node = self.previously_visited_node
        return copied_node

    # restore the node from its auxiliary copy:
    def restore(self, copied_node: 'Node'):
        self.g = copied_node.g
        self.h = copied_node.h
        self.tiebreaker = copied_node.tiebreaker
        self.type = copied_node.type  # NodeType.EMPTY ???
        self.update_sprite_colour()
        self.previously_visited_node = copied_node.previously_visited_node

    # SMART COPYING/RESTORING:
    def smart_copy(self, attributes: list[str]):
        copied_node = Node(self.y, self.x, self.type, self.val)
        self.smart_core(copied_node, attributes)
        return copied_node

    def smart_restore(self, other: 'Node', attributes: list[str]):
        other.smart_core(self, attributes)
        if 'type' in attributes:
            self.update_sprite_colour()

    def smart_core(self, other: 'Node', attributes: list[str]):
        for attribute in attributes:
            other.__dict__[attribute] = self.__getattribute__(attribute)

    # TYPE/SPRITE CHANGE/INIT:
    # makes a solid colour sprite for a node:
    def get_solid_colour_sprite(self, game: Lastar):
        cx, cy, size, colour = self.get_center_n_sizes(game)
        self.sprite = arcade.SpriteSolidColor(size, size, colour)
        self.sprite.center_x, self.sprite.center_y = cx, cy
        game.node_sprite_list.append(self.sprite)

    # aux calculations:
    def get_center_n_sizes(self, game: Lastar):
        return (5 + game.tile_size * self.x + game.tile_size / 2,
                5 + game.tile_size * self.y + game.tile_size / 2,
                game.tile_size - 2 * game.line_width - (1 if game.line_width % 2 != 0 else 0),
                self.type.value)

    # updates the sprite's color (calls after node's type switching)
    def update_sprite_colour(self):
        self.sprite.color = self.type.value

    # appends the arrow shape of the node to the arrow_shape_list in Astar class
    def append_arrow(self, game: Lastar):
        game.arrow_shape_list.append(self.arrow_shape)

    # removes the arrow shape of the node from the arrow_shape_list in Astar class
    def remove_arrow(self, game: Lastar):
        game.arrow_shape_list.remove(self.arrow_shape)
        self.arrow_shape = None

    def remove_arrow_from_shape_list(self, game: 'Lastar'):
        game.arrow_shape_list.remove(self.arrow_shape)

    def __str__(self):
        return f'{self.y, self.x} -->> {self.val}'

    def __repr__(self):
        return str(self)

    # DUNDERS:
    def __eq__(self, other):
        if type(self) != type(other):
            return False
        if Node.aux_equal_flag:
            return (self.y, self.x, self.g, self.h) == (other.y, other.x, other.g, other.h)  # .tiebreaker???
        else:
            return (self.y, self.x) == (other.y, other.x)

    # this is needed for using Node objects in priority queue like heapq and so on
    def __lt__(self, other: 'Node'):
        if self.IS_GREEDY:
            return (self.h, self.tiebreaker) < (other.h, other.tiebreaker)
        else:
            return (self.g + self.h, self.tiebreaker) < (other.g + other.h, other.tiebreaker)

    def __hash__(self):
        return hash((self.y, self.x))

    # CLEARING:
    # entirely clears the node, returning it to the initial state:
    def clear(self):
        self.heur_clear()
        self.type = NodeType.EMPTY
        self.update_sprite_colour()
        self.arrow_shape = None

    # clears the node heuristically:
    def heur_clear(self):
        self.g = np.Infinity
        self.h = 0
        self.tiebreaker = None
        self.previously_visited_node = None
        self.times_visited = 0
        # wave lee:
        self.val = 1

    # HEURISTICS:
    @staticmethod
    def manhattan_distance(node1, node2: 'Node'):
        return abs(node1.y - node2.y) + abs(node1.x - node2.x)

    @staticmethod
    def euclidian_distance(node1, node2: 'Node'):
        return math.sqrt((node1.y - node2.y) ** 2 + (node1.x - node2.x) ** 2)

    @staticmethod
    def max_delta(node1, node2: 'Node'):
        return max(abs(node1.y - node2.y), abs(node1.x - node2.x))

    @staticmethod
    def no_heuristic(node1, node2: 'Node'):
        return 0

    # SELF * OTHER, TIEBREAKER:
    @staticmethod
    def vector_cross_product_deviation(start, end, neigh):
        v1 = neigh.y - start.y, neigh.x - start.x
        v2 = end.y - neigh.y, end.x - neigh.x
        return abs(v1[0] * v2[1] - v1[1] * v2[0])

    @staticmethod
    def coordinates_pair(start, end, neigh):
        return neigh.y, neigh.x

    # NEIGHS:
    # gets neighs of the node, now can be set up:
    def get_neighs(self, game: 'Lastar', forbidden_node_types: list['NodeType']):  # has become smarter
        for dy, dx in self.walk:
            ny, nx = self.y + dy, self.x + dx
            if 0 <= ny < game.tiles_q and 0 <= nx < game.hor_tiles_q:
                # by default can visit the already visited nodes
                if game.grid[ny][nx].type not in forbidden_node_types:
                    yield game.grid[ny][nx]

    # gets extended neighs (with diagonal ones) of the node, generator:
    def get_extended_neighs(self, game: 'Lastar') -> list['Node']:
        for dy, dx in self.extended_walk:
            ny, nx = self.y + dy, self.x + dx
            if 0 <= ny < game.tiles_q and 0 <= nx < game.hor_tiles_q:
                yield game.grid[ny][nx]

    # PATHFINDING:
    # Witchdoctor's algos:
    def bfs(self, other: 'Node', game: 'Lastar'):  # iterative one:
        queue = deque()
        queue.append(self)
        while queue:
            game.iterations += 1
            current_node = queue.pop()
            if current_node.type not in [NodeType.START_NODE, NodeType.END_NODE]:
                current_node.type = NodeType.VISITED_NODE
                current_node.update_sprite_colour()
                current_node.times_visited += 1
            if current_node == other:
                return self.restore_path(other)
            for neigh in current_node.get_neighs(game, [NodeType.START_NODE, NodeType.VISITED_NODE, NodeType.WALL] + (
                    [NodeType.NEIGH] if game.bfs_dfs_ind == 0 else [])):
                if neigh.type != NodeType.END_NODE:
                    neigh.type = NodeType.NEIGH
                    neigh.update_sprite_colour()
                neigh.previously_visited_node = current_node
                # BFS:
                if game.bfs_dfs_ind == 0:
                    queue.appendleft(neigh)
                # DFS:
                else:
                    queue.append(neigh)

    # finished, tested and approved by Levi Gin:
    def wave_lee(self, other: 'Node', game: 'Lastar'):  # TODO: make it MORE INFORMATIVE:::
        # other.get_neighs(game)  # Why is it necessary???
        front_wave = {self}
        iteration = 0
        # wave-spreading:
        while front_wave:
            iteration += 1
            new_front_wave = set()
            for front_node in front_wave:
                front_node.val = iteration
                if front_node not in [self, other]:
                    front_node.type = NodeType.VISITED_NODE
                    front_node.update_sprite_colour()
                if front_node == other:
                    return self.restore_path(other)
                for front_neigh in front_node.get_neighs(game,
                                                         [NodeType.START_NODE, NodeType.VISITED_NODE, NodeType.WALL]):
                    if front_neigh not in new_front_wave:
                        front_neigh.previously_visited_node = front_node
                        new_front_wave.add(front_neigh)
            front_wave = set() | new_front_wave
        return []

    # a common a_star:
    def a_star(self, other: 'Node', game: 'Lastar'):
        Node.IS_GREEDY = False if game.greedy_ind is None else True
        # game.get_all_neighs()
        nodes_to_be_visited = [self]
        self.g = 0
        hq.heapify(nodes_to_be_visited)
        max_times_visited = 0
        # the main cycle:
        while nodes_to_be_visited:
            game.iterations += 1
            curr_node = hq.heappop(nodes_to_be_visited)
            if curr_node not in [self, other]:
                curr_node.type = NodeType.VISITED_NODE
                curr_node.update_sprite_colour()
            curr_node.times_visited += 1
            max_times_visited = max(max_times_visited, curr_node.times_visited)
            game.nodes_visited[curr_node] = 1
            # base case of finding the shortest path:
            if curr_node == other:
                break
            # next step:
            for neigh in curr_node.get_neighs(game, [NodeType.WALL]):
                if neigh.g > curr_node.g + neigh.val:
                    neigh.g = curr_node.g + neigh.val
                    neigh.h = neigh.heuristics[game.heuristic](neigh, other)
                    if game.tiebreaker is not None:
                        neigh.tiebreaker = self.tiebreakers[game.tiebreaker](self, other,
                                                                             neigh)
                    neigh.previously_visited_node = curr_node
                    hq.heappush(nodes_to_be_visited, neigh)
        game.max_times_visited = max_times_visited
        # start point of path restoration (here we begin from the end node of the shortest path found):
        return self.restore_path(other)

    def restore_path(self, other: 'Node'):
        node = other
        shortest_path = []
        # path restoring (here we get the reversed path):
        while node.previously_visited_node:
            shortest_path.append(node)
            node = node.previously_visited_node
        shortest_path.append(self)
        # returns the result:
        return shortest_path


# class representing an algo:
class Algorithm(ABC):
    def __init__(self, name: str, game: Lastar):
        # info:
        self._name = name
        # visual:
        self._icon = None
        self._menu = None
        # visualization:
        self._triangle_shape_list = arcade.ShapeElementList()  # <<-- for more comprehensive path visualization
        self._arrow_shape_list = arcade.ShapeElementList()  # <<-- for more comprehensive algorithm's visualization
        # important nodes:
        self._start_node = None
        self._end_node = None
        # path:
        self._path = None
        self._path_index = 0
        # iterations and time:
        self._iterations = 0
        self._time_elapsed_ms = 0
        # game-link:
        self._game = game

    def set_icon(self, icon):
        self._icon = icon

    def set_menu(self, menu):
        self._menu = menu

    def refresh_important_nodes(self, start: Node, end: Node):
        self._start_node = start
        self._end_node = end

    @abstractmethod
    def prepare(self):
        ...

    def path_up(self):
        if (path_node := self._path[self._path_index]).type not in [NodeType.START_NODE, NodeType.END_NODE]:
            path_node.type = NodeType.PATH_NODE
            path_node.update_sprite_colour()
        # arrows:
        p = -self._path[self._path_index + 1].x + self._path[self._path_index].x, \
            -self._path[self._path_index + 1].y + self._path[self._path_index].y
        p1, p2, p3 = self.get_triangle(self._path[self._path_index + 1], p)
        triangle_shape = arcade.create_triangles_filled_with_colors(
            [p1, p2, p3],
            [arcade.color.WHITE, arcade.color.RED, arcade.color.RED])
        self._triangle_shape_list.append(triangle_shape)
        # line arrows removing:
        node = self._path[self._path_index + 1]
        # if self.inter_types[2] == InterType.PRESSED:
        if node not in [self._start_node, self._end_node]:
            node.remove_arrow_from_shape_list(self)

    # path-recovering process for interactive a_star:
    def recover_path(self):
        # start point of path restoration (here we begin from the end node of the shortest path found):
        node = self._end_node
        shortest_path = []
        # path restoring (here we get the reversed path):
        while node.previously_visited_node:
            shortest_path.append(node)
            node = node.previously_visited_node
        shortest_path.append(self._start_node)
        # returns the result:
        return shortest_path

    def path_down(self):
        if (path_node := self._path[self._path_index]).type not in [NodeType.START_NODE, NodeType.END_NODE]:
            path_node.type = NodeType.VISITED_NODE
            path_node.update_sprite_colour()
            # arrows:
        self._triangle_shape_list.remove(self._triangle_shape_list[self._path_index - 1])
        # line arrows restoring:
        if path_node not in [self._start_node, self._end_node]:
            path_node.append_arrow(self._game)

    @abstractmethod
    def algo_up(self):
        ...

    @abstractmethod
    def algo_down(self):
        ...

    @abstractmethod
    def full_algo(self):
        ...

    # triangle points getting:
    def get_triangle(self, node: 'Node', point: tuple[int, int]):
        scaled_point = point[0] * (self._game.tile_size // 2 - 2), point[1] * (self._game.tile_size // 2 - 2)
        deltas = (
            (
                scaled_point[0] - scaled_point[1],
                scaled_point[0] + scaled_point[1]
            ),
            (
                scaled_point[0] + scaled_point[1],
                -scaled_point[0] + scaled_point[1]
            )
        )
        cx, cy = 5 + node.x * self._game.tile_size + self._game.tile_size / 2, \
                 5 + node.y * self._game.tile_size + self._game.tile_size / 2
        return (cx, cy), (cx + deltas[0][0], cy + deltas[0][1]), (cx + deltas[1][0], cy + deltas[1][1])


class Astar(Algorithm):

    def __init__(self, game):
        super().__init__('Astar', game)
        # visualization (made in super __init__()):
        # self._triangle_shape_list = arcade.ShapeElementList()  # <<-- for more comprehensive path visualization
        # self._arrow_shape_list = arcade.ShapeElementList()  # <<-- for more comprehensive algorithm's visualization
        # path (made in super __init__()):
        # self._path = None
        # self._path_index = 0
        # a_star important pars:
        # 1. important nodes (made in super __init__()):
        # self._start_node = None
        # self._end_node = None
        # 2. a_star_settings:
        self._heuristic = 0
        self._heuristic_names = {0: 'MANHATTAN', 1: 'EUCLIDIAN', 2: 'MAX_DELTA', 3: 'DIJKSTRA'}
        self._tiebreaker = None
        self._tiebreaker_names = {0: 'VECTOR_CROSS', 1: 'COORDINATES'}
        self._greedy_ind = None  # is algorithm greedy?
        self._greedy_names = {0: 'IS_GREEDY'}
        # 3. visiting:
        self._nodes_to_be_visited = []
        self._nodes_visited = {}
        # 4. iterations and time (made in super __init__()):
        # self._iterations = 0
        # self._time_elapsed_ms = 0
        # 5. interactive a_star pars:
        self._curr_node_dict = {}
        self._max_times_visited_dict = {0: 0}
        self._neighs_added_to_heap_dict = {}

    def prepare(self):
        # heap:
        self._nodes_to_be_visited = [self._start_node]
        hq.heapify(self._nodes_to_be_visited)
        # heur/cost:
        self._start_node.g = 0
        # transmitting the greedy flag to the Node class: TODO: fix this strange doing <<--
        Node.IS_GREEDY = False if self._greedy_ind is None else True
        # important pars and dicts:
        self._iterations = 0
        self._neighs_added_to_heap_dict = {0: [self._start_node]}
        self._curr_node_dict = {0: None}
        self._max_times_visited_dict = {0: 0}
        # path nullifying:
        self._path = None
        # lockers locating: TODO: should be relocated to Menu __init__() logic:
        for area in self._menu.areas:
            area.unlock()
        # SETTINGS menu should be closed during the algo's interactive phase!!!
        # arrows list renewal:
        self._arrow_shape_list = arcade.ShapeElementList()

    def algo_up(self):
        if self._iterations == 0:
            self._nodes_to_be_visited = [self._start_node]
        self._neighs_added_to_heap_dict[self._iterations + 1] = []  # memoization
        # popping out the most priority node for a_star from the heap:
        self._curr_node_dict[self._iterations + 1] = hq.heappop(self._nodes_to_be_visited)  # + memoization
        curr_node = self._curr_node_dict[self._iterations + 1]
        if self._iterations > 0 and curr_node != self._end_node:
            curr_node.type = NodeType.CURRENT_NODE
            curr_node.update_sprite_colour()
        curr_node.times_visited += 1
        if self._iterations > 1:
            if (prev_node := self._curr_node_dict[self._iterations]).type not in [NodeType.END_NODE,
                                                                                  NodeType.TWICE_VISITED]:
                prev_node.type = NodeType.VISITED_NODE
                prev_node.update_sprite_colour()
        self._max_times_visited_dict[self._iterations + 1] = max(self._max_times_visited_dict[self._iterations],
                                                                 # memoization
                                                                 curr_node.times_visited)
        # memoization for correct movement back:
        if curr_node in self._nodes_visited.keys():
            self._nodes_visited[curr_node] += 1
            curr_node.type = NodeType.TWICE_VISITED
            curr_node.update_sprite_colour()
        else:
            self._nodes_visited[curr_node] = 1
        # base case of finding the shortest path:
        if curr_node == self._end_node:
            self._path = self.recover_path()
        # next step:
        # we can search for neighs on the fly or use precalculated sets (outdated):
        for neigh in curr_node.get_neighs(self._game, [NodeType.WALL]):  # getting all the neighs 'on the fly;
            if neigh.g > curr_node.g + neigh.val:
                # memoization for further 'undoing':
                self._neighs_added_to_heap_dict[self._iterations + 1].append(
                    neigh.smart_copy(
                        [
                            'g',
                            'h',
                            'tiebreaker',
                            'type',
                            'previously_visited_node'
                        ]
                    )
                )
                # cost and heuristic computing:
                neigh.g = curr_node.g + neigh.val
                neigh.h = neigh.heuristics[self._heuristic](neigh, self._end_node)
                # tie-breaking:
                if self._tiebreaker is not None:
                    neigh.tiebreaker = self._start_node.tiebreakers[self._tiebreaker](
                        self._start_node, self._end_node, neigh)
                # previous visited node memoization for further path-recovering process:
                neigh.previously_visited_node = curr_node
                if neigh.type not in [NodeType.START_NODE, NodeType.END_NODE]:  # neigh not in self.nodes_visited and
                    neigh.type = NodeType.NEIGH
                    neigh.update_sprite_colour()
                    arrow = self._game.renderer.create_line_arrow(neigh, (neigh.x - curr_node.x, neigh.y - curr_node.y),
                                                                  self._game)
                    # here the arrow rotates (re-estimating of neigh g-cost):
                    if neigh.arrow_shape is not None:
                        neigh.remove_arrow(self._game)
                    neigh.arrow_shape = arrow
                    neigh.append_arrow(self._game)
                # adding all the valid neighs to the priority heap:
                hq.heappush(self._nodes_to_be_visited, neigh)
        # incrementation:
        self._iterations += 1

    def algo_down(self):
        # getting the previous current node from memo table:
        curr_node = self._curr_node_dict[self._iterations]
        if self._iterations > 1:  # stop condition for preventing the border case errors
            # times visited counter and colour 'backtracking':
            curr_node.times_visited -= 1
            if curr_node.times_visited == 0:
                if curr_node.type != NodeType.END_NODE:
                    curr_node.type = NodeType.NEIGH
                    curr_node.update_sprite_colour()
            elif curr_node.times_visited == 1:
                curr_node.type = NodeType.VISITED_NODE
                curr_node.update_sprite_colour()
            else:
                curr_node.type = NodeType.TWICE_VISITED
                curr_node.update_sprite_colour()
            if self._iterations > 2:
                self._curr_node_dict[self._iterations - 1].type = NodeType.CURRENT_NODE
                self._curr_node_dict[self._iterations - 1].update_sprite_colour()
        if self._iterations > 0:
            # removing the current node from nodes visited:
            if self._nodes_visited[curr_node] > 1:
                self._nodes_visited[curr_node] -= 1
            else:
                self._nodes_visited.pop(curr_node)
            # removing the neighs added from the heap:
            for neigh in self._neighs_added_to_heap_dict[self._iterations]:
                y, x = neigh.y, neigh.x
                node = self._game.grid[y][x]
                Node.aux_equal_flag = True
                self.remove_from_heapq(self._nodes_to_be_visited, self._nodes_to_be_visited.index(node))
                Node.aux_equal_flag = False
                node.smart_restore(
                    neigh,
                    [
                        'g',
                        'h',
                        'tiebreaker',
                        'type',
                        'previously_visited_node'
                    ]
                )
                if node.type not in [NodeType.START_NODE, NodeType.END_NODE]:
                    if node.arrow_shape is not None:
                        node.remove_arrow(self._game)
                if node.type == NodeType.NEIGH:
                    # here the arrow rotates backwards:
                    arrow = self._game.renderer.create_line_arrow(node, (
                        node.x - node.previously_visited_node.x, node.y - node.previously_visited_node.y), self.game)
                    node.arrow_shape = arrow
                    node.append_arrow(self._game)
            # adding current node (popped out at the current iteration) to the heap:
            hq.heappush(self._nodes_to_be_visited, curr_node)
            # iteration steps back:
            self._iterations -= 1

    @staticmethod
    def remove_from_heapq(heap, ind: int):
        heap[ind] = heap[-1]
        heap.pop()
        if ind < len(heap):
            # as far as it is known, possible to copy the source code from the heapq module... but how to do that?..
            Lastar.siftup(heap, ind)
            Lastar.siftdown(heap, 0, ind)

    # source code from: https://github.com/python/cpython/blob/main/Lib/heapq.py
    @staticmethod
    def siftdown(heap, start_pos, pos):
        new_item = heap[pos]
        # Follow the path to the root, moving parents down until finding a place
        # new item fits.
        while pos > start_pos:
            parent_pos = (pos - 1) >> 1
            parent = heap[parent_pos]
            if new_item < parent:
                heap[pos] = parent
                pos = parent_pos
                continue
            break
        heap[pos] = new_item

    # source code from: https://github.com/python/cpython/blob/main/Lib/heapq.py
    @staticmethod
    def siftup(heap, pos):
        end_pos = len(heap)
        start_pos = pos
        new_item = heap[pos]
        # Bubble up the smaller child until hitting a leaf.
        child_pos = 2 * pos + 1  # leftmost child position
        while child_pos < end_pos:
            # Set child pos to index of smaller child.
            right_pos = child_pos + 1
            if right_pos < end_pos and not heap[child_pos] < heap[right_pos]:
                child_pos = right_pos
            # Move the smaller child up.
            heap[pos] = heap[child_pos]
            pos = child_pos
            child_pos = 2 * pos + 1
        # The leaf at pos is empty now. Put new item there, and bubble it up
        # to its final resting place (by sifting its parents down).
        heap[pos] = new_item
        Lastar.siftdown(heap, start_pos, pos)

    def full_algo(self):
        # False if game.greedy_ind is None else True
        self._nodes_to_be_visited = [self._start_node]
        self._start_node.g = 0
        hq.heapify(self._nodes_to_be_visited)
        max_times_visited = 0
        # the main cycle:
        while self._nodes_to_be_visited:
            self._game.iterations += 1
            curr_node = hq.heappop(self._nodes_to_be_visited)
            if curr_node not in [self._start_node, self._end_node]:
                curr_node.type = NodeType.VISITED_NODE
                curr_node.update_sprite_colour()
            curr_node.times_visited += 1
            max_times_visited = max(max_times_visited, curr_node.times_visited)
            self._game.nodes_visited[curr_node] = 1
            # base case of finding the shortest path:
            if curr_node == self._end_node:
                break
            # next step:
            for neigh in curr_node.get_neighs(self._game, [NodeType.WALL]):
                if neigh.g > curr_node.g + neigh.val:
                    neigh.g = curr_node.g + neigh.val
                    neigh.h = neigh.heuristics[self._game.heuristic](neigh, self._end_node)
                    if self._game.tiebreaker is not None:
                        neigh.tiebreaker = self._start_node.tiebreakers[self._game.tiebreaker](self, self._end_node,
                                                                                               neigh)
                    neigh.previously_visited_node = curr_node
                    hq.heappush(self._nodes_to_be_visited, neigh)
        self._game.max_times_visited = max_times_visited


class WaveLee(Algorithm):
    def __init__(self, game: Lastar):
        super().__init__('WaveLee', game)
        # wave lee algo's important pars:
        self._front_wave_lee = None
        self._next_wave_lee = None
        self._fronts_dict = None
        self._iterations = 0

    def prepare(self):
        # starting attributes' values:
        self._front_wave_lee = []
        self._next_wave_lee = [self._start_node]
        # self._start_node.val = 1  # node.val must not be changed during the algo's interactive phase!!!
        self._iterations = 0
        self._fronts_dict = {}

    def algo_up(self):
        self._iterations += 1
        self._front_wave_lee = self._next_wave_lee[:]
        self._fronts_dict[self._iterations] = self._front_wave_lee
        self._next_wave_lee = []
        for curr_node in self._front_wave_lee:
            # curr_node.val = self._iterations
            if curr_node not in [self._end_node, self._start_node]:
                curr_node.type = NodeType.VISITED_NODE
                curr_node.update_sprite_colour()
            if curr_node == self._end_node:
                # self._end_node.val = self._iterations
                self._path = self.recover_path()
                break
            for neigh in curr_node.get_neighs(self, [NodeType.START_NODE, NodeType.WALL, NodeType.VISITED_NODE]):
                if neigh.type == NodeType.EMPTY:  # it is equivalent to if neigh.val == 1, TODO: decide if is it needed???
                    if neigh not in self._next_wave_lee:
                        if neigh != self._end_node:
                            neigh.type = NodeType.NEIGH
                            neigh.update_sprite_colour()
                            arrow = self._game.renderer.create_line_arrow(
                                neigh,
                                (neigh.x - curr_node.x, neigh.y - curr_node.y),
                                self._game
                            )
                            neigh.arrow_shape = arrow
                            neigh.append_arrow(self)
                        self._next_wave_lee.append(neigh)
                        neigh.previously_visited_node = curr_node

    def algo_down(self):
        # possibility check of wave_lee's stepping back:
        if self._iterations > 0:
            # decrementation:
            self._iterations -= 1
            # neighs have become EMPTY ones:
            for neigh in self._next_wave_lee:
                if neigh not in [self._start_node, self._end_node]:
                    neigh.type = NodeType.EMPTY
                    neigh.update_sprite_colour()
                    neigh.remove_arrow(self)
            if self._iterations != 0:
                # the front nodes have become NEIGHS:
                for node in self._front_wave_lee:
                    if node not in [self._start_node, self._end_node]:
                        node.type = NodeType.NEIGH
                        node.update_sprite_colour()
                # current and next fronts stepping back:
                self._next_wave_lee = self._front_wave_lee[:]
                self._front_wave_lee = self._fronts_dict[self._iterations]
            else:
                # the starting point:
                self._next_wave_lee = [self._start_node]
                self._front_wave_lee = []

    def full_algo(self):
        # other.get_neighs(game)  # Why is it necessary???
        front_wave = {self._start_node}
        iteration = 0
        # wave-spreading:
        while front_wave:
            iteration += 1
            new_front_wave = set()
            for front_node in front_wave:
                # front_node.val = iteration
                if front_node not in [self._start_node, self._end_node]:
                    front_node.type = NodeType.VISITED_NODE
                    front_node.update_sprite_colour()
                if front_node == self._end_node:
                    return self.recover_path()
                for front_neigh in front_node.get_neighs(
                        self._game,
                        [NodeType.START_NODE, NodeType.VISITED_NODE, NodeType.WALL]):
                    if front_neigh not in new_front_wave:
                        front_neigh.previously_visited_node = front_node
                        new_front_wave.add(front_neigh)
            front_wave = set() | new_front_wave
        return []


class BfsDfs(Algorithm):
    def __init__(self, game: Lastar, is_bfs=True):
        super().__init__('Bfs/Dfs', game)
        # important algo's attributes:
        self._queue = None
        self._is_bfs = is_bfs
        # dicts:
        self._curr_node_dict = {}
        self._neighs_added_to_heap_dict = {}

    @property
    def bfs_dfs_ind(self):
        return 0 if self._is_bfs else 1

    def prepare(self):
        self._queue = deque()
        self._queue.append(self._start_node)
        self._iterations = 0
        # dicts:
        self._curr_node_dict = {0: None}

    def algo_up(self):
        # one bfs step up:
        self._iterations += 0
        curr_node = self._queue.pop()
        self._curr_node_dict[self._iterations + 1] = curr_node
        if self._iterations > 0:
            if curr_node.type != NodeType.END_NODE:
                curr_node.type = NodeType.CURRENT_NODE
                curr_node.update_sprite_colour()
        if self._iterations > 1:
            self._curr_node_dict[self._iterations].type = NodeType.VISITED_NODE
            self._curr_node_dict[self._iterations].update_sprite_colour()
        curr_node.times_visited += 1
        if curr_node == self._end_node:
            self._path = self.recover_path()
        self._neighs_added_to_heap_dict[self._iterations + 1] = set()
        for neigh in curr_node.get_neighs(self, [NodeType.START_NODE, NodeType.VISITED_NODE, NodeType.WALL] + (
                [NodeType.NEIGH] if self.bfs_dfs_ind == 0 else [])):
            if neigh.type != NodeType.END_NODE:
                # at first memoization for further 'undoing':
                self._neighs_added_to_heap_dict[self._iterations + 1].add(neigh.aux_copy())
                # then changing neigh's pars:
                neigh.type = NodeType.NEIGH
                neigh.update_sprite_colour()
                arrow = self._game.renderer.create_line_arrow(neigh, (neigh.x - curr_node.x, neigh.y - curr_node.y),
                                                              self)
                if neigh.arrow_shape is not None:
                    neigh.remove_arrow(self)
                neigh.arrow_shape = arrow
                neigh.append_arrow(self)
            neigh.previously_visited_node = curr_node
            # BFS:
            if self.bfs_dfs_ind == 0:
                self._queue.appendleft(neigh)
            # DFS:
            else:
                self._queue.append(neigh)
        self._iterations += 1

    def algo_down(self):
        if self._iterations > 0:
            # now the neighs of current node should become EMPTY ones:
            for neigh in self._neighs_added_to_heap_dict[self._iterations]:
                # TODO: neigh type restoring needed!!!
                y, x = neigh.y, neigh.x
                node = self._game.grid[y][x]
                node.restore(neigh)
                if node not in [self._start_node, self._end_node]:
                    node.remove_arrow(self._game)
                if node.type == NodeType.NEIGH:
                    # here the arrow rotates backwards:
                    arrow = self._game.renderer.create_line_arrow(
                        node,
                        (
                            node.x - self._curr_node_dict[self._iterations].x,
                            node.y - self._curr_node_dict[self._iterations].y
                        ),
                        self._game
                    )
                    node.arrow_shape = arrow
                    node.append_arrow(self._game)
                    # deque changing:
                    # BFS:
                    if self.bfs_dfs_ind == 0:
                        self._queue.popleft()
                    # DFS:
                    else:
                        self._queue.pop()
            # current node has become the NEIGH:
            curr_node = self._curr_node_dict[self._iterations]
            if curr_node not in [self._start_node, self._end_node]:
                curr_node.type = NodeType.NEIGH
                curr_node.update_sprite_colour()
            # adding back to the deque:
            # BFS & DFS:
            self._queue.append(curr_node)
            if self._iterations > 1:
                # previous step current node has become the current step current node:
                prev_node = self._curr_node_dict[self._iterations - 1]
                if prev_node not in [self._start_node, self._end_node]:
                    prev_node.type = NodeType.CURRENT_NODE
                    prev_node.update_sprite_colour()
            # step back:
            self._iterations -= 1

    def full_algo(self):
        queue = deque()
        queue.append(self._start_node)
        while queue:
            self._iterations += 1
            current_node = queue.pop()
            if current_node.type not in [NodeType.START_NODE, NodeType.END_NODE]:
                current_node.type = NodeType.VISITED_NODE
                current_node.update_sprite_colour()
                current_node.times_visited += 1
            if current_node == self._end_node:
                return self.recover_path()
            for neigh in current_node.get_neighs(self._game,
                                                 [NodeType.START_NODE, NodeType.VISITED_NODE, NodeType.WALL] + (
                                                         [NodeType.NEIGH] if self._game.bfs_dfs_ind == 0 else [])):
                if neigh.type != NodeType.END_NODE:
                    neigh.type = NodeType.NEIGH
                    neigh.update_sprite_colour()
                neigh.previously_visited_node = current_node
                # BFS:
                if self.bfs_dfs_ind == 0:
                    queue.appendleft(neigh)
                # DFS:
                else:
                    queue.append(neigh)


class Menu(ABC):
    def __init__(self, icon_linked: 'Icon'):
        self._icon_linked = icon_linked
        self._areas = []

    @property
    def areas(self):
        return self._areas

    def append_area(self, area: 'Area'):
        self._areas.append(area)

    def setup(self):
        for area in self._areas:
            area.setup()

    def on_press(self, x, y):
        for area in self._areas:
            area.on_press(x, y)

    def draw(self):
        if self._icon_linked.inter_type == InterType.PRESSED:
            for area in self._areas:
                area.draw()


class Area(ABC):
    def __init__(self, cx, cy, delta, sq_size, sq_line_w, header: str, fields: dict[int, str]):
        # options, that need to be added: 1. multiple_choice: bool, 2. no_choice: bool
        self._cx = cx
        self._cy = cy
        self._delta = delta
        self._sq_size = sq_size
        self._sq_line_w = sq_line_w
        self._header = header
        self._fields = fields
        # choosing:
        self._field_chosen = None  # can be None in some cases or keep several values simultaneously
        # locking:
        self._is_locked = False
        # presets:
        self._header_text = None
        self._field_texts = []
        self._rectangle_shapes = arcade.ShapeElementList()

    def lock(self):
        self._is_locked = True

    def unlock(self):
        self._is_locked = False

    # presets:
    @abstractmethod
    def setup(self):
        self._header_text = arcade.Text(f'{self._header}: ', self._cx, self._cy, arcade.color.BLACK, bold=True)
        for i in range(len(self._fields)):
            self._rectangle_shapes.append(
                arcade.create_rectangle_outline(
                    self._cx + 10,
                    self._cy - 30 - self._delta * i,
                    self._sq_size, self._sq_size,
                    arcade.color.BLACK, self._sq_line_w
                )
            )
            self._field_texts.append(
                arcade.Text(
                    f'{self._fields[i]}',
                    self._cx + 10 + self._sq_size + 2 * self._sq_line_w,
                    self._cy - 30 - self._delta * i - 6,
                    arcade.color.BLACK, bold=True
                )
            )

    def on_press(self):
        ...

    @abstractmethod
    # on every frame:
    def draw(self):
        self._header_text.draw()
        self._rectangle_shapes.draw()

        for text in self._field_texts:
            text.draw()

        if self._field_chosen is not None:
            arcade.draw_rectangle_filled(self._cx + 10, self._cy - 30 - self._delta * self._field_chosen,
                                         14, 14, arcade.color.BLACK)
        # heuristics lock:
        if self._is_locked:
            if self._field_chosen is not None:
                self.draw_lock(self._cx + 10, self._cy - 30 - self._delta * self._field_chosen)
            for i in range(len(self._fields)):
                if self._field_chosen is None or i != self._field_chosen:
                    self.draw_cross(self._cx + 10, self._cy - 30 - self._delta * i)

    # draws a lock for right window part: 36 366 98 989
    @staticmethod
    def draw_lock(center_x: int, center_y: int):
        arcade.draw_rectangle_filled(center_x, center_y, 14, 14, arcade.color.RED)
        arcade.draw_rectangle_outline(center_x, center_y + 7, 8.4, 16.8, arcade.color.RED, border_width=2)

    # draws the cross of forbiddance:
    @staticmethod
    def draw_cross(center_x: int, center_y: int):
        arcade.draw_line(center_x - 9, center_y + 9, center_x + 9, center_y - 9, arcade.color.BLACK, line_width=2)
        arcade.draw_line(center_x + 9, center_y + 9, center_x - 9,
                         center_y - 9, arcade.color.BLACK, line_width=2)


# class for design element:
class Icon(ABC):
    def __init__(self, cx, cy):
        self._cx = cx
        self._cy = cy
        self._incrementer = 0
        self._inter_type = InterType.NONE
        self._vertices = []

    # on initialization:
    @abstractmethod
    def setup(self):
        ...

    @abstractmethod
    def update(self):
        ...

    # renders the icon:
    @abstractmethod
    def draw(self):
        ...

    # implements the icon's behaviour in Astar on_press/on_motion methods:
    @abstractmethod
    def on_motion(self, x, y):
        ...

    @abstractmethod
    def on_press(self, x, y):
        ...

    @abstractmethod
    def on_drug(self, x, y):
        ...

    # helpful auxiliary methods:
    @staticmethod
    def is_point_in_square(cx, cy, size, x, y):
        return cx - size / 2 <= x <= cx + size / 2 and cy - size / 2 <= y <= cy + size / 2

    @staticmethod
    def is_point_in_circle(cx, cy, r, x, y):
        return (cx - x) ** 2 + (cy - y) ** 2 <= r ** 2

    @property
    def inter_type(self):
        return self._inter_type


class PlayButton(Icon):
    DELTAS = [0.5, 0.015]  # pixels/radians

    def __init__(self, cx, cy, r, line_w):
        super().__init__(cx, cy)
        self._incrementer = [0, 0]
        self._r = r
        self._line_w = line_w
        self._multiplier = 1
        # self._counter = 1

    def setup(self):
        pass

    def update(self):
        if self._inter_type == InterType.PRESSED:
            if self._multiplier == 1:
                if self._incrementer[0] < self._r // 2:
                    self._incrementer[0] += self.DELTAS[0]
            else:
                if 0 <= self._incrementer[0]:
                    self._incrementer[0] -= self.DELTAS[0]
        else:
            if 0 <= self._incrementer[0]:
                self._incrementer[0] -= self.DELTAS[0]

        if self._inter_type != InterType.NONE:
            self._incrementer[1] += 0.015

    def draw(self):
        #
        #     self.h_increment +g= 0.5
        dh = self._r / math.sqrt(3)
        delta_line_w = 0 if self._inter_type == InterType.NONE else 1
        arcade.draw_circle_outline(self._cx, self._cy, self._r + self._line_w + 1, arcade.color.BLACK,
                                   self._line_w + 2 * delta_line_w)

        left_ratio, right_ratio = 1 - 1 / math.sqrt(3), 2 / math.sqrt(3) - 1

        self._vertices = [
            (self._cx + dh - right_ratio * self._incrementer[0], self._cy + self._incrementer[0]),
            (self._cx - dh / 2 - left_ratio * self._incrementer[0], self._cy + self._r // 2),
            (self._cx - dh / 2 - left_ratio * self._incrementer[0], self._cy - self._r // 2),
            (self._cx + dh - right_ratio * self._incrementer[0], self._cy - self._incrementer[0])
        ]

        if self._inter_type == InterType.PRESSED:
            arcade.draw_polygon_filled(self._vertices, arcade.color.RED)

        arcade.draw_polygon_outline(self._vertices, arcade.color.BLACK, self._line_w + delta_line_w)

        if self._inter_type != InterType.NONE:
            self._draw_dashed_line_circle(12)

    def _draw_dashed_line_circle(self, q, clockwise=True):
        angular_size = math.pi / q
        _r = self._r - self._r / 4 + 3
        angle = self._incrementer[1] if clockwise else -self._incrementer[1]  # in radians
        for i in range(q):
            _a, a_ = (angle - angular_size / 2), (angle + angular_size / 2)
            _rx, _ry = _r * math.cos(_a), _r * math.sin(_a)
            rx_, ry_ = _r * math.cos(a_), _r * math.sin(a_)
            arcade.draw_line(self._cx + _rx, self._cy + _ry, self._cx + rx_, self._cy + ry_, arcade.color.BLACK,
                             self._line_w + (1 if self._inter_type == InterType.PRESSED else 0))
            angle += angular_size * 2

    def on_motion(self, x, y):
        if self._inter_type != InterType.PRESSED:
            if self.is_point_in_circle(self._cx, self._cy, self._r, x, y):
                self._inter_type = InterType.HOVERED
            else:
                self._inter_type = InterType.NONE
        else:
            if self.is_point_in_circle(self._cx, self._cy, self._r, x, y):
                self._multiplier = 1
            else:
                self._multiplier = -1

    def on_press(self, x, y):
        # if self._inter_type != InterType.PRESSED:
        if self.is_point_in_circle(self._cx, self._cy, self._r, x, y):
            if self._inter_type == InterType.PRESSED:
                self._inter_type = InterType.HOVERED
            elif self._inter_type == InterType.HOVERED:
                self._inter_type = InterType.PRESSED

    def on_drug(self, x, y):
        ...


class StepButton(Icon):
    DELTAS = [0.15, 0.1, 0.05]
    THRESHOLD = 8
    TICKS_THRESHOLD = 12
    MAGNITUDE = 3

    def __init__(self, cx, cy, a, dh, line_w, is_right=True):
        super().__init__(cx, cy)
        self._incrementer = [0, 0, 0, 0]  # horizontal movement, sin oscillating, blinking and ticks
        self._a = a
        self._dh = dh
        self._line_w = line_w  # TODO: MAY BE SHOULD BE INITIALIZED ONCE FOR THE ENTIRE GAME???
        self._is_right = is_right
        self._cycle_breaker = False  # TODO: SHOULD BE REWORKED!!! HOW SHOULD THE ALGO KNOW IF IT IS CHANGED???

    def setup(self):
        pass

    def update(self):
        # long press logic:
        if self._cycle_breaker:
            self._incrementer[3] += 1
            if self._incrementer[3] >= self.TICKS_THRESHOLD:
                if self._incrementer[0] <= self.THRESHOLD:
                    self._incrementer[0] += self.DELTAS[0]
                self._incrementer[1] = (self._incrementer[1] + self.DELTAS[1]) % 3
                self._incrementer[2] += self.DELTAS[2]
        else:
            if self._incrementer[0] > 0:
                self._incrementer[0] -= self.DELTAS[0]
            else:
                self._incrementer[0] = 0
            self._incrementer[1] = 0
            self._incrementer[2] = 0

    @property
    def multiplier(self):
        return 1 if self._is_right else -1

    @property
    def h(self):
        return self.multiplier * self._a / 3

    @property
    def length(self):
        return self.multiplier * self._dh

    @property
    def vertices(self):
        vertices = [
            (self._cx - self.h / 2, self._cy + self._a / 2),
            (self._cx - self.h / 2, self._cy + self._a / 2 + self._dh),
            (self._cx + self.h + self.length, self._cy),
            (self._cx - self.h / 2, self._cy - self._a / 2 - self._dh),
            (self._cx - self.h / 2, self._cy - self._a / 2),
            (self._cx + self.h, self._cy)
        ]
        self._vertices.append(vertices)
        return vertices

    def draw(self):
        # aux method that changes the figure sizes saving its shape by decreasing some pars:
        def decrease_stats():
            self._cx += self.multiplier * self._incrementer[0]
            self._a -= self._a / 4
            self._dh -= self._dh / 4
            self._cx += self.length + self.length - self.h

        self._vertices = []
        cx = self._cx
        cy = self._cy
        a = self._a
        dh = self._dh

        for i in range(3):
            if self._cycle_breaker:
                self._cy += self.MAGNITUDE * math.sin(i * math.pi / 2 + self._incrementer[2])

            vertices = self.vertices

            if self._incrementer[3] >= self.TICKS_THRESHOLD:
                if int(self._incrementer[1]) == i:
                    arcade.draw_polygon_filled(vertices, arcade.color.RED)
            elif self._inter_type == InterType.PRESSED:
                arcade.draw_polygon_filled(vertices, arcade.color.RED)
            arcade.draw_polygon_outline(vertices, arcade.color.BLACK,
                                        self._line_w + (0 if self._inter_type == InterType.NONE else 1))

            if i < 2:
                decrease_stats()

        self._cx = cx
        self._cy = cy
        self._a = a
        self._dh = dh

    # mouse:
    def on_motion(self, x, y):
        if self._inter_type != InterType.PRESSED:
            # print(f'length: {len(self._vertices)}')
            if len(self._vertices) == 3:
                # print(f'self._vertices: {self._vertices}')
                if reduce(lambda a, b: a or arcade.is_point_in_polygon(x, y, self._vertices[b]), list(range(3)), False):
                    self._inter_type = InterType.HOVERED
                else:
                    self._inter_type = InterType.NONE

    def on_press(self, x, y):
        if len(self._vertices) == 3:
            if reduce(lambda a, b: a or arcade.is_point_in_polygon(x, y, self._vertices[b]), list(range(3)), False):
                self._inter_type = InterType.PRESSED
                self._cycle_breaker = True

    def on_drug(self, x, y):
        pass

    def on_release(self, x, y):
        self._inter_type = InterType.NONE
        self._cycle_breaker = False
        self._incrementer[3] = 0

    # keys:
    def on_key_press(self):
        self._inter_type = InterType.PRESSED
        self._cycle_breaker = True

    def on_key_release(self):
        self._inter_type = InterType.NONE
        self._cycle_breaker = False
        self._incrementer[3] = 0


class Eraser(Icon):

    def __init__(self, cx, cy, h, w, r, line_w):
        super().__init__(cx, cy)
        self._h = h
        self._w = w
        self._r = r
        self._line_w = line_w
        self._centers = []

    def setup(self):
        self._vertices = [
            [self._cx - self._w / 2, self._cy + self._h / 2 + self._r],
            [self._cx + self._w / 2, self._cy + self._h / 2 + self._r],
            [self._cx + self._w / 2 + self._r, self._cy + self._h / 2],
            [self._cx + self._w / 2 + self._r, self._cy - self._h / 2],
            [self._cx + self._w / 2, self._cy - self._h / 2 - self._r],
            [self._cx - self._w / 2, self._cy - self._h / 2 - self._r],
            [self._cx - self._w / 2 - self._r, self._cy + self._h / 2],
            [self._cx - self._w / 2 - self._r, self._cy - self._h / 2]
        ]

        self._centers = [
            [self._cx + self._w / 2, self._cy + self._h / 2],
            [self._cx - self._w / 2, self._cy + self._h / 2],
            [self._cx - self._w / 2, self._cy - self._h / 2],
            [self._cx + self._w / 2, self._cy - self._h / 2]
        ]

    def update(self):
        pass

    def draw(self):
        line_w = self._line_w + (0 if self._inter_type == InterType.NONE else 1)

        if self._inter_type == InterType.PRESSED:
            arcade.draw_rectangle_filled(self._cx, self._cy, self._w, self._h + 2 * self._r, arcade.color.RED)
        arcade.draw_line(self._vertices[0][0], self._vertices[0][1], self._vertices[5][0], self._vertices[5][1],
                         arcade.color.BLACK, line_w)
        arcade.draw_line(self._vertices[1][0], self._vertices[1][1], self._vertices[4][0], self._vertices[4][1],
                         arcade.color.BLACK, line_w)

        for i in range(len(self._vertices) // 2):
            p1, p2 = self._vertices[2 * i: 2 * i + 2]
            arcade.draw_line(p1[0], p1[1], p2[0], p2[1], arcade.color.BLACK, line_w)

        for i, [x, y] in enumerate(self._centers):
            arcade.draw_arc_outline(x, y, 2 * self._r, 2 * self._r, arcade.color.BLACK, 90 * i, 90 * (i + 1),
                                    2 * line_w)

    def on_motion(self, x, y):
        if self._inter_type != InterType.PRESSED:
            if arcade.is_point_in_polygon(x, y, self._vertices):
                self._inter_type = InterType.HOVERED
            else:
                self._inter_type = InterType.NONE

    def on_press(self, x, y):
        if arcade.is_point_in_polygon(x, y, self._vertices):
            self._inter_type = InterType.PRESSED

    def on_release(self, x, y):
        self._inter_type = InterType.NONE

    def on_drug(self, x, y):
        pass


class Undo(Icon):
    def __init__(self, cx, cy, a, dh, r, line_w, is_right=False):
        super().__init__(cx, cy)
        self._a = a
        self._dh = dh
        self._r = r
        self._line_w = line_w
        self._is_right = is_right

    @property
    def multiplier(self):
        return -1 if self._is_right else 1

    @property
    def angles(self):
        return (90, 360) if self._is_right else (-180, 90)

    @property
    def line_w(self):
        return self._line_w + (0 if self._inter_type == InterType.NONE else 1)

    def setup(self):
        pass

    def update(self):
        pass

    def draw(self):
        start_angle, end_angle = self.angles

        arcade.draw_arc_outline(self._cx, self._cy, 2 * self._r, 2 * self._r, arcade.color.BLACK, start_angle,
                                end_angle, 2 * self.line_w)
        arcade.draw_arc_outline(self._cx, self._cy, 2 * (self._r + self._dh), 2 * (self._r + self._dh),
                                arcade.color.BLACK, start_angle, end_angle,
                                2 * self.line_w)
        arcade.draw_line(self._cx - self.multiplier * self._r, self._cy,
                         self._cx - self.multiplier * (self._r + self._dh), self._cy, arcade.color.BLACK, self.line_w)
        vs = self._vertices = [
            (self._cx - self.multiplier * self._a * math.sqrt(3) / 2, self._cy + self._r + self._dh / 2),
            (self._cx, self._cy + self._r + self._dh / 2 + self._a / 2),
            (self._cx, self._cy + self._r + self._dh / 2 - self._a / 2)
        ]

        if self._inter_type == InterType.PRESSED:
            arcade.draw_triangle_filled(vs[0][0], vs[0][1], vs[1][0], vs[1][1], vs[2][0], vs[2][1], arcade.color.RED)
        arcade.draw_triangle_outline(vs[0][0], vs[0][1], vs[1][0], vs[1][1], vs[2][0], vs[2][1], arcade.color.BLACK,
                                     self.line_w)

    def on_motion(self, x, y):
        if self._inter_type != InterType.PRESSED:
            if self.is_point_in_circle(self._cx, self._cy, self._r + self._dh, x, y) or arcade.is_point_in_polygon(x, y,
                                                                                                                   self._vertices):
                self._inter_type = InterType.HOVERED
            else:
                self._inter_type = InterType.NONE

    def on_press(self, x, y):
        if self.is_point_in_circle(self._cx, self._cy, self._r + self._dh, x, y) or arcade.is_point_in_polygon(x, y,
                                                                                                               self._vertices):
            self._inter_type = InterType.PRESSED

    def on_release(self, x, y):
        self._inter_type = InterType.NONE

    def on_drug(self, x, y):
        pass


class GearWheelButton(Icon):
    DELTA = 0.02

    def __init__(self, cx, cy, r, cog_size=8, multiplier=1.5, line_w=2, clockwise=True):
        super().__init__(cx, cy)
        self._r = r
        self._cog_size = cog_size
        self._multiplier = multiplier
        self._line_w = line_w
        self._clockwise = clockwise

    def setup(self):
        ...

    def update(self):
        self._incrementer += self.DELTA

    def draw(self):
        circumference = 2 * math.pi * self._r  # approximately if cog_size << radius
        angular_size = (2 * math.pi) * self._cog_size / circumference
        max_cogs_fit_in_the_gear_wheel = int(circumference / self._cog_size)
        cogs_q = max_cogs_fit_in_the_gear_wheel // 2
        fit_angular_size = (2 * math.pi - cogs_q * angular_size) / cogs_q
        angle = self._incrementer if self._clockwise else -self._incrementer  # in radians

        for i in range(cogs_q):
            # aux pars:
            _a, a_ = (angle - angular_size / 2), (angle + angular_size / 2)
            _rx, _ry, rx_, ry_ = self._r * math.cos(_a), self._r * math.sin(_a), self._r * math.cos(
                a_), self._r * math.sin(a_)
            _dx, _dy = self._cog_size * math.cos(_a), self._cog_size * math.sin(_a)
            dx_, dy_ = self._cog_size * math.cos(a_), self._cog_size * math.sin(a_)
            # polygon's points:
            self._vertices.append([self._cx + _rx, self._cy + _ry])
            self._vertices.append([self._cx + _rx + _dx, self._cy + _ry + _dy])
            self._vertices.append([self._cx + rx_ + dx_, self._cy + ry_ + dy_])
            self._vertices.append([self._cx + rx_, self._cy + ry_])
            # angle incrementation:
            angle += angular_size + fit_angular_size
        # upper gear wheel:
        # arcade.draw_polygon_filled(upper_vertices_list, arcade.color.PASTEL_GRAY)
        if self._inter_type == InterType.PRESSED:
            arcade.draw_polygon_filled(self._vertices, arcade.color.RED)
        arcade.draw_polygon_outline(self._vertices, arcade.color.BLACK,
                                    self._line_w + (0 if self._inter_type == InterType.NONE else 1))
        # hole:
        arcade.draw_circle_filled(self._cx, self._cy, 2 * (self._r - self._multiplier * self._cog_size),
                                  arcade.color.DUTCH_WHITE)
        arcade.draw_circle_outline(self._cx, self._cy, 2 * (self._r - self._multiplier * self._cog_size),
                                   arcade.color.BLACK,
                                   self._line_w + (0 if self._inter_type == InterType.NONE else 1))

    def on_motion(self, x, y):
        if self._inter_type != InterType.PRESSED:
            if arcade.is_point_in_polygon(x, y, self._vertices):
                self._inter_type = InterType.HOVERED
            else:
                self._inter_type = InterType.NONE

    def on_press(self, x, y):
        if arcade.is_point_in_polygon(x, y, self._vertices):
            if self._inter_type == InterType.PRESSED:
                self._inter_type = InterType.HOVERED
            else:
                self._inter_type = InterType.PRESSED

    def on_drug(self, x, y):
        ...


class AstarIcon(Icon):
    DELTA = 0.05

    def __init__(self, cx, cy, size_w, size_h, line_w=2, clockwise=True):
        super().__init__(cx, cy)
        self._size_w = size_w
        self._size_h = size_h
        self._line_w = line_w
        self._clockwise = clockwise

    def setup(self):
        pass

    def update(self):
        if self._inter_type == InterType.HOVERED:
            self._incrementer += self.DELTA

    def draw(self):
        # drawing A:
        self.draw_a(self._cx, self._cy, self._size_w, self._size_h, self._size_w / 3, self._line_w)
        # Star spinning around A:
        self.draw_star(self._cx + self._size_h / 2 + self._size_h / 3, self._cy + self._size_h, r=self._size_h / 4,
                       line_w=self._line_w, clockwise=self._clockwise)

    # draws a spinning star:
    def draw_star(self, cx, cy, vertices=5, r=32, line_w=2, clockwise=True):
        delta_angle = 2 * math.pi / vertices
        d = vertices // 2
        angle = self._incrementer if clockwise else -self._incrementer  # in radians
        for i in range(vertices):
            da = d * delta_angle
            arcade.draw_line(cx + r * math.cos(angle),
                             cy + r * math.sin(angle),
                             cx + r * math.cos(angle + da),
                             cy + r * math.sin(angle + da),
                             arcade.color.BLACK, line_w)
            angle += da

    # draws 'A' letter:
    def draw_a(self, cx, cy, length, height, a_w, line_w):
        upper_hypot = math.sqrt(length ** 2 + height ** 2)
        cos, sin = height / upper_hypot, length / upper_hypot
        line_w_hour_projection = a_w / cos
        dh = a_w / sin
        delta = (height - a_w - dh) / 2
        k, upper_k = (delta - a_w / 2) * length / height, (delta + a_w / 2) * length / height

        # a_outer_points -> self._vertices[0], a_inner_points -> self._vertices[1]
        self._vertices.append(
            [
                [cx, cy],
                [cx + length, cy + height],
                [cx + 2 * length, cy],
                [cx + 2 * length - line_w_hour_projection, cy],
                [cx + 2 * length - line_w_hour_projection - k, cy + delta - a_w / 2],
                [cx + k + line_w_hour_projection, cy + delta - a_w / 2],
                [cx + line_w_hour_projection, cy]
            ]
        )

        self._vertices.append(
            [
                [cx + length, cy + height - dh],
                [cx + 2 * length - line_w_hour_projection - upper_k, cy + delta + a_w / 2],
                [cx + line_w_hour_projection + upper_k, cy + delta + a_w / 2]
            ]
        )

        if self._inter_type == InterType.NONE:
            arcade.draw_polygon_outline(self._vertices[0], arcade.color.BLACK, line_w)
            arcade.draw_polygon_outline(self._vertices[1], arcade.color.BLACK, line_w)
        elif self._inter_type == InterType.HOVERED:
            arcade.draw_polygon_outline(self._vertices[0], arcade.color.BLACK, line_w + 1)
            arcade.draw_polygon_outline(self._vertices[1], arcade.color.BLACK, line_w + 1)
        else:
            arcade.draw_polygon_filled(self._vertices[0], arcade.color.RED)
            arcade.draw_polygon_outline(self._vertices[0], arcade.color.BLACK, line_w + 1)
            arcade.draw_polygon_filled(self._vertices[1], arcade.color.DUTCH_WHITE)
            arcade.draw_polygon_outline(self._vertices[1], arcade.color.BLACK, line_w + 1)

    def on_motion(self, x, y):
        if self._inter_type != InterType.PRESSED:
            if arcade.is_point_in_polygon(x, y, self._vertices[0]):
                self._inter_type = InterType.HOVERED
            else:
                self._inter_type = InterType.NONE

    def on_press(self, x, y):
        if arcade.is_point_in_polygon(x, y, self._vertices[0]):
            if self._inter_type == InterType.PRESSED:
                self._inter_type = InterType.HOVERED
            else:
                self._inter_type = InterType.PRESSED

    def on_drug(self, x, y):
        pass


class Waves(Icon):
    DELTA = 0.25

    def __init__(self, cx, cy, size=32, waves_q=5, line_w=2):
        super().__init__(cx, cy)
        self._size = size
        self._waves_q = waves_q
        self._line_w = line_w

    def setup(self):
        pass

    def update(self):
        self._incrementer += self.DELTA

    def draw(self):
        ds = self._size / self._waves_q
        s_list = sorted([(i * ds + self._incrementer) % self._size for i in range(self._waves_q)], reverse=True)
        for i, curr_s in enumerate(s_list):
            if self._inter_type == InterType.PRESSED:
                arcade.draw_circle_filled(self._cx, self._cy, curr_s, arcade.color.RED if i % 2 == 0 else arcade.color.DUTCH_WHITE)
            arcade.draw_circle_outline(self._cx, self._cy, curr_s, arcade.color.BLACK,
                                       self._line_w + (0 if self._inter_type == InterType.NONE else 1))

    def on_motion(self, x, y):
        if self._inter_type != InterType.PRESSED:
            if self.is_point_in_circle(self._cx, self._cy, self._size, x, y):
                self._inter_type = InterType.HOVERED
            else:
                self._inter_type = InterType.NONE

    def on_press(self, x, y):
        if self.is_point_in_circle(self._cx, self._cy, self._size, x, y):
            if self._inter_type == InterType.PRESSED:
                self._inter_type = InterType.HOVERED
            else:
                self._inter_type = InterType.PRESSED

    def on_drug(self, x, y):
        pass


class BfsDfsIcon(Icon):
    DELTA = 0.15

    def __init__(self, cx, cy, size, line_w):
        super().__init__(cx, cy)
        self._size = size
        self._line_w = line_w

    def setup(self):
        pass

    def update(self):
        self._incrementer += self.DELTA

    def draw(self):
        # filling:
        if self._inter_type == InterType.PRESSED:
            arcade.draw_rectangle_filled(self._cx, self._cy, self._size, self._size, arcade.color.RED)
        # border:
        arcade.draw_rectangle_outline(self._cx, self._cy, self._size, self._size, arcade.color.BLACK,
                                      self._line_w + (0 if self._inter_type == InterType.NONE else 1))
        # text:
        text_size = self._size / 4
        magnitude = self._size / 12
        arcade.Text('B', self._cx - self._size / 3, self._cy + text_size / 4 + magnitude * math.sin(self._incrementer),
                    arcade.color.BLACK, text_size,
                    bold=True).draw()
        arcade.Text('D', self._cx - self._size / 3, self._cy - text_size - text_size / 4 + magnitude * math.sin(self._incrementer),
                    arcade.color.BLACK,
                    text_size, bold=True).draw()

        arcade.Text('F', self._cx - self._size / 3 + self._size / 4,
                    self._cy + text_size / 4 + magnitude * math.sin(math.pi / 2 + self._incrementer),
                    arcade.color.BLACK,
                    text_size,
                    bold=True).draw()
        arcade.Text('F', self._cx - self._size / 3 + self._size / 4,
                    self._cy - text_size - text_size / 4 + magnitude * math.sin(math.pi / 2 + self._incrementer),
                    arcade.color.BLACK, text_size,
                    bold=True).draw()

        arcade.Text('S', self._cx - self._size / 3 + self._size / 4 + self._size / 4 - self._size / 32,
                    self._cy + text_size / 4 + magnitude * math.sin(math.pi + self._incrementer),
                    arcade.color.BLACK, text_size,
                    bold=True).draw()
        arcade.Text('S', self._cx - self._size / 3 + self._size / 4 + self._size / 4 - self._size / 32,
                    self._cy - text_size - text_size / 4 + magnitude * math.sin(math.pi + self._incrementer),
                    arcade.color.BLACK, text_size,
                    bold=True).draw()

    def on_motion(self, x, y):
        if self._inter_type != InterType.PRESSED:
            if self.is_point_in_square(self._cx, self._cy, self._size, x, y):
                self._inter_type = InterType.HOVERED
            else:
                self._inter_type = InterType.NONE

    def on_press(self, x, y):
        if self.is_point_in_square(self._cx, self._cy, self._size, x, y):
            if self._inter_type == InterType.PRESSED:
                self._inter_type = InterType.HOVERED
            else:
                self._inter_type = InterType.PRESSED

    def on_drug(self, x, y):
        pass


# enum for node type:
class NodeType(Enum):
    EMPTY = arcade.color.WHITE
    WALL = arcade.color.BLACK
    VISITED_NODE = arcade.color.ROSE_QUARTZ
    NEIGH = arcade.color.BLUEBERRY
    CURRENT_NODE = arcade.color.ROSE
    START_NODE = arcade.color.GREEN
    END_NODE = (75, 150, 0)
    PATH_NODE = arcade.color.RED
    TWICE_VISITED = arcade.color.PURPLE


class InterType(Enum):
    NONE = 1
    HOVERED = 2
    PRESSED = 3


class MenuType(Enum):
    SETTINGS = 1
    BFS_DFS = 2
    A_STAR = 3
    WAVE_LEE = 4


# the main method for a game run:
def main():
    # line_width par should be even number for correct grid&nodes representation:
    game = Lastar(SCREEN_WIDTH, SCREEN_HEIGHT)
    game.setup()
    arcade.run()


# start:
if __name__ == "__main__":
    main()

# v0.1 base Node class created
# v0.2 base Astar(arcade.Window) class created
# v0.3 grid lines drawing added
# v0.4 grid of nodes (self.grid) for Astar class, consisting of Nodes objects created, some fields added for both classes,
# drawing of walls (not passable nodes), start and end nodes added
# v0.5 on_mouse_press() and on_mouse_release methods() overwritten
# v0.6 on_mouse_motion() method overwritten
# v0.7 on_mouse_scroll() method overwritten, now it is possible to draw walls while mouse button pressed,
# switch drawing modes and erase walls, choose start and end node (4 modes are available for now)
# v0.8 on_key_press() method overwritten, clear() method for Node class implemented for resetting the temporal fields to its defaults,
# clear() method for Astar class added to clear all the game field (every Node), now by pressing the 'ENTER' key user can clear all the map
# v0.9 info displaying added (current mode, a_star related information)
# v1.0 a_star now is called by pressing the 'SPACE' key, the shortest way is shown on the grid
# v1.1 visited_nodes are now displayed after a_star call, info extended, hash() dunder method added for class Node
# v1.2 tiebreaker (vector cross product absolute deviation) added
# v1.3 erase separate drawing mode merged with build mode, now there is one build/erase draw mode for building walls by pressing the left mouse button
# and erasing them by pressing the right one
# v1.4 fixed a bug, when some heuristic related temporal pars have not been cleared after a_star had been called
# v1.5 now it is possible to reset all heuristic related pars for the every node on the grid but leave all the walls
# and start and end nodes at their positions by pressing the 'BACKSPACE' key, clear method for Astar class divided into two methods:
# clear_empty_nodes() for partial clearing and clear_grid() for entire clearing
# v1.6 3 auxiliary heuristics added
# v1.7 user interface for heuristic  and tiebreaker choosing added
# v1.8 fixed a bug when start and end nodes have been removed after heuristic had been chosen
# v1.9 start node choosing and end node choosing drawing modes merged into one start & end nodes choosing drawing mode,
# start node is chosen by pressing the left mouse button when end node is chosen by pressing the right one
# v1.10 coordinate pairs tiebreaker added
# v1.11 fixed bug when cross vector product deviation heuristic causes no impact on a_star
# v1.12 interface for scale choosing added
# v1.13 fixed bug when node's filled rectangle has been located not in the center of related grid cell, scaling improved
# v1.14 erase_all_linked_nodes() method added to erase all coherent wall-regions by pressing the middle mouse button on the any cell of them
# v1.15 greedy interaction added, greedy_case's of a_star logic implemented, now it is possible to find some non-shortest ways fast
# v1.16 fixed bug when the time elapsed ms have not been reset after pressing keys such as 'BACKSPACE' and 'ENTER'
# v1.17 fixed bug when greedy flag has had no impact on a_star, fixed closely related to this clearing bug when if there has been at least one
# important node (start or end) unselected clearing process has been finished with error
# --- trying to visualize more visited node by numbers at first and then by colour gradient, bad idea...
# v1.171 max times visited var is now shown in the info, hotfix: bad location bug resolved, GREEDY FLAG -->> GREEDY_FLAG
# v1.18 Wave-spreading lee pathfinding algorithm been implemented, further tests needed...
# v2.0 A-star is now fully interactive (there are two methods: a_star_step_up() -->> RIGHT arrow key and a_star_step_down() -->> LEFT arrow key)
# for moving forward and back through s_star iterations if the flag is_interactive is on. Switcher related added to the window.
# v2.1 Info-getting drawing mode added. Now it is possible to get the information about every node during the a_star call by pressing
# left/right mouse button while in interactive drawing mode
# v2.2 Fixed problem when the heap invariant has been violated during a_star_step_down() calls. Implementations of sift_up() and sift_down()
# methods are borrowed from CPython.heapq github
# v2.3 fixed bug when a rare exception raised during the consecutive calls of a_star_step_down() method
# Current node to be removed from nodes_visited set have been absent. Nodes_visited now is dict instead of set as it was before
# v2.4 Added the correct path restoring to the a_star_interactive mode
# v2.5 Added two methods: path_up() and path_down() to visualize the path restoring phase step by step up and down respectively
# --- trying to represent the leading node of restoring path by a purple circle, bad idea
# v2.6 Now the leading node of the path found is represented by a directed triangle turning the way consistent with the shortest path's direction
# v2.7 fixed bug when the exception raised if user try undoing the path restoring and continue pressing the left arrow key (a_star_down calls)
# v2.8 added fast path_up() and path_down() consecutive calls during the right and the left mouse keys respectively (after a short delay)
# v2.9 fixed bug when ENTER clearing have not been working correctly, some walls have become passable, neighs is now renewed after both types of clearing
# v2.10 wave-spreading lee pathfinding algorithm has been tested and then fully corrected
#
# v2.11 fixed bug when the right arrow key is being pressed after the fully undoing of the path recovering by pressing the left arrow key and an error is raised
# removed an additional interactive step between the null-index of path found and the first call of a_star_step_down() method, the end node (
# the first node of the reversed path) now is marked by a red inner circle
# v2.12 now it is possible to get the full steps of algo by one long pressing the right mouse key and reverse this process by pressing the left moue key
# v2.13 added number representation of f = g + h par for the every visited/heap node than can be turned on/off by pressing the key 'F'
# v2.14 two excess pars have been deleted from Lastar init signature
# v3.0 undo and redo commands for wall-structures have been implemented, they can be called by pressing the 'Z' and 'Y' keys respectively
# v3.1 fixed bug when consecutive erasing linked regions by pressing the middle mouse key could not be undone correctly,passability par has been removed from class Node
# v3.2 fixed bug when walls_built_erased dict is widening enormously quick
# v3.3 common pieces of code from 2 clearing and one rebuilding methods has been merged into new method aux_clear(), good code lines decreasing
# v3.4 fixed bug when END_NODE could not be visited
# v3.5 added a lock-flag that prohibits the is_interactive flag switching after the space has been pressed in interactive mode
# v3.6 fixed all clearing methods, added a lock_flag that prohibits greedy_flag changing during interactive a_star phase, lock colour has been changed to RED, aux_clear() method has been extended
# v4.0 lockers added, while in_interaction is True this flag cannot be switched until the field is fully cleared by pressing the 'ENTER' key, also
# heuristic, tiebreaker, greedy_flag and size_in_tiles cannot be changed too during the interactive phase, added method draw_lock()
# v4.1 now all lockers are red and work correctly
# v4.2 added method draw_cross(), now all the empty fields are crossed while in interactive phase
# v4.3 added a walls set to keep the information about all the walls, that currently exist, methods number_repr() and coords() for number representation of a node
# and the inverse process
# v4.4 the logic of the wall-sprites list shaping has been fully changed, now it is constructed only once at the beginning and then updating...
# performance has increased significantly. The colour of empty node is now WHITE
# v4.5 the way of grid lines drawing has been changed, now they are created through the create_line() method instead of draw_line. And then are formed into a ShapeElementList
# v4.6 fixed bug when redo by pressing the 'Y' key have not been working and bug with walls sets
# v4.7 save/load methods for wall now work correctly in alpha state, should be finished afterwards
# v4.8 comments added, some refactoring and minor fixes
# v4.9 some code reorganization and minor fixes
# v5.0 gear wheel and star drawing methods have been implemented
# v5.1 Lee waves and A star label drawing methods added, some tiles_q have been deleted. Now three icons are situated
# on the left part of the main window, some flags and pars have been added to Lastar class. on_draw(), update()
# and on_mouse_motion() / on_mouse_press() methods renewal
# v5.2 BFS/DFS label added, other icons improved, some minor bugs fixed
# v6.0 BIG BUG with get_neighs() method's two problems has been fixed, now get_neighs has become smarter and works correctly,
# the path now is being visualized more comprehensive (guiding arrows added)
# v6.1 BIG UPDATE: now all the interactive algos has a new option of showing the guided arrows from previous node to its neigh
# for visualizing the way the appropriate algorithm flows, the arrows-icon has been added in order to turn on/off this option,
# many new bugs fixed, some minor refactoring
# v6.2 fixed interactive and lightning bfs and dfs algos' bugs, now it works in the proper way
# v6.3 some interface reorganization for Lastar app in order to achieve more convenience and become user friendlier
# v6.4 methods of getting copy of node and restoring then have become smarter and now can be easily set up
# v6.5 many various bugs fixed, some minor improvements
# v6.6 fixed bugs with a_star interactive, arrows are rarely removed not in a proper way and sometimes arrow-shapes,
# saved in smart copies of neighs were incorrect
# v6.7 now the rare nodes that have been visited twice and more times are displayed by PURPLE colour and have type 'TWICE_VISITED'
# v7.0 GLOBAL UPDATE: added choosing of directions priorities in .het_neighs() method, relative interface part implemented too, someUI reorganization
# v7.1 serious reorganization of drawing methods, new class Renderer that now draws everything non-elementary, weighty refactoring
# v7.2 method .on_mouse_press() has been refactored, now it a way more readable and comprehensible
# v7.3 fixed bug with method .create_line_arrow() absence
#
#
#
#
# TODO: add some other tiebreakers (medium, easy) +-
# TODO: create an info/help pages (high, hard)
# TODO: add a visualization for the most priority nodes in the heap (medium, medium)
# TODO: add heuristic tiebreaker and combined tiebreakers (medium, high) -+
# TODO: add an interaction-prohibition for a large grids (high, easy)
# TODO: find and implement other core algorithms (like Lee and Astar/Dijkstra) (low, high)
# TODO: change the way of drawing, rectangles and ellipses/circles should be switched by sprites for fast .batch rendering (high, high) --+
# TODO: add an info changing depending on a_star heuristic and greedy_flag (high, easy)
# TODO: add correct UI system (high, high)
# TODO: improve the icons interaction (high, high)
# TODO: wise class refactoring with SOLID principles (very high, very high) -+
# TODO: implement a training mode (Levi Gin further task) (very, high, high)
# TODO: proceed from the Renderer class to abstract class Icon and extended child classes like Gear Wheel and so on (very high, very high) --+
# TODO:
