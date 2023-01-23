import heapq as hq
import time
import math
from enum import Enum
import numpy as np
# graphics:
import arcade
import arcade.gui
# data
import shelve
import pickle
# queues:
from collections import deque

# screen sizes:
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1050
TILE_SIZE: int


class Astar(arcade.Window):  # 36 366 98 989 LL
    def __init__(self, width: int, height: int):
        super().__init__(width, height)
        arcade.set_background_color(arcade.color.DUTCH_WHITE)
        self.set_update_rate(1 / 60)
        print(f'walk: {Node.walk}')
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
        # a_star important pars:
        self.start_node = None
        self.end_node = None
        self.heuristic = 0
        self.heuristic_names = {0: 'MANHATTAN', 1: 'EUCLIDIAN', 2: 'MAX_DELTA', 3: 'DIJKSTRA'}
        self.tiebreaker = None
        self.tiebreaker_names = {0: 'VECTOR_CROSS', 1: 'COORDINATES'}
        self.greedy_flag = False  # is algorithm greedy?
        self.nodes_to_be_visited = []
        self.iterations = 0
        self.nodes_visited = {}
        self.time_elapsed_ms = 0
        # interactive a_star pars:
        self.curr_node_dict = {}
        self.max_times_visited_dict = {0: 0}
        self.neighs_added_to_heap_dict = {}
        # lee_wave_spreading important pars:
        # Levi Gin area!
        # interactive pars (game logic):
        self.is_interactive = False
        self.in_interaction = False
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
        # walls building/erasing dicts:
        self.walls_built_erased = [([], True)]
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
        # LEE WAVES:
        ...
        # RIGHT MENU LOGIC:
        self.inter_types = [InterType.NONE for _ in range(4)]  # <<-- GEAR WHEEL, BFS/DFS, A_STAR and WAVE LEE <<--
        self.incrementers = [0 for _ in range(4)]  # <<-- GEAR WHEEL, BFS/DFS, A_STAR and WAVE LEE <<--
        # WAVE_LEE (LEVI GIN):
        self.front_wave_lee = []
        self.next_wave_lee = []
        self.iterations_wave_lee = 0
        self.fronts_dict = {}
        self.curr_node_wave_lee = None
        # BFS/DFS:
        self.is_bfs = False

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
        Node.IS_GREEDY = self.greedy_flag
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

    # a step back for path restoring phase visualisation (interactive a_star mode):
    def path_down(self):
        if (path_node := self.path[self.path_index]).type not in [NodeType.START_NODE, NodeType.END_NODE]:
            path_node.type = NodeType.VISITED_NODE
            path_node.update_sprite_colour()

    # a step forward in path-finding phase visualisation (interactive a_star mode)
    def a_star_step_up(self):
        self.neighs_added_to_heap_dict[self.iterations + 1] = []  # memoization
        # popping out the most priority node for a_star from the heap:
        self.curr_node_dict[self.iterations + 1] = hq.heappop(self.nodes_to_be_visited)  # + memoization
        curr_node = self.curr_node_dict[self.iterations + 1]
        if self.iterations > 0 and curr_node != self.end_node:
            curr_node.type = NodeType.CURRENT_NODE
            curr_node.update_sprite_colour()
        curr_node.times_visited += 1
        if self.iterations > 1:
            if (prev_node := self.curr_node_dict[self.iterations]).type != NodeType.END_NODE:
                prev_node.type = NodeType.VISITED_NODE
                prev_node.update_sprite_colour()
        self.max_times_visited_dict[self.iterations + 1] = max(self.max_times_visited_dict[self.iterations],
                                                               # memoization
                                                               curr_node.times_visited)
        # memoization for correct movement back:
        if curr_node in self.nodes_visited.keys():
            self.nodes_visited[curr_node] += 1
        else:
            self.nodes_visited[curr_node] = 1
        # base case of finding the shortest path:
        if curr_node == self.end_node:
            self.path = self.recover_path()
        # next step:
        # we can search for neighs on the fly or use precalculated sets:
        for neigh in curr_node.get_neighs(self):  # getting all the neighs 'on the fly;
            if neigh.g > curr_node.g + neigh.val:
                # memoization for further 'undoing':
                self.neighs_added_to_heap_dict[self.iterations + 1].append(neigh.aux_copy())
                # cost and heuristic computing:
                neigh.g = curr_node.g + neigh.val
                neigh.h = neigh.heuristics[self.heuristic](neigh, self.end_node)
                # tie-breaking:
                if self.tiebreaker is not None:
                    neigh.tiebreaker = self.start_node.tiebreakers[self.tiebreaker](
                        self.start_node, self.end_node, neigh)
                # previous visited node memoization for further path-recovering process:
                neigh.previously_visited_node = curr_node
                if neigh not in self.nodes_visited and neigh not in [self.start_node, self.end_node]:
                    neigh.type = NodeType.NEIGH
                    neigh.update_sprite_colour()
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
            else:
                curr_node.type = NodeType.VISITED_NODE
                curr_node.update_sprite_colour()
            if self.iterations > 2:
                self.curr_node_dict[self.iterations - 1].type = NodeType.CURRENT_NODE
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
                self.remove_from_heapq(self.nodes_to_be_visited, self.nodes_to_be_visited.index(node))
                node.restore(neigh)
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
            Astar.siftup(heap, ind)
            Astar.siftdown(heap, 0, ind)

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
        Astar.siftdown(heap, start_pos, pos)

    # WAVE LEE LEVI GIN WORK (APPROVED):
    def wave_lee_preparation(self):
        # starting pars:
        self.front_wave_lee = []
        self.next_wave_lee = [self.start_node]
        self.start_node.val = 1
        self.curr_node_wave_lee = self.end_node
        # variable ones:
        self.iterations_wave_lee = 0
        self.fronts_dict = {}

    def wave_lee_step_up(self):
        self.iterations_wave_lee += 1
        self.front_wave_lee = self.next_wave_lee[:]
        self.fronts_dict[self.iterations_wave_lee] = self.front_wave_lee
        self.next_wave_lee = []
        for node in self.front_wave_lee:
            node.val = self.iterations_wave_lee
            if node not in [self.end_node, self.start_node]:
                node.type = NodeType.VISITED_NODE
                node.update_sprite_colour()
            if node == self.end_node:
                self.end_node.val = self.iterations_wave_lee
                self.path = self.recover_path()
                break
            for neigh in node.get_neighs(self):
                if neigh.val == 1 and neigh != self.start_node:
                    if neigh not in self.next_wave_lee:
                        if neigh != self.end_node:
                            neigh.type = NodeType.NEIGH
                            neigh.update_sprite_colour()
                        self.next_wave_lee.append(neigh)
                        neigh.previously_visited_node = node

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

    def bfs_dfs_preparation(self):
        ...

    # PRESETS:
    def setup(self):
        # game set up is located below:
        # sprites, shapes and etc...
        # blocks:
        self.get_sprites()
        # grid lines:
        self.make_grid_lines()

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

    # pre-calculations (kind of optimization, but really is not needed at all...)
    def get_all_neighs(self):
        for row in self.grid:
            for node in row:
                node.get_neighs(self)

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
        # arcade.draw_arc_outline(1825, 500, 56, 28, arcade.color.BLACK, 90, 180, border_width=18, num_segments=128)
        # arcade.create_line_loop([(1815, 500), (1825 + 11, 500 + 9), (1825 + 11, 500 + 4), (1825 + 11 + 22, 500 + 4),
        #                          (1825 + 11 + 22, 500 - 4), (1825 + 11, 500 - 4), (1825 + 11, 500 - 9)],
        #                         arcade.color.BLACK, 2).draw()
        # HINTS:
        arcade.Text(
            f'A* iters: {self.iterations}, path length: {len(self.path) if self.path else "No path found"}, nodes visited: {len(self.nodes_visited)}, '
            f'max times visited: {self.max_times_visited_dict[self.iterations] if self.is_interactive else "LALA"}, time elapsed: {self.time_elapsed_ms} ms',
            365, SCREEN_HEIGHT - 35, arcade.color.BROWN, bold=True).draw()
        arcade.Text(f'Mode: {self.mode_names[self.mode]}', 25, SCREEN_HEIGHT - 35, arcade.color.BLACK, bold=True).draw()
        if self.mode == 2:
            if self.node_chosen:
                arcade.Text(
                    f"NODE'S INFO -->> pos: {self.node_chosen.y, self.node_chosen.x}, g: {self.node_chosen.g}, "
                    f"h: {self.node_chosen.h}, f=g+h: {self.node_chosen.g + self.node_chosen.h}, val: {self.node_chosen.val}, t: {self.node_chosen.tiebreaker}, times visited: {self.node_chosen.times_visited}, type: {self.node_chosen.type}",
                    1050, SCREEN_HEIGHT - 35, arcade.color.PURPLE, bold=True).draw()
        # SET-UPS FOR A_STAR:
        # heuristics:
        if self.inter_types[2] == InterType.PRESSED:
            arcade.Text(f'Heuristics: ', SCREEN_WIDTH - 235, SCREEN_HEIGHT - 70 - 120, arcade.color.BLACK,
                        bold=True).draw()
            for i in range(len(self.heuristic_names)):
                arcade.draw_rectangle_outline(SCREEN_WIDTH - 225, SCREEN_HEIGHT - 100 - 120 - (18 + 2 * 2 + 18) * i, 18,
                                              18,
                                              arcade.color.BLACK, 2)
                arcade.Text(f'{self.heuristic_names[i]}', SCREEN_WIDTH - 225 + (18 + 2 * 2),
                            SCREEN_HEIGHT - 100 - 120 - (18 + 2 * 2 + 18) * i - 6, arcade.color.BLACK, bold=True).draw()

            arcade.draw_rectangle_filled(SCREEN_WIDTH - 225,
                                         SCREEN_HEIGHT - 100 - 120 - (18 + 2 * 2 + 18) * self.heuristic,
                                         14,
                                         14,
                                         arcade.color.BLACK)
            # heuristics lock:
            if self.heuristic_lock:
                self.draw_lock(SCREEN_WIDTH - 225, SCREEN_HEIGHT - 100 - 120 - (18 + 2 * 2 + 18) * self.heuristic)
                for i in range(len(self.heuristic_names)):
                    if i != self.heuristic:
                        self.draw_cross(SCREEN_WIDTH - 225, SCREEN_HEIGHT - 100 - 120 - (18 + 2 * 2 + 18) * i)
            # tiebreakers:
            arcade.Text(f'Tiebreakers: ', SCREEN_WIDTH - 235,
                        SCREEN_HEIGHT - 100 - 120 - (18 + 2 * 2 + 18) * 3 - 18 * 3,
                        arcade.color.BLACK, bold=True).draw()
            for i in range(len(self.tiebreaker_names)):
                arcade.draw_rectangle_outline(SCREEN_WIDTH - 225,
                                              SCREEN_HEIGHT - 100 - 120 - (18 + 2 * 2 + 18) * (3 + i) - 18 * 3 - 30, 18,
                                              18, arcade.color.BLACK, 2)
                arcade.Text(self.tiebreaker_names[i], SCREEN_WIDTH - 225 + (18 + 2 * 2),
                            SCREEN_HEIGHT - 100 - 120 - (18 + 2 * 2 + 18) * (3 + i) - 18 * 3 - 30 - 6,
                            arcade.color.BLACK,
                            bold=True).draw()

            if self.tiebreaker is not None:
                arcade.draw_rectangle_filled(SCREEN_WIDTH - 225,
                                             SCREEN_HEIGHT - 100 - 120 - (18 + 2 * 2 + 18) * (
                                                     3 + self.tiebreaker) - 18 * 3 - 30,
                                             14,
                                             14, arcade.color.BLACK)
            # tiebreakers lock:
            if self.tiebreakers_lock:
                if self.tiebreaker is not None:
                    self.draw_lock(SCREEN_WIDTH - 225,
                                   SCREEN_HEIGHT - 100 - 120 - (18 + 2 * 2 + 18) * (3 + self.tiebreaker) - 18 * 3 - 30)
                for i in range(len(self.tiebreaker_names)):
                    if self.tiebreaker is None or i != self.tiebreaker:
                        self.draw_cross(SCREEN_WIDTH - 225,
                                        SCREEN_HEIGHT - 100 - 120 - (18 + 2 * 2 + 18) * (3 + i) - 18 * 3 - 30)
            # greedy flag:
            arcade.Text('Is greedy: ', SCREEN_WIDTH - 235,
                        SCREEN_HEIGHT - 130 - 120 - (18 + 2 * 2 + 18) * 4 - 2 * 18 * 3,
                        arcade.color.BLACK, bold=True).draw()
            arcade.draw_rectangle_outline(SCREEN_WIDTH - 225,
                                          SCREEN_HEIGHT - 130 - 120 - (18 + 2 * 2 + 18) * 4 - 2 * 18 * 3 - 30, 18, 18,
                                          arcade.color.BLACK, 2)
            arcade.Text(f'GREEDY_FLAG', SCREEN_WIDTH - 225 + (18 + 2 * 2),
                        SCREEN_HEIGHT - 130 - 120 - (18 + 2 * 2 + 18) * 4 - 2 * 18 * 3 - 30 - 6,
                        arcade.color.BLACK, bold=True).draw()

            if self.greedy_flag:
                # greedy flag lock:
                if self.greedy_flag_lock:
                    self.draw_lock(SCREEN_WIDTH - 225,
                                   SCREEN_HEIGHT - 130 - 120 - (18 + 2 * 2 + 18) * 4 - 2 * 18 * 3 - 30)
                else:
                    arcade.draw_rectangle_filled(SCREEN_WIDTH - 225,
                                                 SCREEN_HEIGHT - 130 - 120 - (18 + 2 * 2 + 18) * 4 - 2 * 18 * 3 - 30,
                                                 14,
                                                 14,
                                                 arcade.color.BLACK)
            else:
                if self.greedy_flag_lock:
                    self.draw_cross(SCREEN_WIDTH - 225,
                                    SCREEN_HEIGHT - 130 - 120 - (18 + 2 * 2 + 18) * 4 - 2 * 18 * 3 - 30)
            # a_star show mode:
            arcade.Text('Show mode: ', SCREEN_WIDTH - 235,
                        SCREEN_HEIGHT - 190 - (18 + 2 * 2 + 18) * 14 - 4 * 18 * 3, arcade.color.BLACK,
                        bold=True).draw()

            arcade.draw_rectangle_outline(SCREEN_WIDTH - 225,
                                          SCREEN_HEIGHT - 190 - (18 + 2 * 2 + 18) * 14 - 4 * 18 * 3 - 30, 18,
                                          18,
                                          arcade.color.BLACK, 2)

            arcade.Text('IS_A_STAR_INTERACTIVE', SCREEN_WIDTH - 225 + (18 + 2 * 2),
                        SCREEN_HEIGHT - 190 - (18 + 2 * 2 + 18) * 14 - 4 * 18 * 3 - 30 - 6, arcade.color.BLACK,
                        bold=True).draw()

            if self.is_interactive:
                # a_star show mode lock:
                if self.in_interaction_mode_lock:
                    self.draw_lock(SCREEN_WIDTH - 225,
                                   SCREEN_HEIGHT - 190 - (18 + 2 * 2 + 18) * 14 - 4 * 18 * 3 - 30)
                else:
                    arcade.draw_rectangle_filled(SCREEN_WIDTH - 225,
                                                 SCREEN_HEIGHT - 190 - (18 + 2 * 2 + 18) * 14 - 4 * 18 * 3 - 30,
                                                 14, 14,
                                                 arcade.color.BLACK)
        # SETTINGS:
        elif self.inter_types[0] == InterType.PRESSED:
            # scaling:
            arcade.Text('Sizes in tiles: ', SCREEN_WIDTH - 235,
                        SCREEN_HEIGHT - 160 - 120 - (18 + 2 * 2 + 18) * 4 - 3 * 18 * 3,
                        arcade.color.BLACK, bold=True).draw()

            for i in range(len(self.scale_names)):
                arcade.draw_rectangle_outline(SCREEN_WIDTH - 225,
                                              SCREEN_HEIGHT - 160 - 120 - (18 + 2 * 2 + 18) * (4 + i) - 3 * 18 * 3 - 30,
                                              18,
                                              18,
                                              arcade.color.BLACK, 2)
                arcade.Text(f'{self.scale_names[i]}x{self.get_hor_tiles(i)}', SCREEN_WIDTH - 225 + (18 + 2 * 2),
                            SCREEN_HEIGHT - 160 - 120 - (18 + 2 * 2 + 18) * (4 + i) - 3 * 18 * 3 - 30 - 6,
                            arcade.color.BLACK, bold=True).draw()
            # scaling lock:
            if self.scale_lock:
                self.draw_lock(SCREEN_WIDTH - 225,
                               SCREEN_HEIGHT - 160 - 120 - (18 + 2 * 2 + 18) * (4 + self.scale) - 3 * 18 * 3 - 30)
                for i in range(len(self.scale_names)):
                    if i != self.scale:
                        self.draw_cross(SCREEN_WIDTH - 225,
                                        SCREEN_HEIGHT - 160 - 120 - (18 + 2 * 2 + 18) * (4 + i) - 3 * 18 * 3 - 30)
            else:
                arcade.draw_rectangle_filled(SCREEN_WIDTH - 225,
                                             SCREEN_HEIGHT - 160 - 120 - (18 + 2 * 2 + 18) * (
                                                     4 + self.scale) - 3 * 18 * 3 - 30,
                                             14,
                                             14,
                                             arcade.color.BLACK)
        # NODE CHOSEN (should work for every algo):
        if self.node_chosen:
            arcade.draw_circle_filled(5 + self.node_chosen.x * self.tile_size + self.tile_size / 2,
                                      5 + self.node_chosen.y * self.tile_size + self.tile_size / 2, self.tile_size / 4,
                                      arcade.color.YELLOW)
        # CURRENT PATH NODE ON START OR END NODE (should work for every algo), TODO: should be implemented in a separated method:
        if self.in_interaction:
            if self.start_node and self.end_node:
                if self.path is not None and self.path_index > 0:
                    p = -self.path[self.path_index].x + self.path[self.path_index - 1].x, -self.path[
                        self.path_index].y + \
                        self.path[self.path_index - 1].y
                    points = self.get_triangle(self.path[self.path_index], p)
                    arcade.draw_triangle_filled(points[0], points[1], points[2], points[3], points[4], points[5],
                                                arcade.color.RED)
            if self.path:
                arcade.draw_circle_filled(5 + self.end_node.x * self.tile_size + self.tile_size / 2,
                                          5 + self.end_node.y * self.tile_size + self.tile_size / 2,
                                          self.tile_size / 4,
                                          arcade.color.RED)
        # ICONS OF INTERACTION:

        # GEAR WHEEL:
        self.draw_gear_wheel(1785, 1000, 24, 24, 6)

        # BFS and DFS icons:
        self.draw_bfs_dfs(1750 - 30, 1020 - 73 - 25, 54, 2)

        # A STAR LABEL:
        self.draw_a_star(1750 + 15, 1020 - 100 - 25, 22, 53)  # 36 366 98 989

        # LEE WAVES:
        self.draw_waves(1750 + 50 + 48 + 10, 1020 - 73 - 25, 32, 5)

        # self.draw_bfs(1235, 935, 32, 2)

        # DASHED LINE CIRCLE:
        # self.draw_dashed_line_circle(1535, 935, 32, 16, 2)

    # triangle points getting:
    def get_triangle(self, node: 'Node', point: tuple[int, int]):
        scaled_point = point[0] * (self.tile_size // 2 - 2), point[1] * (self.tile_size // 2 - 2)
        deltas = (scaled_point[0] - scaled_point[1], scaled_point[0] + scaled_point[1]), (
            scaled_point[0] + scaled_point[1], -scaled_point[0] + scaled_point[1])
        cx, cy = 5 + node.x * self.tile_size + self.tile_size / 2, 5 + node.y * self.tile_size + self.tile_size / 2
        return cx, cy, cx + deltas[0][0], cy + deltas[0][1], cx + deltas[1][0], cy + deltas[1][1]

    # draws a lock for right window part:
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

    # draws left or right arrow:
    def make_arrow(self):
        ...

    def draw_gear_wheel(self, cx, cy, rx=32, ry=32, cog_size=8, multiplier=1.5, line_w=2, shift=False, clockwise=True):
        self.cx, self.cy = cx, cy
        self.hole_rx, self.hole_ry = rx - multiplier * cog_size, ry - multiplier * cog_size
        circumference = math.pi * (rx + ry)  # approximately if cog_size << radius
        angular_size = (2 * math.pi) * cog_size / circumference
        max_cogs_fit_in_the_gear_wheel = int(circumference / cog_size)
        cogs_q = max_cogs_fit_in_the_gear_wheel // 2
        fit_angular_size = (2 * math.pi - cogs_q * angular_size) / cogs_q
        angle = (angular_size if shift else 0) + (
            self.incrementers[0] if clockwise else -self.incrementers[0])  # in radians
        self.upper_vertices_list = []
        for i in range(cogs_q):
            # aux pars:
            _a, a_ = (angle - angular_size / 2), (angle + angular_size / 2)
            _rx, _ry, rx_, ry_ = rx * math.cos(_a), ry * math.sin(_a), rx * math.cos(a_), ry * math.sin(a_)
            _dx, _dy = cog_size * math.cos(_a), cog_size * math.sin(_a)
            dx_, dy_ = cog_size * math.cos(a_), cog_size * math.sin(a_)
            # polygon's points:
            self.upper_vertices_list.append([cx + _rx, cy + _ry])
            self.upper_vertices_list.append([cx + _rx + _dx, cy + _ry + _dy])
            self.upper_vertices_list.append([cx + rx_ + dx_, cy + ry_ + dy_])
            self.upper_vertices_list.append([cx + rx_, cy + ry_])
            # angle incrementation:
            angle += angular_size + fit_angular_size
        # upper gear wheel:
        # arcade.draw_polygon_filled(upper_vertices_list, arcade.color.PASTEL_GRAY)
        if self.inter_types[0] == InterType.PRESSED:
            arcade.draw_polygon_filled(self.upper_vertices_list, arcade.color.RED)
        arcade.draw_polygon_outline(self.upper_vertices_list, arcade.color.BLACK,
                                    line_w + (0 if self.inter_types[0] == InterType.NONE else 1))
        # hole:
        arcade.draw_ellipse_filled(cx, cy, 2 * (rx - multiplier * cog_size), 2 * (ry - multiplier * cog_size),
                                   arcade.color.DUTCH_WHITE)
        arcade.draw_ellipse_outline(cx, cy, 2 * (rx - multiplier * cog_size), 2 * (ry - multiplier * cog_size),
                                    arcade.color.BLACK,
                                    line_w + (0 if self.inter_types[0] == InterType.NONE else 1))

    def draw_waves(self, cx, cy, size=32, waves_q=5, line_w=2):
        ds = size / waves_q
        s_list = sorted([(i * ds + self.incrementers[3]) % size for i in range(waves_q)], reverse=True)
        for i, curr_s in enumerate(s_list):
            if self.inter_types[3] == InterType.PRESSED:
                arcade.draw_circle_filled(cx, cy, curr_s, arcade.color.RED if i % 2 == 0 else arcade.color.DUTCH_WHITE)
            arcade.draw_circle_outline(cx, cy, curr_s, arcade.color.BLACK,
                                       line_w + (0 if self.inter_types[3] == InterType.NONE else 1))

    def draw_star(self, cx, cy, vertices=5, r=32, line_w=2, clockwise=True):
        delta_angle = 2 * math.pi / vertices
        d = vertices // 2
        angle = self.incrementers[2] if clockwise else -self.incrementers[2]  # in radians
        for i in range(vertices):
            da = d * delta_angle
            arcade.draw_line(cx + r * math.cos(angle),
                             cy + r * math.sin(angle),
                             cx + r * math.cos(angle + da),
                             cy + r * math.sin(angle + da),
                             arcade.color.BLACK, line_w)
            angle += da

    def draw_a_star(self, cx, cy, size_w, size_h, line_w=2, clockwise=True):
        # drawing A:
        self.draw_a(cx, cy, size_w, size_h, size_w / 3, line_w)
        # Star spinning around A:
        self.draw_star(cx + size_h / 2 + size_h / 3, cy + size_h, r=size_h / 4, line_w=line_w, clockwise=clockwise)

    def draw_a(self, cx, cy, length, height, a_w, line_w):
        upper_hypot = math.sqrt(length ** 2 + height ** 2)
        cos, sin = height / upper_hypot, length / upper_hypot
        line_w_hour_projection = a_w / cos
        dh = a_w / sin
        delta = (height - a_w - dh) / 2
        k, upper_k = (delta - a_w / 2) * length / height, (delta + a_w / 2) * length / height

        self.a_outer_points = [
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

        if self.inter_types[2] == InterType.NONE:
            arcade.draw_polygon_outline(self.a_outer_points, arcade.color.BLACK, line_w)
            arcade.draw_polygon_outline(a_inner_points, arcade.color.BLACK, line_w)
        elif self.inter_types[2] == InterType.HOVERED:
            arcade.draw_polygon_outline(self.a_outer_points, arcade.color.BLACK, line_w + 1)
            arcade.draw_polygon_outline(a_inner_points, arcade.color.BLACK, line_w + 1)
        else:
            arcade.draw_polygon_filled(self.a_outer_points, arcade.color.RED)
            arcade.draw_polygon_outline(self.a_outer_points, arcade.color.BLACK, line_w + 1)
            arcade.draw_polygon_filled(a_inner_points, arcade.color.DUTCH_WHITE)
            arcade.draw_polygon_outline(a_inner_points, arcade.color.BLACK, line_w + 1)

    def draw_bfs(self, cx, cy, r, line_w):
        arcade.draw_circle_outline(cx, cy, r, arcade.color.BLACK, line_w)
        self.draw_line_arrow(cx + r + r / 3, cy, r, arrow_l=r / 4, is_left=False, line_w=line_w)
        self.draw_line_arrow(cx - r - r / 3, cy, r, arrow_l=r / 4, is_left=True, line_w=line_w)
        self.draw_dashed_line_circle(cx + 3 * r + 2 * r / 3, cy, r, r // 2, line_w)
        self.draw_dashed_line_circle(cx - 3 * r - 2 * r / 3, cy, r, r // 2, line_w)

    def draw_dfs(self):
        ...

    # simplified version:
    def draw_bfs_dfs(self, cx, cy, size, line_w):
        self.bfs_dfs_cx, self.bfs_dfs_cy = cx, cy
        self.bfs_dfs_size = size
        # filling:
        if self.inter_types[1] == InterType.PRESSED:
            arcade.draw_rectangle_filled(cx, cy, size, size, arcade.color.RED)
        # border:
        arcade.draw_rectangle_outline(cx, cy, size, size, arcade.color.BLACK,
                                      line_w + (0 if self.inter_types[1] == InterType.NONE else 1))
        # text:
        text_size = size / 4
        magnitude = size / 12
        arcade.Text('B', cx - size / 3, cy + text_size / 4 + magnitude * math.sin(self.incrementers[1]),
                    arcade.color.BLACK, text_size,
                    bold=True).draw()
        arcade.Text('D', cx - size / 3, cy - text_size - text_size / 4 + magnitude * math.sin(self.incrementers[1]),
                    arcade.color.BLACK,
                    text_size, bold=True).draw()

        arcade.Text('F', cx - size / 3 + size / 4,
                    cy + text_size / 4 + magnitude * math.sin(math.pi / 2 + self.incrementers[1]),
                    arcade.color.BLACK,
                    text_size,
                    bold=True).draw()
        arcade.Text('F', cx - size / 3 + size / 4,
                    cy - text_size - text_size / 4 + magnitude * math.sin(math.pi / 2 + self.incrementers[1]),
                    arcade.color.BLACK, text_size,
                    bold=True).draw()

        arcade.Text('S', cx - size / 3 + size / 4 + size / 4 - size / 32,
                    cy + text_size / 4 + magnitude * math.sin(math.pi + self.incrementers[1]),
                    arcade.color.BLACK, text_size,
                    bold=True).draw()
        arcade.Text('S', cx - size / 3 + size / 4 + size / 4 - size / 32,
                    cy - text_size - text_size / 4 + magnitude * math.sin(math.pi + self.incrementers[1]),
                    arcade.color.BLACK, text_size,
                    bold=True).draw()

    def draw_dashed_line_circle(self, cx, cy, r, q, line_w, shift=False, clockwise=True):
        angular_size = math.pi / q
        angle = (angular_size if shift else 0) + (self.twist_angle if clockwise else -self.twist_angle)  # in radians
        for i in range(q):
            _a, a_ = (angle - angular_size / 2), (angle + angular_size / 2)
            _rx, _ry = r * math.cos(_a), r * math.sin(_a)
            rx_, ry_ = r * math.cos(a_), r * math.sin(a_)
            arcade.draw_line(cx + _rx, cy + _ry, cx + rx_, cy + ry_, arcade.color.BLACK, line_w)
            angle += angular_size * 2

    # by default the arrow to be drawn is left sided:
    def draw_line_arrow(self, cx, cy, length=50, angle=math.pi / 4, arrow_l=8, is_left=False, line_w=2):
        _cx_ = cx + (-length if is_left else length)
        arcade.draw_line(cx, cy, _cx_, cy, arcade.color.BLACK, line_w)
        arcade.draw_line(_cx_, cy, _cx_ + (1 if is_left else -1) * arrow_l * math.cos(angle),
                         cy + arrow_l * math.sin(angle), arcade.color.BLACK, line_w)
        arcade.draw_line(_cx_, cy, _cx_ + (1 if is_left else -1) * arrow_l * math.cos(angle),
                         cy - arrow_l * math.sin(angle), arcade.color.BLACK, line_w)

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
        # consecutive calls during key pressing:
        ticks_threshold = 12
        if self.cycle_breaker_right:
            self.ticks_before += 1
            if self.ticks_before >= ticks_threshold:
                if self.path is None:
                    if self.inter_types[1] == InterType.PRESSED:
                        ...
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
                        ...
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
                        self.a_star_step_down()

    # KEYBOARD:
    def on_key_press(self, symbol: int, modifiers: int):
        # is called when user press the symbol key:
        match symbol:
            # a_star_call:
            case arcade.key.SPACE:
                if not self.loading:
                    if self.start_node and self.end_node:
                        # STEP BY STEP:
                        if self.is_interactive:
                            # game logic:
                            self.in_interaction = True
                            # prepare:
                            if self.inter_types[1] == InterType.PRESSED:
                                self.bfs_dfs_preparation()
                            elif self.inter_types[2] == InterType.PRESSED:
                                self.a_star_preparation()
                            elif self.inter_types[3] == InterType.PRESSED:
                                self.wave_lee_preparation()
                        # ONE SPACEBAR PRESS:
                        else:
                            start = time.time_ns()
                            # getting paths:
                            if self.inter_types[1] == InterType.PRESSED:
                                self.path = self.start_node.bfs(self.end_node, self) if self.is_bfs \
                                    else self.start_node.dfs(self.end_node, self)
                            elif self.inter_types[2] == InterType.PRESSED:
                                self.path = self.start_node.a_star(self.end_node, self)
                            elif self.inter_types[3] == InterType.PRESSED:
                                self.path = self.start_node.wave_lee(self.end_node, self)
                            finish = time.time_ns()
                            self.time_elapsed_ms = self.get_ms(start, finish)
                            # path's drawing:
                            print(f"PATH'S LENGTH: {len(self.path)}")
                            for node in self.path:
                                if node.type not in [NodeType.START_NODE, NodeType.END_NODE]:
                                    node.type = NodeType.PATH_NODE
                                    node.update_sprite_colour()
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
                            ...
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
                            ...
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
                                ...
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

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int):
        # A_STAR BLOCK:
        if self.inter_types[2] == InterType.PRESSED:
            # setting_up heuristic and tiebreaker:
            if not self.heuristic_lock:
                for i in range(len(self.heuristic_names)):
                    if SCREEN_WIDTH - 225 - 9 <= x <= SCREEN_WIDTH - 225 + 9 and SCREEN_HEIGHT - 100 - 120 - (
                            18 + 2 * 2 + 18) * i - 9 <= y <= SCREEN_HEIGHT - 100 - 120 - (18 + 2 * 2 + 18) * i + 9:
                        self.heuristic = i
                        break
            if not self.tiebreakers_lock:
                for i in range(len(self.tiebreaker_names)):
                    if SCREEN_WIDTH - 225 - 9 <= x <= SCREEN_WIDTH - 225 + 9 and SCREEN_HEIGHT - 100 - 120 - (
                            18 + 2 * 2 + 18) * (3 + i) - 18 * 3 - 30 - 9 <= y <= SCREEN_HEIGHT - 100 - 120 - (
                            18 + 2 * 2 + 18) * (3 + i) - 18 * 3 - 30 + 9:
                        self.tiebreaker = None if self.tiebreaker == i else i
                        break
            # setting up the greedy flag:
            if SCREEN_WIDTH - 225 - 9 <= x <= SCREEN_WIDTH - 225 + 9 and SCREEN_HEIGHT - 130 - 120 - (
                    18 + 2 * 2 + 18) * 4 - 2 * 18 * 3 - 30 - 9 <= y <= SCREEN_HEIGHT - 130 - 120 - (
                    18 + 2 * 2 + 18) * 4 - 2 * 18 * 3 - 30 + 9:
                if not self.greedy_flag_lock:
                    self.greedy_flag = not self.greedy_flag
            # setting up the interactive a_star flag:
            if SCREEN_WIDTH - 225 - 9 <= x <= SCREEN_WIDTH - 225 + 9 and SCREEN_HEIGHT - 190 - (
                    18 + 2 * 2 + 18) * 14 - 4 * 18 * 3 - 30 - 9 <= y <= SCREEN_HEIGHT - 190 - (
                    18 + 2 * 2 + 18) * 14 - 4 * 18 * 3 - 30 + 9:
                if not self.in_interaction_mode_lock:
                    self.is_interactive = not self.is_interactive
        # SETTINGS BLOCK:
        # choosing the scale factor:
        if not self.scale_lock:
            for i in range(len(self.scale_names)):
                if SCREEN_WIDTH - 225 - 9 <= x <= SCREEN_WIDTH - 225 + 9 and SCREEN_HEIGHT - 160 - 120 - (
                        18 + 2 * 2 + 18) * (4 + i) - 3 * 18 * 3 - 30 - 9 <= y <= SCREEN_HEIGHT - 160 - 120 - (
                        18 + 2 * 2 + 18) * (4 + i) - 3 * 18 * 3 - 30 + 9:
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
        if (x - (1750 + 50 + 48 + 10)) ** 2 + (y - (1020 - 73 - 25)) ** 2 <= 32 ** 2:
            if self.inter_types[3] == InterType.HOVERED:
                self.inter_types[3] = InterType.PRESSED
                self.clear_inter_types(3)
            elif self.inter_types[3] == InterType.PRESSED:
                self.inter_types[3] = InterType.HOVERED

    # helpful auxiliary methods:
    @staticmethod
    def is_point_in_square(cx, cy, size, x, y):
        return cx - size / 2 <= x <= cx + size / 2 and cy - size / 2 <= y <= cy + size / 2

    @staticmethod
    def is_point_in_circle(cx, cy, r, x, y):
        return (cx - x) ** 2 + (cy - y) ** 2 <= r ** 2

    # game mode switching by scrolling the mouse wheel:
    def on_mouse_scroll(self, x: int, y: int, scroll_x: int, scroll_y: int):
        self.mode = (self.mode + 1) % len(self.mode_names)

    def on_mouse_release(self, x: int, y: int, button: int, modifiers: int):
        self.building_walls_flag = False

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


# class for a node representation:
class Node:
    # horizontal and vertical up and down moves:
    walk = [(dy, dx) for dx in range(-1, 2) for dy in range(-1, 2) if dy * dx == 0 and (dy, dx) != (0, 0)]
    extended_walk = [(dy, dx) for dx in range(-1, 2) for dy in range(-1, 2) if (dy, dx) != (0, 0)]
    IS_GREEDY = False

    def __init__(self, y, x, val, node_type: 'NodeType'):
        # type and sprite:
        self.type = node_type
        self.sprite = None
        # important pars:
        self.y, self.x = y, x
        self.val = val
        self.neighs = set()  # the nearest neighbouring nodes, now this par is not needed
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
        # bfs/dfs pars:
        self.visited = False

    # COPYING/RESTORING:
    # makes an auxiliary copy for a nde, it is needed for a_star interactive:
    def aux_copy(self):
        copied_node = Node(self.y, self.x, self.type, self.val)
        copied_node.g = self.g
        copied_node.h = self.h
        copied_node.tiebreaker = self.tiebreaker
        return copied_node

    # restore the node from its auxiliary copy:
    def restore(self, copied_node: 'Node'):
        self.g = copied_node.g
        self.h = copied_node.h
        self.tiebreaker = copied_node.tiebreaker
        if self.type != NodeType.END_NODE:
            self.type = NodeType.EMPTY
            self.sprite.color = arcade.color.WHITE

    # TYPE/SPRITE CHANGE/INIT:
    # makes a solid colour sprite for a node:
    def get_solid_colour_sprite(self, game: Astar):
        cx, cy, size, colour = self.get_center_n_sizes(game)
        self.sprite = arcade.SpriteSolidColor(size, size, colour)
        self.sprite.center_x, self.sprite.center_y = cx, cy
        game.node_sprite_list.append(self.sprite)

    # aux calculations:
    def get_center_n_sizes(self, game: Astar):
        return (5 + game.tile_size * self.x + game.tile_size / 2,
                5 + game.tile_size * self.y + game.tile_size / 2,
                game.tile_size - 2 * game.line_width - (1 if game.line_width % 2 != 0 else 0),
                self.type.value)

    # updates the sprite's color (calls after node's type switching)
    def update_sprite_colour(self):
        self.sprite.color = self.type.value

    def __str__(self):
        return f'{self.y, self.x} -->> {self.val}'

    def __repr__(self):
        return str(self)

    # DUNDERS:
    def __eq__(self, other):
        if type(self) != type(other):
            return False
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

    # clears the node heuristically:
    def heur_clear(self):
        self.g = np.Infinity
        self.h = 0
        self.tiebreaker = None
        self.previously_visited_node = None
        self.times_visited = 0
        self.neighs = set()
        # bfs/dfs:
        self.visited = False
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
    # gets neighs of the node:
    def get_neighs(self, game: 'Astar'):
        for dy, dx in self.walk:
            ny, nx = self.y + dy, self.x + dx
            if 0 <= ny < game.tiles_q and 0 <= nx < game.hor_tiles_q:
                if game.grid[ny][nx].type in [NodeType.EMPTY, NodeType.END_NODE]:
                    self.neighs.add(game.grid[ny][nx])
        return self.neighs

    # gets extended neighs (with diagonal ones) of the node:
    def get_extended_neighs(self, game: 'Astar') -> list['Node']:
        for dy, dx in self.extended_walk:
            ny, nx = self.y + dy, self.x + dx
            if 0 <= ny < game.tiles_q and 0 <= nx < game.hor_tiles_q:
                yield game.grid[ny][nx]

    # PATHFINDING:
    # Witchdoctor's algos:
    def bfs(self, other: 'Node', game: 'Astar'):  # iterative one:
        queue = deque()
        queue.append(self)
        self.visited = True
        while queue:
            game.iterations += 1
            current_node = queue.pop()
            current_node.type = NodeType.VISITED_NODE
            current_node.update_sprite_colour()
            current_node.times_visited += 1
            if current_node == other:
                return self.restore_path(other)
            for neigh in current_node.get_neighs(game):
                if not neigh.visited:
                    neigh.visited = True
                    neigh.previously_visited_node = current_node
                    queue.appendleft(neigh)

    def dfs(self, other: 'Node', game: 'Astar'):  # recursive one (can be easily implemented through the queue/stack):
        def rec_dfs(current_node):
            game.iterations += 1
            print(f'{game.iterations}-th iteration, current node: {current_node}')
            current_node.type = NodeType.VISITED_NODE
            current_node.update_sprite_colour()
            current_node.times_visited += 1
            if current_node == other:
                return self.restore_path(other)
            for neigh in current_node.get_neighs(game):
                if not neigh.visited:
                    neigh.previously_visited_node = current_node
                    neigh.visited = True
                    if (r := rec_dfs(neigh)) is not None:
                        return r
        path = rec_dfs(self)
        return path if path is not None else []

    # finished, tested and approved by Levi Gin:
    def wave_lee(self, other: 'Node', game: 'Astar'):
        # other.get_neighs(game)  # Why is it necessary???
        front_wave = [self]
        iteration = 0
        # wave-spreading:
        while front_wave:
            iteration += 1
            new_front_wave = []
            for front_node in front_wave:
                front_node.val = iteration
                if front_node not in [self, other]:
                    front_node.type = NodeType.VISITED_NODE
                    front_node.update_sprite_colour()
                if front_node == other:
                    return self.restore_path(other)
                for front_neigh in front_node.get_neighs(game):
                    if front_neigh.val == 1 and front_neigh not in new_front_wave and front_neigh != self:
                        front_neigh.previously_visited_node = front_node
                        new_front_wave.append(front_neigh)
            front_wave = new_front_wave[:]
        return []

    # a common a_star:
    def a_star(self, other: 'Node', game: 'Astar'):
        Node.IS_GREEDY = game.greedy_flag
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
            for neigh in curr_node.get_neighs(game):
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
    game = Astar(SCREEN_WIDTH, SCREEN_HEIGHT)
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
#
#
# TODO: add some other tiebreakers (medium, easy) +-
# TODO: upgrade the visual part (medium, medium) -+
# TODO: add a core-algo switcher that can change the right interactive panel (Lee, Astar and so on) (high, medium)
# TODO: create an info/help pages (high, hard)
# TODO: extend the algo base with Lee wave pathfinding algorithm (medium, medium) Levi Gin +-
# TODO: add a visualization for the most priority nodes in the heap (medium, medium)
# TODO: add heuristic tiebreaker and combined tiebreakers (medium, high)
# TODO: add a scroller on the right side of the window (high, medium)
# TODO: add an interaction-prohibition for a large grids (high, easy)
# TODO: find and implement other core algorithms (like Lee and Astar/Dijkstra) (low, high)
# TODO: change the way of drawing, rectangles and ellipses/circles should be switched by sprites for fast .batch rendering (high, high) --+
# TODO: add a command of wall-pattern saving and further loading (low, high) -+
# TODO: simplify the drawing (high, high)
# TODO: add an info changing depending on a_star heuristic and greedy_flag (high, easy)
# TODO: add correct UI system (high, high)
# TODO: add a_star and wave_lee management menus (high, medium)
# TODO: improve the icons interaction (high, high)
# TODO: fix bug with red arrow during the path restoring phase (high, high)
# TODO:
