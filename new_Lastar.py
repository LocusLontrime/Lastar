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


class Lastar(arcade.Window):
    def __init__(self, width: int, height: int):
        super().__init__(width, height)
        arcade.set_background_color(arcade.color.DUTCH_WHITE)
        self.set_update_rate(1 / 60)
        # interaction mode ON/OFF:
        self._interactive_ind = None
        self._in_interaction = False
        self._in_interaction_mode_lock = False
        # save/load mode:
        self._loading = False
        # game modes and flags:
        self._mode = 0  # 0 for building the walls and erasing them afterwards, 1 for a start and end nodes choosing and 2 for info getting for every node
        self._mode_names = {0: 'BUILDING/ERASING', 1: 'START&END_NODES_CHOOSING', 2: 'INFO_GETTING'}
        self._building_walls_flag = False
        self._build_or_erase = True  # True for building and False for erasing
        # names dicts:
        self._heuristic_names = {0: 'MANHATTAN', 1: 'EUCLIDIAN', 2: 'MAX_DELTA', 3: 'DIJKSTRA'}
        self._tiebreaker_names = {0: 'VECTOR_CROSS', 1: 'COORDINATES'}
        self._greedy_names = {0: 'IS_GREEDY'}
        self._guide_arrows_names = {0: f'ON/OFF'}
        self._interactive_names = {0: f'IS_INTERACTIVE'}
        # grid and walls_manager:
        self._grid = Grid()
        self._walls_manager = WallsManager()
        # algorithms:
        self._astar = None
        self._wave_lee = None
        self._bfs_dfs = None
        # current_algo:
        self._current_algo = None
        # algos' icons:
        self._astar_icon = None
        self._wave_lee_icon = None
        self._bfs_dfs_icon = None
        # algos' menus:
        self._wave_lee_menu = None
        self._astar_menu = None
        self._bfs_dfs_menu = None
        # settings icon:
        self._gear_wheel = None
        # settings menu and arrows menu:
        self._arrows_menu = None
        self._settings_menu = None
        # two aux areas:
        self._guide_arrows_area = None
        self._show_mode_area = None

        self.elements_setup()

        # icons_dict:
        self._icons_dict = {0: self._gear_wheel, 1: self._bfs_dfs_icon, 2: self._astar_icon, 3: self._wave_lee_icon}
        # menus_dict:
        self._menus_dict = {0: self._settings_menu, 1: self._bfs_dfs_menu, 2: self._astar_menu, 3: self._wave_lee_menu}

        print(f'LALA')

    def elements_setup(self):
        # pars:
        sq_size, line_w = 18, 2
        delta = 2 * (sq_size + line_w)
        # ALGOS:
        self._astar = Astar()
        self._wave_lee = WaveLee()
        self._bfs_dfs = BfsDfs()
        # ICONS and MENUS:
        # icon setting up for a_star:
        self._astar_icon = AstarIcon(1750 + 15 + 6, 1020 - 100 - 25, 22, 53)
        # areas setting up:
        bot_menu_x, bot_menu_y = SCREEN_WIDTH - 235, SCREEN_HEIGHT - 70 - 120
        heurs_area = Area(bot_menu_x, bot_menu_y, delta, sq_size, line_w, f'Heuristics', self._heuristic_names)
        heurs_area.connect_to_func(self._astar.set_heuristic)
        bot_menu_y -= 30 + (len(self._heuristic_names) - 1) * delta + 3 * sq_size
        tiebreakers_area = Area(bot_menu_x, bot_menu_y, delta, sq_size, line_w, f'Tiebreakers', self._tiebreaker_names)
        tiebreakers_area.connect_to_func(self._astar.set_tiebreaker)
        bot_menu_y -= 30 + (len(self._tiebreaker_names) - 1) * delta + 3 * sq_size
        is_greedy_area = Area(bot_menu_x, bot_menu_y, delta, sq_size, line_w, f'Is greedy', self._greedy_names)
        is_greedy_area.connect_to_func(self._astar.set_greedy_ind)
        # bot_menu_y -= 30 + 3 * sq_size
        # guide_arrows_area = Area(bot_menu_x, bot_menu_y, delta, sq_size, line_w, f'Guide arrows',
        #                          self._guide_arrows_names)
        # bot_menu_y -= 30 + 3 * sq_size
        # show_mode_area = Area(bot_menu_x, bot_menu_y, delta, sq_size, line_w, f'Show mode', self._interactive_names)
        # menu composing and connecting to an icon:
        self._astar_menu = Menu()
        self._astar_menu.multiple_append(heurs_area, tiebreakers_area, is_greedy_area)
        self._astar_menu.connect(self._astar_icon)
        # icon setting up for wave_lee:
        self._wave_lee_icon = Waves(1750 + 50 + 48 + 10 + 6, 1020 - 73 - 25, 32, 5)
        # icon setting up for bfs/dfs:
        self._bfs_dfs_icon = BfsDfsIcon(1750 - 30 + 6, 1020 - 73 - 25, 54, 2)
        # area setting up:
        bot_menu_x, bot_menu_y = SCREEN_WIDTH - 235, SCREEN_HEIGHT - 70 - 120
        area = Area(bot_menu_x, bot_menu_y, delta, sq_size, line_w, f'Core', {0: f'BFS', 1: f'DFS'})
        area.connect_to_func(self._bfs_dfs.set_is_bfs)
        # menu composing and connecting to an icon:
        self._bfs_dfs_menu = Menu()
        self._bfs_dfs_menu.append_area(area)
        self._bfs_dfs_menu.connect(self._bfs_dfs_icon)
        # GRID:
        # grid icon setting up:
        self._gear_wheel = GearWheelButton(1785 + 6, 1000, 24, 6)
        # arrows menu setting up:
        bot_menu_x, bot_menu_y = SCREEN_WIDTH - 235, SCREEN_HEIGHT - 70 - 120
        self._arrows_menu = ArrowsMenu(bot_menu_x, bot_menu_y, 2 * sq_size, sq_size)
        self._arrows_menu.connect(self._gear_wheel)
        # area setting up :
        bot_menu_y -= 3 * 3 * sq_size
        scaling_area = Area(bot_menu_x, bot_menu_y, delta, sq_size, line_w, f'Sizes in tiles',
                            {k: f'{v}x{self._grid.get_hor_tiles(k)}' for k, v in self._grid.scale_names.items()})
        scaling_area.connect_to_func(self._grid.set_scale)
        # menu composing connecting to an icon:
        self._settings_menu = Menu()
        self._settings_menu.append_area(scaling_area)
        self._settings_menu.connect(self._gear_wheel)
        # TWO AUX AREAS:
        bot_menu_y = 250
        self._guide_arrows_area = Area(bot_menu_x, bot_menu_y, delta, sq_size, line_w, f'Guide arrows',
                                       self._guide_arrows_names)
        self._guide_arrows_area.connect_to_func(self._grid.set_guide_arrows_ind)
        bot_menu_y -= 30 + 3 * sq_size
        self._show_mode_area = Area(bot_menu_x, bot_menu_y, delta, sq_size, line_w, f'Show mode',
                                    self._interactive_names)
        self._show_mode_area.connect_to_func(self.set_interactive_ind)
        # menu setting up:
        # MANAGE MENU:
        ...

    def set_interactive_ind(self, ind: int or None):
        self._interactive_ind = ind

    # PRESETS:
    def setup(self):
        # algos' menus/icons:
        for icon in self._icons_dict.values():
            icon.setup()
        for menu in self._menus_dict.values():
            if menu is not None:
                menu.setup()
        self._arrows_menu.setup()
        self._guide_arrows_area.setup()
        self._show_mode_area.setup()
        # grid:
        self._grid.setup()
        # let us make the new mouse cursor:
        cursor = self.get_system_mouse_cursor(self.CURSOR_HAND)
        self.set_mouse_cursor(cursor)
        # let us change the app icon:
        self.set_icon(pyglet.image.load('C:\\Users\\langr\\PycharmProjects\\Lastar\\366.png'))
        # manage icons setting up:
        ...

    # CLEARING/REBUILDING:
    def rebuild_map(self):
        self._grid.tile_size, self._grid.hor_tiles_q = self.get_pars()
        # grid's renewing:
        self._grid.initialize()
        # pars resetting:
        self.aux_clear()
        self._grid.start_node = None
        self._grid.end_node = None
        self._grid.node_chosen = None
        self._grid.node_sprite_list = arcade.SpriteList()
        self._grid.grid_line_shapes = arcade.ShapeElementList()
        self._walls_manager.walls = set()
        self._grid.setup()

    # clears all the nodes except start, end and walls
    def clear_empty_nodes(self):
        # clearing the every empty node:
        for row in self._grid.grid:
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
        for row in self._grid.grid:
            for node in row:
                node.clear()
        # clearing the nodes-relating pars of the game:
        self._grid._start_node, self._grid._end_node = None, None
        self.aux_clear()
        self._walls_manager.walls = set()

    def aux_clear(self):
        # grid's pars clearing:
        self._grid.triangle_shape_list = arcade.ShapeElementList()  # <<-- for more comprehensive path visualization
        self._grid.arrow_shape_list = arcade.ShapeElementList()  # <<-- for more comprehensive algorithm's visualization
        # builder clearing:
        self._walls_manager.walls_built_erased = [([], True)]
        self._walls_manager.walls_index = 0
        # Lastar clearing:
        self._in_interaction = False
        self._in_interaction_mode_lock = False
        # algo pars clearing:
        if self._current_algo is not None:
            self._current_algo.clear()

    # UPDATING:
    # long press logic for mouse buttons, incrementers changing for living icons and so on:
    def update(self, delta_time: float):
        # algos' icons:
        for icon in self._icons_dict.values():
            icon.update()
        for menu in self._menus_dict.values():
            if menu is not None:
                menu.update()
        # manage icons:
        ...

    def on_draw(self):
        # renders this screen:
        arcade.start_render()
        # GRID:
        self._grid.draw()
        # HINTS:
        ...
        arcade.Text(f'Mode: {self._mode_names[self._mode]}', 25, SCREEN_HEIGHT - 35, arcade.color.BLACK,
                    bold=True).draw()
        ...
        # ICONS and MENUS:
        for icon in self._icons_dict.values():
            icon.draw()
        for menu in self._menus_dict.values():
            if menu is not None:
                menu.draw()
        self._arrows_menu.draw()
        self._guide_arrows_area.draw()
        self._show_mode_area.draw()
        # NODE CHOSEN:
        ...
        # MANAGE ICONS:
        ...

    # KEYBOARD:
    def on_key_press(self, symbol: int, modifiers: int):
        # is called when user press the symbol key:
        match symbol:
            # a_star_call:
            case arcade.key.SPACE:
                if not self._loading:
                    if self._grid.start_node and self._grid.end_node:
                        # STEP BY STEP:
                        if self._interactive_ind is not None:  # TODO: add interactive area to wave_lee and bfs_dfs!!!
                            # game logic:
                            self._in_interaction = True
                            # prepare:
                            ...

    def on_key_release(self, symbol: int, modifiers: int):
        ...

    def on_mouse_motion(self, x: int, y: int, dx: int, dy: int):
        for icon in self._icons_dict.values():
            icon.on_motion(x, y)
        for menu in self._menus_dict.values():
            if menu is not None:
                menu.on_motion(x, y)
        self._arrows_menu.on_motion(x, y)
        self._guide_arrows_area.on_motion(x, y)
        self._show_mode_area.on_motion(x, y)

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int):
        for icon in self._icons_dict.values():
            if icon.on_press(x, y) is not None:
                self.clear_icons_inter_types(icon)
        for menu in self._menus_dict.values():
            if menu is not None:
                menu.on_press(x, y)
        self._arrows_menu.on_press(x, y)
        self._guide_arrows_area.on_press(x, y)
        self._show_mode_area.on_press(x, y)

    def on_mouse_release(self, x: int, y: int, button: int, modifiers: int):
        ...

    # game mode switching by scrolling the mouse wheel:
    def on_mouse_scroll(self, x: int, y: int, scroll_x: int, scroll_y: int):
        self._mode = (self.mode + 1) % len(self.mode_names)

    def clear_icons_inter_types(self, icon_chosen: 'Icon'):
        for icon in self._icons_dict.values():
            if icon != icon_chosen:
                icon.clear_inter_type()

    @staticmethod
    def get_ms(start, finish):
        return (finish - start) // 10 ** 6


class DrawLib:
    @staticmethod
    # by default the arrow to be drawn is left sided:
    def create_line_arrow(node: 'Node', deltas: tuple[int, int] = (-1, 0),
                          grid: 'Grid' = None):  # left arrow by default
        cx, cy = 5 + node.x * grid.tile_size + grid.tile_size / 2, 5 + node.y * grid.tile_size + grid.tile_size / 2
        h = 2 * grid.tile_size // 3
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

    # helpful auxiliary methods:
    @staticmethod
    def is_point_in_square(cx, cy, size, x, y):
        return cx - size / 2 <= x <= cx + size / 2 and cy - size / 2 <= y <= cy + size / 2

    @staticmethod
    def is_point_in_circle(cx, cy, r, x, y):
        return (cx - x) ** 2 + (cy - y) ** 2 <= r ** 2


class Drawable(ABC):

    # on initialization (loads presets):
    @abstractmethod
    def setup(self):
        ...

    # on every frame:
    @abstractmethod
    def update(self):
        ...

    # renders the element:
    @abstractmethod
    def draw(self):
        ...


class Interactable(ABC):

    # implements the element's behaviour in on_press/on_motion methods:
    @abstractmethod
    def on_motion(self, x, y):
        ...

    @abstractmethod
    def on_press(self, x, y):
        ...

    @abstractmethod
    def on_release(self, x, y):
        ...

    @abstractmethod
    def on_key_press(self, x, y):
        ...

    @abstractmethod
    def on_key_release(self, x, y):
        ...


class Connected(ABC):  # Connected
    def __init__(self):
        # object to be managed:
        self._obj = None

    @abstractmethod
    def connect(self, obj):
        self._obj = obj


class FuncConnected(ABC):
    def __init__(self):
        # function connected:
        self._func = None

    @abstractmethod
    def connect_to_func(self, func):
        self._func = func


class Grid(Drawable):

    def __init__(self):
        super().__init__()
        # the grid itself:
        self._grid = None
        # important nodes:
        self._start_node = None
        self._end_node = None
        # the current node chosen (for getting info):
        self._node_chosen = None
        # guide arrows ON/OFF:
        self._guide_arrows_ind = None
        # visualization:
        self._node_sprite_list = arcade.SpriteList()
        self._grid_line_shapes = arcade.ShapeElementList()
        # algo steps visualization:
        self._triangle_shape_list = arcade.ShapeElementList()  # <<-- for more comprehensive path visualization
        self._arrow_shape_list = arcade.ShapeElementList()  # <<-- for more comprehensive algorithm's visualization
        # sizes:
        self._X, self._Y = self.Y, self.X = SCREEN_HEIGHT - 60, SCREEN_WIDTH - 250
        # scaling:  TODO: add AI to calculate the sizes for every resolution possible:
        self._scale = 0
        self._scale_names = {0: 10, 1: 15, 2: 22, 3: 33, 4: 45, 5: 66, 6: 90,
                             7: 110}  # {0: 5, 1: 10, 2: 15, 3: 22, 4: 33, 5: 45, 6: 66, 7: 90, 8: 110, 9: 165, 10: 198}  # factors of 990 num
        # pars:
        self._tiles_q = None
        self._line_width = None
        self._tile_size, self._hor_tiles_q = self.get_pars()
        # WALLS MANAGER:
        # memoization for undo/redo area:
        self._walls_built_erased = [([], True)]  # TODO: swap to DICT!!!
        self._walls_index = 0
        self._walls = set()  # all walls located on the map at the time being
        # save/load:
        self._loading = False
        self._loading_ind = 0

    # INITIALIZATION AUX:
    # calculating grid visualization pars for vertical tiles number given:
    def get_pars(self):
        self._Y, self._X = SCREEN_HEIGHT - 60, SCREEN_WIDTH - 250
        self._tiles_q = self._scale_names[self._scale]
        self._line_width = int(math.sqrt(max(self._scale_names.values()) / self._tiles_q))
        tile_size = self.Y // self.tiles_q
        hor_tiles_q = self.X // tile_size
        self.Y, self.X = self.tiles_q * tile_size, hor_tiles_q * tile_size
        return tile_size, hor_tiles_q

    def get_hor_tiles(self, i):
        return (SCREEN_WIDTH - 250) // (
                (SCREEN_HEIGHT - 30) // self._scale_names[i])  # TODO: ELIMINATE THE DEVIATION IN Y coordinates!!!

    def set_scale(self, ind: int):
        self._scale = ind

    def set_guide_arrows_ind(self, ind: int):
        self._guide_arrows_ind = ind

    # AUX:
    # gets the number representation for the node:
    def number_repr(self, node: 'Node'):
        return node.y * self._hor_tiles_q + node.x

    # gets node's coordinates for its number representation:
    def coords(self, number: int):
        return divmod(number, self._hor_tiles_q)

    # gets the node itself for its number representation:
    def node(self, num: int) -> 'Node':
        y, x = self.coords(num)
        return self._grid[y][x]

    # gets the node from the current mouse coordinates:
    def get_node(self, mouse_x, mouse_y):
        x_, y_ = mouse_x - 5, mouse_y - 5
        x, y = x_ // self._tile_size, y_ // self._tile_size
        return self.grid[y][x] if 0 <= x < self._hor_tiles_q and 0 <= y < self.tiles_q else None

    @property
    def grid(self):
        return self._grid

    @property
    def start_node(self):
        return self._start_node

    @start_node.setter
    def start_node(self, start_node):
        self._start_node = start_node

    @property
    def end_node(self):
        return self._end_node

    @end_node.setter
    def end_node(self, end_node):
        self._end_node = end_node

    @property
    def node_chosen(self):
        return self._node_chosen

    @node_chosen.setter
    def node_chosen(self, node_chosen):
        self._node_chosen = node_chosen

    @property
    def tiles_q(self):
        return self._tiles_q

    @tiles_q.setter
    def tiles_q(self, tiles_q):
        self._tiles_q = tiles_q

    @property
    def hor_tiles_q(self):
        return self._hor_tiles_q

    @hor_tiles_q.setter
    def hor_tiles_q(self, hor_tiles_q):
        self._hor_tiles_q = hor_tiles_q

    @property
    def tile_size(self):
        return self._tile_size

    @tile_size.setter
    def tile_size(self, tile_size):
        self._tile_size = tile_size

    @property
    def line_width(self):
        return self._line_width

    @property
    def arrow_shape_list(self):
        return self._arrow_shape_list

    @arrow_shape_list.setter
    def arrow_shape_list(self, arrow_shape_list):
        self._arrow_shape_list = arrow_shape_list

    @property
    def triangle_shape_list(self):
        return

    @triangle_shape_list.setter
    def triangle_shape_list(self, triangle_shape_list):
        self._triangle_shape_list = triangle_shape_list

    @property
    def node_sprite_list(self):
        return self._node_sprite_list

    @node_sprite_list.setter
    def node_sprite_list(self, node_sprite_list):
        self._node_sprite_list = node_sprite_list

    @property
    def grid_line_shapes(self):
        return self._grid_line_shapes

    @grid_line_shapes.setter
    def grid_line_shapes(self, grid_line_shapes):
        self._grid_line_shapes = grid_line_shapes

    @property
    def scale_names(self):
        return self._scale_names

    @property
    def walls_built_erased(self):
        return self._walls_built_erased

    @walls_built_erased.setter
    def walls_built_erased(self, walls_built_erased):
        self._walls_built_erased = walls_built_erased

    @property
    def walls_index(self):
        return self._walls_index

    @walls_index.setter
    def walls_index(self, walls_index):
        self._walls_index = walls_index

    @property
    def walls(self):
        return self._walls

    @walls.setter
    def walls(self, walls):
        self._walls = walls

    def initialize(self):
        self._grid = [[Node(j, i, 1, NodeType.EMPTY) for i in range(self._hor_tiles_q)] for j in range(self._tiles_q)]

    # make a node the chosen one:
    def choose_node(self, node: 'Node'):
        self._node_chosen = node
        # draw a frame

    def setup(self):
        # initialization:
        self.initialize()
        # sprites, shapes and etc...
        # blocks:
        self.get_sprites()
        # grid lines:
        self.make_grid_lines()

    def update(self):
        pass

    def draw(self):
        # grid:
        self._grid_line_shapes.draw()
        # blocks:
        self._node_sprite_list.draw()
        # arrows:
        if self._guide_arrows_ind is not None:
            if len(self.arrow_shape_list) > 0:
                self.arrow_shape_list.draw()

    # creates sprites for all the nodes:
    def get_sprites(self):  # batch -->
        for row in self.grid:
            for node in row:
                node.get_solid_colour_sprite(self)

    # shaping shape element list of grid lines:
    def make_grid_lines(self):
        for j in range(self.tiles_q + 1):
            self._grid_line_shapes.append(
                arcade.create_line(5, 5 + self._tile_size * j, 5 + self.X, 5 + self._tile_size * j,
                                   arcade.color.BLACK,
                                   self._line_width))

        for i in range(self._hor_tiles_q + 1):
            self._grid_line_shapes.append(
                arcade.create_line(5 + self._tile_size * i, 5, 5 + self._tile_size * i, 5 + self.Y,
                                   arcade.color.BLACK,
                                   self._line_width))

    # WALLS MANAGER:
    # EMPTIES -->> WALLS and BACK:
    def change_nodes_type(self, node_type: 'NodeType', walls_set: set or list):
        for node_num in walls_set:
            y, x = self.coords(node_num)
            self.grid[y][x].type = node_type
            self.grid[y][x].update_sprite_colour()

    # builds/erases walls:
    def build_wall(self, x, y):
        # now building the walls:
        n = self.get_node(x, y)
        if n and n.type != NodeType.WALL:
            n.type = NodeType.WALL
            self._walls.add(self.number_repr(n))
            n.update_sprite_colour()
            if self._walls_index < len(self._walls_built_erased) - 1:
                self._walls_built_erased = self._walls_built_erased[:self._walls_index + 1]
            self._walls_built_erased.append(([self.number_repr(n)], True))
            self._walls_index += 1

    def erase_wall(self, x, y):
        # now erasing the walls:
        n = self.get_node(x, y)
        if n and n.type == NodeType.WALL:
            n.type = NodeType.EMPTY
            self._walls.remove(self.number_repr(n))
            n.update_sprite_colour()
            if self._walls_index < len(self._walls_built_erased) - 1:
                self._walls_built_erased = self._walls_built_erased[:self._walls_index + 1]
            self._walls_built_erased.append(([self.number_repr(n)], False))
            self._walls_index += 1

    # erases all nodes, that are connected vertically, horizontally or diagonally to a chosen one,
    # then nodes connected to them the same way and so on recursively...

    def erase_all_linked_nodes(self, node: 'Node'):
        node.type = NodeType.EMPTY
        node.update_sprite_colour()
        self._walls.remove(self.number_repr(node))
        self._walls_built_erased[self._walls_index][0].append(self.number_repr(node))
        for neigh in node.get_extended_neighs(
                self):  # TODO: FIT .get_extended_neighs() method in Node class!!!
            if neigh.type == NodeType.WALL:
                self.erase_all_linked_nodes(neigh)

    # undo/redo manager:
    def undo(self):
        if self._walls_index > 0:
            for num in (l := self._walls_built_erased[self._walls_index])[0]:
                node = self.node(num)
                node.type = NodeType.EMPTY if l[1] else NodeType.WALL
                node.update_sprite_colour()
                if l[1]:
                    self._walls.remove(self.number_repr(node))
                else:
                    self._walls.add(self.number_repr(node))
            self._walls_index -= 1

    def redo(self):
        if self._walls_index < len(self._walls_built_erased) - 1:
            for num in (l := self._walls_built_erased[self._walls_index + 1])[0]:
                node = self.node(num)
                node.type = NodeType.WALL if l[1] else NodeType.EMPTY
                node.update_sprite_colour()
                if l[1]:
                    self._walls.add(self.number_repr(node))
                else:
                    self._walls.remove(self.number_repr(node))
            self._walls_index += 1

    # save/load area:
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
    def get_solid_colour_sprite(self, grid: Grid):
        cx, cy, size, colour = self.get_center_n_sizes(grid)
        self.sprite = arcade.SpriteSolidColor(size, size, colour)
        self.sprite.center_x, self.sprite.center_y = cx, cy
        grid.node_sprite_list.append(self.sprite)

    # aux calculations:
    def get_center_n_sizes(self, grid: Grid):
        return (5 + grid.tile_size * self.x + grid.tile_size / 2,
                5 + grid.tile_size * self.y + grid.tile_size / 2,
                grid.tile_size - 2 * grid.line_width - (1 if grid.line_width % 2 != 0 else 0),
                self.type.value)

    # updates the sprite's color (calls after node's type switching)
    def update_sprite_colour(self):
        self.sprite.color = self.type.value

    # appends the arrow shape of the node to the arrow_shape_list in Astar class
    def append_arrow(self, grid: Grid):
        grid.arrow_shape_list.append(self.arrow_shape)

    # removes the arrow shape of the node from the arrow_shape_list in Astar class
    def remove_arrow(self, grid: Grid):
        grid.arrow_shape_list.remove(self.arrow_shape)
        self.arrow_shape = None

    def remove_arrow_from_shape_list(self, grid: Grid):
        grid.arrow_shape_list.remove(self.arrow_shape)

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
    def get_neighs(self, grid: Grid, forbidden_node_types: list['NodeType']):  # has become smarter
        for dy, dx in self.walk:
            ny, nx = self.y + dy, self.x + dx
            if 0 <= ny < grid.tiles_q and 0 <= nx < grid.hor_tiles_q:
                # by default can visit the already visited nodes
                if grid.grid[ny][nx].type not in forbidden_node_types:
                    yield grid.grid[ny][nx]

    # gets extended neighs (with diagonal ones) of the node, generator:
    def get_extended_neighs(self, grid: Grid) -> list['Node']:
        for dy, dx in self.extended_walk:
            ny, nx = self.y + dy, self.x + dx
            if 0 <= ny < grid.tiles_q and 0 <= nx < grid.hor_tiles_q:
                yield grid.grid[ny][nx]


# class representing an algo:
class Algorithm(Connected):
    def __init__(self, name: str):
        super().__init__()
        # info:
        self._name = name
        # path:
        self._path = None
        self._path_index = 0
        # iterations and time:
        self._iterations = 0
        self._time_elapsed_ms = 0

    def connect(self, grid: Grid):
        self._obj = grid

    def base_clear(self):
        # visualization:
        self._obj.triangle_shape_list = arcade.ShapeElementList()  # <<-- for more comprehensive path visualization
        self._obj.arrow_shape_list = arcade.ShapeElementList()  # <<-- for more comprehensive algorithm's visualization
        # path:
        self._path = None
        self._path_index = 0
        # iterations and time:
        self._iterations = 0
        self._time_elapsed_ms = 0

    @abstractmethod
    def clear(self):
        ...

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
        self._obj.triangle_shape_list.append(triangle_shape)  # NOT A BUG!!!
        # line arrows removing:
        node = self._path[self._path_index + 1]
        # if self.inter_types[2] == InterType.PRESSED:
        if node not in [self._obj.start_node, self._obj.end_node]:
            node.remove_arrow_from_shape_list(self)

    # path-recovering process for interactive a_star:
    def recover_path(self):
        # start point of path restoration (here we begin from the end node of the shortest path found):
        node = self._obj.end_node
        shortest_path = []
        # path restoring (here we get the reversed path):
        while node.previously_visited_node:
            shortest_path.append(node)
            node = node.previously_visited_node
        shortest_path.append(self._obj.start_node)
        # returns the result:
        return shortest_path

    def path_down(self):
        if (path_node := self._path[self._path_index]).type not in [NodeType.START_NODE, NodeType.END_NODE]:
            path_node.type = NodeType.VISITED_NODE
            path_node.update_sprite_colour()
            # arrows:
        self._obj.triangle_shape_list.remove(self._obj.triangle_shape_list[self._path_index - 1])  # NOT A BUG!!!
        # line arrows restoring:
        if path_node not in [self._obj.start_node, self._obj.end_node]:
            path_node.append_arrow(self._obj)

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
        scaled_point = point[0] * (self._obj.tile_size // 2 - 2), point[1] * (self._obj.tile_size // 2 - 2)
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
        cx, cy = 5 + node.x * self._obj.tile_size + self._obj.tile_size / 2, \
                 5 + node.y * self._obj.tile_size + self._obj.tile_size / 2
        return (cx, cy), (cx + deltas[0][0], cy + deltas[0][1]), (cx + deltas[1][0], cy + deltas[1][1])


class Astar(Algorithm):

    def __init__(self):
        super().__init__('Astar')
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
        self._tiebreaker = None
        self._greedy_ind = None  # is algorithm greedy?
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
        self._max_times_visited = 0

    def clear(self):
        self.base_clear()
        # 3. visiting:
        self._nodes_visited = {}
        # 5. interactive a_star pars:
        self._curr_node_dict = {}
        self._max_times_visited_dict = {0: 0}
        self._neighs_added_to_heap_dict = {}

    def set_heuristic(self, ind: int or None):
        self._heuristic = ind

    def set_tiebreaker(self, ind: int or None):
        self._tiebreaker = ind

    def set_greedy_ind(self, ind: int or None):
        self._greedy_ind = ind

    def prepare(self):
        # heap:
        self._nodes_to_be_visited = [self._obj.start_node]
        hq.heapify(self._nodes_to_be_visited)
        # heur/cost:
        self._obj.start_node.g = 0
        # transmitting the greedy flag to the Node class: TODO: fix this strange doing <<--
        Node.IS_GREEDY = False if self._greedy_ind is None else True
        # important pars and dicts:
        self._iterations = 0
        self._neighs_added_to_heap_dict = {0: [self._obj.start_node]}
        self._curr_node_dict = {0: None}
        self._max_times_visited_dict = {0: 0}
        # path nullifying:
        self._path = None
        # SETTINGS menu should be closed during the algo's interactive phase!!!
        # arrows list renewal:
        self._obj.arrow_shape_list = arcade.ShapeElementList()

    def algo_up(self):
        if self._iterations == 0:
            self._nodes_to_be_visited = [self._obj.start_node]
        self._neighs_added_to_heap_dict[self._iterations + 1] = []  # memoization
        # popping out the most priority node for a_star from the heap:
        self._curr_node_dict[self._iterations + 1] = hq.heappop(self._nodes_to_be_visited)  # + memoization
        curr_node = self._curr_node_dict[self._iterations + 1]
        if self._iterations > 0 and curr_node != self._obj.end_node:
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
        if curr_node == self._obj.end_node:
            self._path = self.recover_path()
        # next step:
        # we can search for neighs on the fly or use precalculated sets (outdated):
        for neigh in curr_node.get_neighs(self._obj, [NodeType.WALL]):  # getting all the neighs 'on the fly;
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
                neigh.h = neigh.heuristics[self._heuristic](neigh, self._obj.end_node)
                # tie-breaking:
                if self._tiebreaker is not None:
                    neigh.tiebreaker = self._obj.start_node.tiebreakers[self._tiebreaker](
                        self._obj.start_node, self._obj.end_node, neigh)
                # previous visited node memoization for further path-recovering process:
                neigh.previously_visited_node = curr_node
                if neigh.type not in [NodeType.START_NODE, NodeType.END_NODE]:  # neigh not in self.nodes_visited and
                    neigh.type = NodeType.NEIGH
                    neigh.update_sprite_colour()
                    arrow = DrawLib.create_line_arrow(neigh, (neigh.x - curr_node.x, neigh.y - curr_node.y),
                                                      self._obj)
                    # here the arrow rotates (re-estimating of neigh g-cost):
                    if neigh.arrow_shape is not None:
                        neigh.remove_arrow(self._obj)
                    neigh.arrow_shape = arrow
                    neigh.append_arrow(self._obj)
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
                node = self._obj.grid[y][x]
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
                        node.remove_arrow(self._obj)
                if node.type == NodeType.NEIGH:
                    # here the arrow rotates backwards:
                    arrow = DrawLib.create_line_arrow(node, (
                        node.x - node.previously_visited_node.x, node.y - node.previously_visited_node.y), self._obj)
                    node.arrow_shape = arrow
                    node.append_arrow(self._obj)
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

    def full_algo(self):
        # False if game.greedy_ind is None else True
        self._nodes_to_be_visited = [self._obj.start_node]
        self._obj.start_node.g = 0
        hq.heapify(self._nodes_to_be_visited)
        self._max_times_visited = 0
        # the main cycle:
        while self._nodes_to_be_visited:
            self._iterations += 1
            curr_node = hq.heappop(self._nodes_to_be_visited)
            if curr_node not in [self._obj.start_node, self._obj.end_node]:
                curr_node.type = NodeType.VISITED_NODE
                curr_node.update_sprite_colour()
            curr_node.times_visited += 1
            self._max_times_visited = max(self._max_times_visited, curr_node.times_visited)
            self._nodes_visited[curr_node] = 1
            # base case of finding the shortest path:
            if curr_node == self._obj.end_node:
                break
            # next step:
            for neigh in curr_node.get_neighs(self._obj, [NodeType.WALL]):
                if neigh.g > curr_node.g + neigh.val:
                    neigh.g = curr_node.g + neigh.val
                    neigh.h = neigh.heuristics[self._heuristic](neigh, self._obj.end_node)
                    if self._tiebreaker is not None:
                        neigh.tiebreaker = self._obj.start_node.tiebreakers[self._tiebreaker](self, self._obj.end_node,
                                                                                              neigh)
                    neigh.previously_visited_node = curr_node
                    hq.heappush(self._nodes_to_be_visited, neigh)


class WaveLee(Algorithm):
    def __init__(self):
        super().__init__('WaveLee')
        # wave lee algo's important pars:
        self._front_wave_lee = None
        self._next_wave_lee = None
        self._fronts_dict = None

    def clear(self):
        self.base_clear()
        # wave lee algo's important pars:
        self._front_wave_lee = None
        self._next_wave_lee = None
        self._fronts_dict = None

    def prepare(self):
        # starting attributes' values:
        self._front_wave_lee = []
        self._next_wave_lee = [self._obj.start_node]
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
            if curr_node not in [self._obj.end_node, self._obj.start_node]:
                curr_node.type = NodeType.VISITED_NODE
                curr_node.update_sprite_colour()
            if curr_node == self._obj.end_node:
                # self._end_node.val = self._iterations
                self._path = self.recover_path()
                break
            for neigh in curr_node.get_neighs(self, [NodeType.START_NODE, NodeType.WALL, NodeType.VISITED_NODE]):
                if neigh.type == NodeType.EMPTY:  # it is equivalent to if neigh.val == 1, TODO: decide if is it needed???
                    if neigh not in self._next_wave_lee:
                        if neigh != self._obj.end_node:
                            neigh.type = NodeType.NEIGH
                            neigh.update_sprite_colour()
                            arrow = DrawLib.create_line_arrow(
                                neigh,
                                (neigh.x - curr_node.x, neigh.y - curr_node.y),
                                self._obj
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
                if neigh not in [self._obj.start_node, self._obj.end_node]:
                    neigh.type = NodeType.EMPTY
                    neigh.update_sprite_colour()
                    neigh.remove_arrow(self)
            if self._iterations != 0:
                # the front nodes have become NEIGHS:
                for node in self._front_wave_lee:
                    if node not in [self._obj.start_node, self._obj.end_node]:
                        node.type = NodeType.NEIGH
                        node.update_sprite_colour()
                # current and next fronts stepping back:
                self._next_wave_lee = self._front_wave_lee[:]
                self._front_wave_lee = self._fronts_dict[self._iterations]
            else:
                # the starting point:
                self._next_wave_lee = [self._obj.start_node]
                self._front_wave_lee = []

    def full_algo(self):
        # other.get_neighs(game)  # Why is it necessary???
        front_wave = {self._obj.start_node}
        iteration = 0
        # wave-spreading:
        while front_wave:
            iteration += 1
            new_front_wave = set()
            for front_node in front_wave:
                # front_node.val = iteration
                if front_node not in [self._obj.start_node, self._obj.end_node]:
                    front_node.type = NodeType.VISITED_NODE
                    front_node.update_sprite_colour()
                if front_node == self._obj.end_node:
                    return self.recover_path()
                for front_neigh in front_node.get_neighs(
                        self._obj,
                        [NodeType.START_NODE, NodeType.VISITED_NODE, NodeType.WALL]):
                    if front_neigh not in new_front_wave:
                        front_neigh.previously_visited_node = front_node
                        new_front_wave.add(front_neigh)
            front_wave = set() | new_front_wave
        return []


class BfsDfs(Algorithm):
    def __init__(self):
        super().__init__('Bfs/Dfs')
        # important algo's attributes:
        self._queue = None
        self._is_bfs = True
        # dicts:
        self._curr_node_dict = {}
        self._neighs_added_to_heap_dict = {}

    @property
    def bfs_dfs_ind(self):
        return 0 if self._is_bfs else 1

    def clear(self):
        self.base_clear()
        # important algo's attributes:
        self._queue = None
        self._is_bfs = True
        # dicts:
        self._curr_node_dict = {}
        self._neighs_added_to_heap_dict = {}

    def set_is_bfs(self, ind: int or None):
        self._is_bfs = (ind == 0)

    def prepare(self):
        self._queue = deque()
        self._queue.append(self._obj.start_node)
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
        if curr_node == self._obj.end_node:
            self._path = self.recover_path()
        self._neighs_added_to_heap_dict[self._iterations + 1] = set()
        for neigh in curr_node.get_neighs(self, [NodeType.START_NODE, NodeType.VISITED_NODE, NodeType.WALL] + (
                [NodeType.NEIGH] if self._is_bfs else [])):
            if neigh.type != NodeType.END_NODE:
                # at first memoization for further 'undoing':
                self._neighs_added_to_heap_dict[self._iterations + 1].add(neigh.aux_copy())
                # then changing neigh's pars:
                neigh.type = NodeType.NEIGH
                neigh.update_sprite_colour()
                arrow = DrawLib.create_line_arrow(neigh, (neigh.x - curr_node.x, neigh.y - curr_node.y),
                                                  self._obj)
                if neigh.arrow_shape is not None:
                    neigh.remove_arrow(self)
                neigh.arrow_shape = arrow
                neigh.append_arrow(self)
            neigh.previously_visited_node = curr_node
            # BFS:
            if self._is_bfs:
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
                node = self._obj.grid[y][x]
                node.restore(neigh)
                if node not in [self._obj.start_node, self._obj.end_node]:
                    node.remove_arrow(self._obj)
                if node.type == NodeType.NEIGH:
                    # here the arrow rotates backwards:
                    arrow = DrawLib.create_line_arrow(
                        node,
                        (
                            node.x - self._curr_node_dict[self._iterations].x,
                            node.y - self._curr_node_dict[self._iterations].y
                        ),
                        self._obj
                    )
                    node.arrow_shape = arrow
                    node.append_arrow(self._obj)
                    # deque changing:
                    # BFS:
                    if self._is_bfs:
                        self._queue.popleft()
                    # DFS:
                    else:
                        self._queue.pop()
            # current node has become the NEIGH:
            curr_node = self._curr_node_dict[self._iterations]
            if curr_node not in [self._obj.start_node, self._obj.end_node]:
                curr_node.type = NodeType.NEIGH
                curr_node.update_sprite_colour()
            # adding back to the deque:
            # BFS & DFS:
            self._queue.append(curr_node)
            if self._iterations > 1:
                # previous step current node has become the current step current node:
                prev_node = self._curr_node_dict[self._iterations - 1]
                if prev_node not in [self._obj.start_node, self._obj.end_node]:
                    prev_node.type = NodeType.CURRENT_NODE
                    prev_node.update_sprite_colour()
            # step back:
            self._iterations -= 1

    def full_algo(self):
        queue = deque()
        queue.append(self._obj.start_node)
        while queue:
            self._iterations += 1
            current_node = queue.pop()
            if current_node.type not in [NodeType.START_NODE, NodeType.END_NODE]:
                current_node.type = NodeType.VISITED_NODE
                current_node.update_sprite_colour()
                current_node.times_visited += 1
            if current_node == self._obj.end_node:
                return self.recover_path()
            for neigh in current_node.get_neighs(self._obj,
                                                 [NodeType.START_NODE, NodeType.VISITED_NODE, NodeType.WALL] + (
                                                         [NodeType.NEIGH] if self._is_bfs else [])):
                if neigh.type != NodeType.END_NODE:
                    neigh.type = NodeType.NEIGH
                    neigh.update_sprite_colour()
                neigh.previously_visited_node = current_node
                # BFS:
                if self._is_bfs:
                    queue.appendleft(neigh)
                # DFS:
                else:
                    queue.append(neigh)


class Menu(Drawable, Interactable, Connected):

    def __init__(self):
        super().__init__()
        self._areas = []

    @property
    def areas(self):
        return self._areas

    def connect(self, icon: 'Icon'):
        self._obj = icon

    def append_area(self, area: 'Area'):
        self._areas.append(area)

    def multiple_append(self, *areas: 'Area'):
        for area in areas:
            self.append_area(area)

    def update(self):
        pass

    def setup(self):
        for area in self._areas:
            area.setup()

    def draw(self):
        if self._obj.inter_type == InterType.PRESSED:
            for area in self._areas:
                area.draw()

    def on_motion(self, x, y):
        pass

    def on_press(self, x, y):
        for area in self._areas:
            area.on_press(x, y)

    def on_release(self, x, y):
        pass

    def on_key_press(self, x, y):
        pass

    def on_key_release(self, x, y):
        pass


class Area(Drawable, Interactable, FuncConnected):

    def __init__(self, cx, cy, delta, sq_size, sq_line_w, header: str, fields: dict[int, str], no_choice=False):
        super().__init__()
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
        # logic flags:
        self._no_choice = no_choice

    def connect_to_func(self, _function):
        self._func = _function

    def choose_field(self, field_chosen_ind: int):
        if 0 <= field_chosen_ind < len(self._fields):
            self._field_chosen = field_chosen_ind
        else:
            raise IndexError(
                f"Wrong field's index: {field_chosen_ind} for an area, expected to be in range: [{0},{len(self._fields)})")

    def lock(self):
        self._is_locked = True

    def unlock(self):
        self._is_locked = False

    # presets:
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

    def update(self):
        pass

    # on every frame:
    def draw(self):
        self._header_text.draw()
        self._rectangle_shapes.draw()

        for text in self._field_texts:
            text.draw()

        if self._field_chosen is not None:
            arcade.draw_rectangle_filled(self._cx + 10, self._cy - 30 - self._delta * self._field_chosen,
                                         14, 14, arcade.color.BLACK)
        # lock:
        if self._is_locked:
            if self._field_chosen is not None:
                self.draw_lock(self._cx + 10, self._cy - 30 - self._delta * self._field_chosen)
            for i in range(len(self._fields)):
                if self._field_chosen is None or i != self._field_chosen:
                    self.draw_cross(self._cx + 10, self._cy - 30 - self._delta * i)

    def on_motion(self, x, y):
        pass

    def on_press(self, x, y):
        for i in range(len(self._fields)):
            if DrawLib.is_point_in_square(
                    self._cx + 10,
                    self._cy - 30 - self._delta * i,
                    self._sq_size,
                    x, y):                                            # 36 366 98 989
                if self._field_chosen is not None:
                    if i == self._field_chosen:
                        if self._no_choice:
                            self._field_chosen = None

                    else:
                        self._field_chosen = i
                else:
                    self._field_chosen = i
                # set the index in the algo:
                self._func(self._field_chosen)

    def on_release(self, x, y):
        pass

    def on_key_press(self, x, y):
        pass

    def on_key_release(self, x, y):
        pass

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

    def clear_inter_type(self):
        self._inter_type = InterType.NONE

    def is_pressed(self):
        return self._inter_type == InterType.PRESSED

    @property
    def inter_type(self):
        return self._inter_type


class PlayButton(Icon, Drawable, Interactable, Connected):
    DELTAS = [0.5, 0.015]  # pixels/radians

    def __init__(self, cx, cy, r, line_w):
        super().__init__(cx, cy)
        self._incrementer = [0, 0]
        self._r = r
        self._line_w = line_w
        self._multiplier = 1

    def connect(self, obj):
        pass

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
            if DrawLib.is_point_in_circle(self._cx, self._cy, self._r, x, y):
                self._inter_type = InterType.HOVERED
            else:
                self._inter_type = InterType.NONE
        else:
            if DrawLib.is_point_in_circle(self._cx, self._cy, self._r, x, y):
                self._multiplier = 1
            else:
                self._multiplier = -1

    def on_press(self, x, y):
        # if self._inter_type != InterType.PRESSED:
        if DrawLib.is_point_in_circle(self._cx, self._cy, self._r, x, y):
            if self._inter_type == InterType.PRESSED:
                self._inter_type = InterType.HOVERED
            elif self._inter_type == InterType.HOVERED:
                self._inter_type = InterType.PRESSED

    def on_release(self, x, y):
        pass

    def on_key_press(self, x, y):
        pass

    def on_key_release(self, x, y):
        pass


class StepButton(Icon, Drawable, Interactable, Connected):
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

    def connect(self, obj):
        pass

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

    def on_release(self, x, y):
        self._inter_type = InterType.NONE
        self._cycle_breaker = False
        self._incrementer[3] = 0

    # keys:
    def on_key_press(self, x, y):
        self._inter_type = InterType.PRESSED
        self._cycle_breaker = True

    def on_key_release(self, x, y):
        self._inter_type = InterType.NONE
        self._cycle_breaker = False
        self._incrementer[3] = 0


class Eraser(Icon, Drawable, Interactable, Connected):

    def __init__(self, cx, cy, h, w, r, line_w):
        super().__init__(cx, cy)
        self._h = h
        self._w = w
        self._r = r
        self._line_w = line_w
        self._centers = []

    def connect(self, obj):
        pass

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

    def on_key_press(self, x, y):
        pass

    def on_key_release(self, x, y):
        pass


class Undo(Icon, Drawable, Interactable, Connected):

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

    def connect(self, obj):
        pass

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
            if DrawLib.is_point_in_circle(self._cx, self._cy, self._r + self._dh, x, y) or arcade.is_point_in_polygon(x,
                                                                                                                      y,
                                                                                                                      self._vertices):
                self._inter_type = InterType.HOVERED
            else:
                self._inter_type = InterType.NONE

    def on_press(self, x, y):
        if DrawLib.is_point_in_circle(self._cx, self._cy, self._r + self._dh, x, y) or arcade.is_point_in_polygon(x, y,
                                                                                                                  self._vertices):
            self._inter_type = InterType.PRESSED

    def on_release(self, x, y):
        self._inter_type = InterType.NONE

    def on_key_press(self, x, y):
        pass

    def on_key_release(self, x, y):
        pass


class GearWheelButton(Icon, Drawable, Interactable):
    DELTA = 0.02

    def __init__(self, cx: int, cy: int, r: int, cog_size=8, multiplier=1.5, line_w=2, clockwise=True):
        super().__init__(cx, cy)
        self._r = r
        self._cog_size = cog_size
        self._multiplier = multiplier
        self._line_w = line_w
        self._clockwise = clockwise

    def setup(self):
        ...

    def update(self):
        if self._inter_type == InterType.HOVERED:
            self._incrementer += self.DELTA

    def draw(self):
        circumference = 2 * math.pi * self._r  # approximately if cog_size << radius
        angular_size = (2 * math.pi) * self._cog_size / circumference
        max_cogs_fit_in_the_gear_wheel = int(circumference / self._cog_size)
        cogs_q = max_cogs_fit_in_the_gear_wheel // 2
        fit_angular_size = (2 * math.pi - cogs_q * angular_size) / cogs_q
        angle = self._incrementer if self._clockwise else -self._incrementer  # in radians

        self._vertices = []

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
        arcade.draw_circle_filled(self._cx, self._cy, (self._r - self._multiplier * self._cog_size),
                                  arcade.color.DUTCH_WHITE)
        arcade.draw_circle_outline(self._cx, self._cy, (self._r - self._multiplier * self._cog_size),
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
            if self._inter_type != InterType.PRESSED:
                self._inter_type = InterType.PRESSED
                return self

    def on_release(self, x, y):
        pass

    def on_key_press(self, x, y):
        pass

    def on_key_release(self, x, y):
        pass


class ArrowsMenu(Icon, Drawable, Interactable, Connected):  # cx, cy = (1755, 785)
    arrows_indices = []
    walk_index = 0
    choosing_arrows = True

    def __init__(self, cx, cy, arrow_length, arrow_height):
        super().__init__(cx + 70, cy - 75)
        self._inter_types_arrows = [InterType.NONE for _ in range(4)]
        self._inter_type_reset_square = InterType.NONE

        self._arrows = []
        self._arrows_reset = None

        self._arrow_length = arrow_length
        self._arrow_height = arrow_height

        self.elements_setup()

    def elements_setup(self):
        for ind, (dx, dy) in enumerate(Arrow.walk):
            self._arrows.append(
                Arrow(
                    self._cx + dx * self._arrow_length,
                    self._cy + dy * self._arrow_length,
                    ind,
                    self._arrow_length,
                    self._arrow_height,
                    2, (dx, dy), arcade.color.SANDY_BROWN
                )
            )

        # ARROWS RESET:
        self._arrows_reset = ArrowReset(self._cx, self._cy, self._arrow_height)
        self._arrows_reset.connect(self)

    def connect(self, obj: GearWheelButton):
        self._obj = obj

    @property
    def inter_types_arrows(self):
        return self._inter_types_arrows

    @inter_types_arrows.setter
    def inter_types_arrows(self, inter_types_arrows):
        self._inter_types_arrows = inter_types_arrows

    def setup(self):
        for arrow in self._arrows:
            arrow.setup()
        self._arrows_reset.setup()

    def update(self):
        pass

    def draw(self):
        if self._obj.inter_type == InterType.PRESSED:
            # text:
            arcade.draw_text(f'Directions priority: ', SCREEN_WIDTH - 235, SCREEN_HEIGHT - 70 - 120, arcade.color.BLACK,
                             bold=True)
            # elements:
            for arrow in self._arrows:
                arrow.draw()
            self._arrows_reset.draw()

    def on_motion(self, x, y):
        for arrow in self._arrows:
            arrow.on_motion(x, y)
        self._arrows_reset.on_motion(x, y)

    def on_press(self, x, y):
        if self.choosing_arrows:
            for arrow in self._arrows:
                arrow.on_press(x, y)
        self._arrows_reset.on_press(x, y)

    def on_release(self, x, y):
        pass

    def on_key_press(self, x, y):
        pass

    def on_key_release(self, x, y):
        pass

    @property
    def arrows(self):
        return self._arrows


class Arrow(Icon, Drawable, Interactable):  # part of an arrow menu
    # initial directions priority for all algorithms:
    walk = [(1, 0), (0, -1), (-1, 0), (0, 1)]

    def __init__(self, cx, cy, index, arrow_length, arrow_height, line_w, point: tuple[int, int],
                 colour: tuple[int, int, int]):
        super().__init__(cx, cy)
        self._index = index
        self._arrow_length = arrow_length
        self._arrow_height = arrow_height
        self._line_w = line_w
        self._point = point
        self._colour = colour

    @property
    def _dx(self):  # TODO: this one should not be a protected property!!!
        return self._arrow_length / 2 - self._arrow_height

    @property
    def dx_(self):
        return self._arrow_length / 2

    @property
    def h_(self):
        return self._arrow_height / 2

    def setup(self):
        self._vertices = [
            [
                self._cx + self._point[0] * self._arrow_length / 2,
                self._cy + self._point[1] * self._arrow_length / 2
            ],
            [
                self._cx + self._point[0] * self._dx + self._point[1] * self._arrow_height,
                self._cy + self._point[1] * self._dx + self._point[0] * self._arrow_height
            ],
            [
                self._cx + self._point[0] * self._dx + self._point[1] * self.h_,
                self._cy + self._point[1] * self._dx + self._point[0] * self.h_
            ],
            [
                self._cx - self._point[0] * self.dx_ + self._point[1] * self.h_,
                self._cy - self._point[1] * self.dx_ + self._point[0] * self.h_
            ],
            [
                self._cx - self._point[0] * self.dx_ - self._point[1] * self.h_,
                self._cy - self._point[1] * self.dx_ - self._point[0] * self.h_
            ],
            [
                self._cx + self._point[0] * self._dx - self._point[1] * self.h_,
                self._cy + self._point[1] * self._dx - self._point[0] * self.h_
            ],
            [
                self._cx + self._point[0] * self._dx - self._point[1] * self._arrow_height,
                self._cy + self._point[1] * self._dx - self._point[0] * self._arrow_height
            ]
        ]

    def update(self):
        pass

    def draw(self):
        # index:
        ind = self.walk.index(self._point)
        # center coords:
        cx_ = self._cx - self._point[0] * self._arrow_height / 2
        cy_ = self._cy - self._point[1] * self._arrow_height / 2

        w, h = abs(self._point[0]) * (self._arrow_length - self._arrow_height) + abs(
            self._point[1]) * self._arrow_height, abs(
            self._point[0]) * self._arrow_height + abs(self._point[1]) * (self._arrow_length - self._arrow_height)

        if self._inter_type == InterType.PRESSED:
            arcade.draw_rectangle_filled(cx_, cy_, w, h, self._colour)

            arcade.draw_triangle_filled(
                self._cx + self._point[0] * self._dx + self._point[1] * self._arrow_height,
                self._cy + self._point[1] * self._dx + self._point[0] * self._arrow_height,
                self._cx + self._point[0] * self._dx - self._point[1] * self._arrow_height,
                self._cy + self._point[1] * self._dx - self._point[0] * self._arrow_height,
                self._cx + self._point[0] * self._arrow_length / 2,
                self._cy + self._point[1] * self._arrow_length / 2,
                self._colour
            )

            # numbers-hints:
            signed_delta = self._arrow_height / 2 - self._arrow_height / 12
            arcade.Text(
                f'{ArrowsMenu.arrows_indices.index(ind) + 1}',
                self._cx - self._arrow_height / 3 + self._arrow_height / 12 - self._point[0] * signed_delta,
                self._cy - self._arrow_height / 3 - self._point[1] * signed_delta, arcade.color.BLACK,
                2 * self._arrow_height / 3,
                bold=True).draw()

        arcade.draw_polygon_outline(
            self._vertices,
            arcade.color.BLACK,
            self._line_w if self._inter_type == InterType.NONE else self._line_w + 1
        )

    def on_motion(self, x, y):
        if self._inter_type != InterType.PRESSED:
            if arcade.is_point_in_polygon(x, y, self._vertices):
                self._inter_type = InterType.HOVERED
            else:
                self._inter_type = InterType.NONE

    def on_press(self, x, y):
        if arcade.is_point_in_polygon(x, y, self._vertices):
            if self._inter_type == InterType.HOVERED:
                self._inter_type = InterType.PRESSED
                ArrowsMenu.arrows_indices.append(self._index)
                ArrowsMenu.walk_index += 1
                if ArrowsMenu.walk_index == 4:
                    ArrowsMenu.choosing_arrows = False
                    # Node's directions choosing priority change (in .get_neighs() method):
                    Node.walk = [self.walk[ArrowsMenu.arrows_indices[_]] for _ in range(4)]

    def on_release(self, x, y):
        pass

    def on_key_press(self, x, y):
        pass

    def on_key_release(self, x, y):
        pass


class ArrowReset(Icon, Drawable, Interactable, Connected):

    def __init__(self, cx, cy, arrow_height):
        super().__init__(cx, cy)
        self._arrow_height = arrow_height

    def connect(self, obj: ArrowsMenu):
        self._obj = obj

    def setup(self):
        pass

    def update(self):
        pass

    def draw(self):
        arcade.draw_rectangle_outline(
            self._cx, self._cy,
            self._arrow_height,
            self._arrow_height,
            arcade.color.BLACK,
            2 + (0 if self._inter_type == InterType.NONE else 1)
        )

    def on_motion(self, x, y):
        if DrawLib.is_point_in_square(self._cx, self._cy, self._arrow_height, x, y):
            self._inter_type = InterType.HOVERED
        else:
            self._inter_type = InterType.NONE

    def on_press(self, x, y):
        if DrawLib.is_point_in_square(self._cx, self._cy, self._arrow_height, x, y):
            self._obj.choosing_arrows = True
            self._obj.walk_index = 0
            self._obj.arrows_indices = []
            for arrow in self._obj.arrows:
                arrow._inter_type = InterType.NONE

    def on_release(self, x, y):
        pass

    def on_key_press(self, x, y):
        pass

    def on_key_release(self, x, y):
        pass


class AstarIcon(Icon, Drawable, Interactable):
    DELTA = 0.05

    def __init__(self, cx, cy, size_w, size_h, line_w=2, clockwise=True):
        super().__init__(cx, cy)
        self._size_w = size_w
        self._size_h = size_h
        self._line_w = line_w
        self._clockwise = clockwise

    def setup(self):
        upper_hypot = math.sqrt(self._size_w ** 2 + self._size_h ** 2)
        cos, sin = self._size_h / upper_hypot, self._size_w / upper_hypot
        a_w = self._size_w / 3
        line_w_hour_projection = a_w / cos
        dh = a_w / sin
        delta = (self._size_h - a_w - dh) / 2
        k, upper_k = (delta - a_w / 2) * self._size_w / self._size_h, (delta + a_w / 2) * self._size_w / self._size_h

        # a_outer_points -> self._vertices[0], a_inner_points -> self._vertices[1]
        self._vertices.append(
            [
                [self._cx, self._cy],
                [self._cx + self._size_w, self._cy + self._size_h],
                [self._cx + 2 * self._size_w, self._cy],
                [self._cx + 2 * self._size_w - line_w_hour_projection, self._cy],
                [self._cx + 2 * self._size_w - line_w_hour_projection - k, self._cy + delta - a_w / 2],
                [self._cx + k + line_w_hour_projection, self._cy + delta - a_w / 2],
                [self._cx + line_w_hour_projection, self._cy]
            ]
        )

        self._vertices.append(
            [
                [self._cx + self._size_w, self._cy + self._size_h - dh],
                [self._cx + 2 * self._size_w - line_w_hour_projection - upper_k, self._cy + delta + a_w / 2],
                [self._cx + line_w_hour_projection + upper_k, self._cy + delta + a_w / 2]
            ]
        )

    def update(self):
        if self._inter_type == InterType.HOVERED:
            self._incrementer += self.DELTA

    def draw(self):
        # drawing A:
        self.draw_a(self._line_w)
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
    def draw_a(self, line_w):
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
            if self._inter_type != InterType.PRESSED:
                self._inter_type = InterType.PRESSED
                return self

    def on_release(self, x, y):
        pass

    def on_key_press(self, x, y):
        pass

    def on_key_release(self, x, y):
        pass


class Waves(Icon, Drawable, Interactable):
    DELTA = 0.25

    def __init__(self, cx, cy, size=32, waves_q=5, line_w=2):
        super().__init__(cx, cy)
        self._size = size
        self._waves_q = waves_q
        self._line_w = line_w

    def setup(self):
        pass

    def update(self):
        if self._inter_type == InterType.HOVERED:
            self._incrementer += self.DELTA

    def draw(self):
        ds = self._size / self._waves_q
        s_list = sorted([(i * ds + self._incrementer) % self._size for i in range(self._waves_q)], reverse=True)
        for i, curr_s in enumerate(s_list):
            if self._inter_type == InterType.PRESSED:
                arcade.draw_circle_filled(self._cx, self._cy, curr_s,
                                          arcade.color.RED if i % 2 == 0 else arcade.color.DUTCH_WHITE)
            arcade.draw_circle_outline(self._cx, self._cy, curr_s, arcade.color.BLACK,
                                       self._line_w + (0 if self._inter_type == InterType.NONE else 1))

    def on_motion(self, x, y):
        if self._inter_type != InterType.PRESSED:
            if DrawLib.is_point_in_circle(self._cx, self._cy, self._size, x, y):
                self._inter_type = InterType.HOVERED
            else:
                self._inter_type = InterType.NONE

    def on_press(self, x, y):
        if DrawLib.is_point_in_circle(self._cx, self._cy, self._size, x, y):
            if self._inter_type != InterType.PRESSED:
                self._inter_type = InterType.PRESSED
                return self

    def on_release(self, x, y):
        pass

    def on_key_press(self, x, y):
        pass

    def on_key_release(self, x, y):
        pass


class BfsDfsIcon(Icon, Drawable, Interactable):
    DELTA = 0.15

    def __init__(self, cx, cy, size, line_w):
        super().__init__(cx, cy)
        self._size = size
        self._line_w = line_w

    def setup(self):
        pass

    def update(self):
        if self._inter_type == InterType.HOVERED:
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
        arcade.Text('D', self._cx - self._size / 3,
                    self._cy - text_size - text_size / 4 + magnitude * math.sin(self._incrementer),
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
            if DrawLib.is_point_in_square(self._cx, self._cy, self._size, x, y):
                self._inter_type = InterType.HOVERED
            else:
                self._inter_type = InterType.NONE

    def on_press(self, x, y):
        if DrawLib.is_point_in_square(self._cx, self._cy, self._size, x, y):
            if self._inter_type != InterType.PRESSED:
                self._inter_type = InterType.PRESSED
                return self

    def on_release(self, x, y):
        pass

    def on_key_press(self, x, y):
        pass

    def on_key_release(self, x, y):
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
