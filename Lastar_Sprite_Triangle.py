# -*- coding: utf-8 -*-

# sys:
from pathlib import Path

# functools:
import functools
from functools import reduce

# math:
import math
from typing import Any

import numpy as np

# logging
import logging
from logging import config

# data
import shelve
import time
from enum import Enum

# abstract classes:
from abc import ABC, abstractmethod

# queues:
from collections import deque
import heapq as hq

# graphics:
import arcade
import arcade.gui
from arcade import load_texture

# image/drawing:
import PIL.Image
import PIL.ImageOps
import PIL.ImageDraw

# windows:
import pyglet

# screen sizes:
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1050

# LOGGING CONSTANTS:
graphic_logging = False  # ОТРИСОВКА
pressing_motion_logging = False  # НАЖАТИЯ МЫШИ И КЛАВИШ И ПЕРЕМЕЩЕНИЕ МЫШИ
logic_math_logging = False  # ЛОГИКА АЛГОРИТМОВ, РАСЧЁТЫ И ПРОЦЕССИНГ


# decorator for void funcs, it returns the function's runtime in ms as int instead of void function's return value (None):
def timer(func):
    @functools.wraps(func)
    def _wrapper(*args, **kwargs):
        start = time.perf_counter()
        f = func(*args, **kwargs)
        runtime = 1000 * (time.perf_counter() - start)
        args[0].time_elapsed += runtime
        return f

    return _wrapper


# returns the runtime for recursive function or method:
def rec_timer(func):
    @functools.wraps(func)
    def _wrapper(*args, **kwargs):
        start = time.perf_counter()
        f = func(*args, **kwargs)
        _wrapper.runtime += 1000 * (time.perf_counter() - start)
        return f

    # initial state:
    _wrapper.runtime = 0
    return _wrapper


# decorator, runs function if not in interaction mode:
def lock(func):
    @functools.wraps(func)
    def _wrapper(*args, **kwargs):
        if not Lastar.is_in_interaction():
            return func(*args, **kwargs)

    return _wrapper


# logs non-recursive methods:
def logged(is_debug: bool = True, is_used: bool = True):
    """decorator-constructor for loggers"""

    # inner decorator
    def core(func):  # 36 366 98 989
        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            obj = args[0]  # <<-- self
            if type(func) != dict:
                name = func.__name__
                doc = func.__doc__
            else:
                name = [value.__name__ for value in func.values()]
                doc = [value.__doc__ for value in func.values()]
            # logging in two diff ways:
            foo = obj.log.debug if is_debug else obj.log.info
            foo(f'method(s) .{name}() of {obj.__class__} started')
            foo(f"method(s)' description: {doc}")
            f = func(*args, **kwargs)
            foo(f'method(s) .{name}() of {obj.__class__} successfully finished')
            return f

        return _wrapper if is_used else func

    return core


def class_logged(is_debug: bool = True):
    def core(cls):
        """logging all the methods in class, excluding the dunder ones:"""
        cls_method_list = [
            cls.__dict__[key] for key in cls.__dict__.keys() if
            callable(getattr(cls, key)) and not key.startswith("__")
            and key not in ['update', 'draw', 'on_draw', 'on_motion']
        ]
        for method in cls_method_list:
            logged_method = logged(is_debug)(method)
            setattr(cls, method.__name__, logged_method)
        return cls

    return core


def long_pressable(cls):
    """decorator for icons that should have the long-press logic
    adds new attributes such as:
    _cycle_breaker: bool -->> tumbler for turning icon's living mode on/off
    _ticks: int -->> time incrementer, when it is bigger than THRESHOLD, the icon comes alive
    _interactive_incr: int -->> distance incrementer, moving the icon's parts"""
    if not issubclass(cls, (Icon, Drawable, Interactable)):
        raise SyntaxError(f'long_pressable decorator cannot be used with {cls} class!..')
    # new attrs:
    long_press_attributes = {
        '_cycle_breaker': False,
        '_ticks': 0,
        '_interactive_incr': 0
    }
    # attrs adding:
    for attr_name, attr_val in long_press_attributes.items():
        setattr(cls, attr_name, attr_val)
    # returning new class
    return cls


# depth and calls for recursive function:
def counted(func):
    def reset():
        _wrapper.rec_depth = 0
        _wrapper.rec_calls = 0

    @functools.wraps(func)
    def _wrapper(*args, **kwargs):
        nonlocal _depth
        # what for?..
        if _depth == 0:
            reset()
        # depth and calls incrementation:
        _depth += 1
        _wrapper.rec_calls += 1
        # max depth defining:
        _wrapper.rec_depth = max(_wrapper.rec_depth, _depth)
        f = func(*args, **kwargs)
        # depth backtracking:
        _depth -= 1
        return f

    # starts a wrapper:
    _depth = 0
    reset()
    return _wrapper


class Lastar(arcade.Window):
    logging.config.fileConfig('log.conf', disable_existing_loggers=True)
    # base interaction par:
    _in_interaction = False

    def __init__(self, width: int, height: int):
        super().__init__(width, height)
        arcade.set_background_color(arcade.color.DUTCH_WHITE)
        self.log = logging.getLogger('Lastar')
        self.set_update_rate(1 / 60)
        # sounds:
        self._player = pyglet.media.player.Player()
        # self._source = pyglet.media.load("", streaming=False)
        # self._player.queue(self._source)
        # time elapsed:
        self._time_elapsed = 0
        # interaction mode ON/OFF:
        self._interactive_ind = None
        ...
        self._in_interaction_mode_lock = False
        # names dicts:
        self._heuristic_names = {0: 'MANHATTAN', 1: 'EUCLIDIAN', 2: 'MAX_DELTA', 3: 'DIJKSTRA'}
        self._tiebreaker_names = {0: 'VECTOR_CROSS', 1: 'COORDINATES'}
        self._greedy_names = {0: 'IS_GREEDY'}
        self._guide_arrows_names = {0: f'ON/OFF'}
        self._interactive_names = {0: f'IS_INTERACTIVE'}
        # grid and walls_manager:
        self._grid = Grid()
        self._grid.connect_to_func(self.aux_clearing)
        # ALGOS:
        self._astar = None
        self._wave_lee = None
        self._bfs_dfs = None
        # current_algo:
        self._current_algo = None
        # self._current_icon ???
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
        # HINTS:
        self._mode_info = None
        # manage icons:
        self._play_button = None
        self._step_button_right = None
        self._step_button_left = None
        self._eraser = None
        self._undo_button = None
        self._redo_button = None
        self.elements_setup()

        # icons_dict:
        self._icons_dict = {
            0: self._gear_wheel,
            1: self._bfs_dfs_icon,
            2: self._astar_icon,
            3: self._wave_lee_icon
        }
        # menus_dict:
        self._menus_dict = {
            0: self._settings_menu,
            1: self._bfs_dfs_menu,
            2: self._astar_menu,
            3: self._wave_lee_menu
        }
        # manage icons dict:
        self._manage_icons_dict = {
            0: self._play_button,
            1: self._step_button_right,
            2: self._step_button_left,
            3: self._eraser,
            4: self._undo_button,
            5: self._redo_button
        }

    @staticmethod
    def is_in_interaction():
        return Lastar._in_interaction

    @logged()
    def elements_setup(self):
        """screen pars, algorithms, menus and icons set up, connectors activating"""
        # ALGOS:
        self._astar = Astar()
        self._astar.connect(self._grid)
        self._astar.connect_to_func(Node.set_greedy)
        self._wave_lee = WaveLee()
        self._wave_lee.connect(self._grid)
        self._bfs_dfs = BfsDfs()
        self._bfs_dfs.connect(self._grid)
        # pars:
        sq_size, line_w = 18, 2
        delta = 2 * (sq_size + line_w)
        # ICONS and MENUS:
        # icon setting up for a_star:
        self._astar_icon = AstarIcon(1750 + 15 + 6, 1020 - 100 - 25, 22, 53)
        self._astar_icon.connect_to_func(self.choose_a_star)
        # areas setting up:
        bot_menu_x, bot_menu_y = SCREEN_WIDTH - 235, SCREEN_HEIGHT - 70 - 120
        heurs_area = Area(bot_menu_x, bot_menu_y, delta, sq_size, line_w, f'Heuristics', self._heuristic_names)
        heurs_area.choose_field()
        heurs_area.connect_to_func(self._astar.set_heuristic)
        bot_menu_y -= 30 + (len(self._heuristic_names) - 1) * delta + 3 * sq_size
        tiebreakers_area = Area(bot_menu_x, bot_menu_y, delta, sq_size, line_w, f'Tiebreakers', self._tiebreaker_names,
                                True)
        tiebreakers_area.connect_to_func(self._astar.set_tiebreaker)
        bot_menu_y -= 30 + (len(self._tiebreaker_names) - 1) * delta + 3 * sq_size
        is_greedy_area = Area(bot_menu_x, bot_menu_y, delta, sq_size, line_w, f'Is greedy', self._greedy_names, True)
        is_greedy_area.connect_to_func(self._astar.set_greedy_ind)
        # menu composing and connecting to an icon:
        self._astar_menu = Menu()
        self._astar_menu.multiple_append(heurs_area, tiebreakers_area, is_greedy_area)
        self._astar_menu.connect_to_func(self._astar_icon.is_pressed)
        # icon setting up for wave_lee:
        self._wave_lee_icon = Waves(1750 + 50 + 48 + 10 + 6, 1020 - 73 - 25, 32, 5)
        self._wave_lee_icon.connect_to_func(self.choose_wave_lee)
        # icon setting up for bfs/dfs:
        self._bfs_dfs_icon = BfsDfsIcon(1750 - 30 + 6, 1020 - 73 - 25, 54, 2)
        self._bfs_dfs_icon.connect_to_func(self.choose_bfs_dfs)
        # area setting up:
        bot_menu_x, bot_menu_y = SCREEN_WIDTH - 235, SCREEN_HEIGHT - 70 - 120
        area = Area(bot_menu_x, bot_menu_y, delta, sq_size, line_w, f'Core', {0: f'BFS', 1: f'DFS'})
        area.choose_field()
        area.connect_to_func(self._bfs_dfs.set_is_bfs)
        # menu composing and connecting to an icon:
        self._bfs_dfs_menu = Menu()
        self._bfs_dfs_menu.append_area(area)
        self._bfs_dfs_menu.connect_to_func(self._bfs_dfs_icon.is_pressed)
        # GRID:
        # grid icon setting up:
        self._gear_wheel = GearWheelButton(1785 + 6, 1000, 24, 6)
        self._gear_wheel.set_pressed()
        # arrows menu setting up:
        bot_menu_x, bot_menu_y = SCREEN_WIDTH - 235, SCREEN_HEIGHT - 70 - 120
        self._arrows_menu = ArrowsMenu(bot_menu_x, bot_menu_y, 2 * sq_size, sq_size)
        self._arrows_menu.connect_to_func(self._gear_wheel.is_pressed)
        # area setting up :
        bot_menu_y -= 3 * 3 * sq_size
        scaling_area = Area(bot_menu_x, bot_menu_y, delta, sq_size, line_w, f'Sizes in tiles',
                            {k: f'{v}x{self._grid.get_hor_tiles(k)}' for k, v in self._grid.scale_names.items()})
        scaling_area.connect_to_func(self._grid.set_scale)
        scaling_area.choose_field()
        # menu composing connecting to an icon:
        self._settings_menu = Menu()
        self._settings_menu.append_area(scaling_area)
        self._settings_menu.connect_to_func(self._gear_wheel.is_pressed)
        # TWO AUX AREAS:
        bot_menu_y -= 30 + (len(self._grid.scale_names) - 1) * delta + 3 * sq_size
        self._guide_arrows_area = Area(bot_menu_x, bot_menu_y, delta, sq_size, line_w, f'Guide arrows',
                                       self._guide_arrows_names, True)
        self._guide_arrows_area.connect_to_func(self._grid.set_guide_arrows_ind)
        bot_menu_y -= 30 + 3 * sq_size
        self._show_mode_area = Area(bot_menu_x, bot_menu_y, delta, sq_size, line_w, f'Show mode',
                                    self._interactive_names, True)
        self._show_mode_area.connect_to_func(self.set_interactive_ind)
        # menu setting up:
        # MANAGE MENU:
        self._play_button = PlayButton(1785 + 6, 50, 32, 2)
        self._play_button.connect_to_func(self.play_button_func)
        self._step_button_right = StepButton(1785 + 6 + 50, 50, 24, 16, 2)
        self._step_button_right.connect_to_func(self.up, self.another_ornament)
        self._step_button_left = StepButton(1786 + 6 - 50, 50, 24, 16, 2, False)  #
        self._step_button_left.connect_to_func(self.down, self.another_ornament)
        self._eraser = Eraser(1785 + 6, 125, 16, 32, 8, 2)
        self._eraser.connect_to_func(self._grid.clear_grid)
        self._undo_button = Undo(1785 + 6 - 75, 125, 24, 8, 12, 2)
        self._undo_button.connect_to_func(self._grid.undo)
        self._redo_button = Undo(1785 + 6 + 75, 125, 24, 8, 12, 2, True)
        self._redo_button.connect_to_func(self._grid.redo)
        # HINTS:
        self._mode_info = Info(5 + 26 / 4 - self._grid.line_width, SCREEN_HEIGHT - 30, 26)
        self._mode_info.connect_to_func(self.get_mode_info)

    def get_mode_info(self):
        return f'Mode: {self._grid.mode_names[self._grid.mode]}'

    @logged()
    def set_interactive_ind(self, ind: int or None):
        """sets interactive_ind to ind's value"""
        self._interactive_ind = ind

    @logged()
    def choose_a_star(self):
        """chooses a_star as current_algo"""
        self._grid.clear_grid()
        self._current_algo = self._astar

    @logged()
    def choose_wave_lee(self):
        """chooses wave_lee as current_algo"""
        self._grid.clear_grid()
        self._current_algo = self._wave_lee

    @logged()
    def choose_bfs_dfs(self):
        """chooses bfs_dfs as current_algo"""
        self._grid.clear_grid()
        self._current_algo = self._bfs_dfs

    @logged()
    def aux_clearing(self):
        """auxiliary clearing method, clears interactive pars, current_algo pars and unlocks all the lockers"""
        # Lastar clearing:
        Lastar._in_interaction = False
        self._in_interaction_mode_lock = False
        # algo pars clearing:
        if self._current_algo is not None:
            self._current_algo.clear()
        print(f'Connection APPROVED...')
        self.log.info('Connection APPROVED...')
        # lockers off:
        for menu in self._menus_dict.values():
            if menu is not None:
                if not menu.is_hidden():
                    menu.unlock()
        self._show_mode_area.unlock()

    # PRESETS:
    def setup(self):
        self.log.debug('.setup() main class call')
        """main set up method, that is called only ones"""
        # algos' menus/icons:
        for icon in self._icons_dict.values():
            self.log.debug('.setup() Icon class call')
            icon.setup()
            self.log.debug('.setup() Icon class successfully')
        for menu in self._menus_dict.values():
            if menu is not None:
                self.log.debug('.setup() Menu class call')
                menu.setup()
                self.log.debug('.setup() Menu class successfully')
        for manage_icon in self._manage_icons_dict.values():
            self.log.debug(f'.setup() {type(manage_icon)} class call')
            manage_icon.setup()
            self.log.debug(f'.setup() {type(manage_icon)} class successfully')
        self._arrows_menu.setup()
        self._guide_arrows_area.setup()
        self._show_mode_area.setup()
        # grid:
        self._grid.setup()
        # let us make the new mouse cursor:
        cursor = self.get_system_mouse_cursor(self.CURSOR_HAND)
        self.set_mouse_cursor(cursor)
        # let us change the app icon:
        self.set_icon(pyglet.image.load('366.png'))
        # manage icons setting up:
        ...

        # HINTS:
        self._mode_info.setup()
        self.log.debug('.setup() main class successfully')

    # UPDATING:
    def update(self, delta_time: float):
        """main update method being called once per frame, contains long press logic for mouse buttons,
        incrementers changing for living icons and so on"""
        # algos' and settings' icons:
        for icon in self._icons_dict.values():
            icon.update()
        # algos' and settings' menus:
        for menu in self._menus_dict.values():
            if menu is not None:
                menu.update()
        # manage icons:
        for manage_icon in self._manage_icons_dict.values():
            manage_icon.update()
        # HINTS:
        self._mode_info.update()

    def on_draw(self):
        """main drawing method, draws all the elements once per every frame"""  #
        # renders this screen:
        arcade.start_render()
        # GRID:
        self._grid.draw()
        # HINTS:
        ...
        self._mode_info.draw()
        if self._current_algo is not None:
            arcade.Text(self._current_algo.get_current_state(), 225, SCREEN_HEIGHT - 35, arcade.color.BROWN,
                        bold=True).draw()
            if self._grid.node_chosen is not None:
                arcade.Text(self._current_algo.get_details(), 1050, SCREEN_HEIGHT - 35, arcade.color.BROWN,
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
        nc = self._grid.node_chosen
        if nc:
            cx, cy, dh, _ = nc.get_center_n_sizes(self._grid)
            arcade.draw_rectangle_outline(cx, cy, dh, dh,
                                          arcade.color.YELLOW if nc.type != NodeType.EMPTY else arcade.color.DARK_ORANGE,
                                          self._grid.line_width * 2)  # self._grid.tile_size - 2
        # MANAGE ICONS:
        for manage_icon in self._manage_icons_dict.values():
            manage_icon.draw()
        # AUX:
        # sprite = arcade.Sprite('right-arrow-black-triangle.png', 0.125,  center_x=1000, center_y=500)
        # sprite.angle = 90
        # sprite.draw()

        # sprite = SpriteTriangle(98, arcade.color.BLACK)
        # sprite.center_x, sprite.center_y = 1000 + 25, 500 + 25
        # # sprite.angle = 90
        # sprite.draw()

        # sprite = arcade.SpriteCircle(100, arcade.color.PURPLE)
        # sprite.center_x, sprite.center_y = 1000, 500
        # sprite.draw()

    def play_button_func(self, is_pressed: bool = False):
        """function-connector for play_button_icon"""
        self.log.debug('.play_button_func() call')
        if is_pressed:
            self.log.debug('is_pressed -> .clear_empty_nodes() call')
            self._grid.clear_empty_nodes()
        else:
            self.log.debug('.start_algo() call')
            self.start_algo()

    @logged()
    def start_algo(self):
        """starts and prepares the current algorithm for the further using"""
        if not self._grid.loading:
            if self._grid.start_node and self._grid.end_node:
                # STEP BY STEP:
                if self._interactive_ind is not None:  # TODO: add interactive area to wave_lee and bfs_dfs!!!
                    # game logic:
                    Lastar._in_interaction = True
                    # prepare:
                    self._current_algo.prepare()
                    # lockers on:
                    for menu in self._menus_dict.values():
                        if menu is not None:
                            if not menu.is_hidden():
                                menu.lock()
                    self._show_mode_area.lock()
                else:
                    # getting paths:
                    self._time_elapsed = self._current_algo.full_algo()
                    self._current_algo.recover_path()
                    # path's drawing for all three algos cores:
                    self._current_algo.visualize_path()

    @logged()
    def up(self):
        """function-connector for right step_button_icon,
        steps up during the algo or path-restoring phase"""
        if self._current_algo.path is None:
            self._current_algo.algo_up()
        else:
            self._current_algo.path_up()

    @logged()
    def down(self):
        """function-connector for left step_button_icon,
        steps down during the algo or path-restoring phase"""
        if self._current_algo.path is None:
            self._current_algo.algo_down()
        else:
            self._current_algo.path_down()

    @logged()
    def another_ornament(self, is_next: bool):
        """switches the current walls-ornament during the loading"""
        return self._grid.change_wall_ornament(is_next)

    # KEYBOARD:
    def on_key_press(self, symbol: int, modifiers: int):
        """main key-pressing method"""
        # is called when user press the symbol key:
        match symbol:
            # a_star_call:
            case arcade.key.SPACE:
                self.log.info('key.SPACE -> _play_button.press() call')
                self._play_button.press()
            # entirely grid clearing:
            case arcade.key.ENTER:
                self.log.info('key.ENTER -> .clear_grid() call')
                self._grid.clear_grid()
            # a_star interactive:
            case arcade.key.RIGHT:
                self.log.info('key.RIGHT -> .on_key_press()[1 -> _step_button_right] call')
                self._step_button_right.on_key_press()
            case arcade.key.LEFT:
                self.log.info('key.LEFT -> .on_key_press()[2 -> _step_button_left] call')
                self._step_button_left.on_key_press()  #
            # undoing and cancelling:
            case arcade.key.Z:  # undo
                if not Lastar._in_interaction:
                    self.log.info('key.Z -> .undo() for _walls_built_erased call')
                    self._grid.undo()
            case arcade.key.Y:  # cancels undo
                if not Lastar._in_interaction:
                    self.log.info('key.Y -> redo() for _walls_built_erased call')
                    self._grid.redo()
            # saving and loading:
            case arcade.key.S:
                self.log.info('key.S -> save() for _walls call')
                self._grid.save()
            case arcade.key.L:
                # cannot loading while in interaction:
                if not self.in_interaction:
                    self.log.info('key.L -> load() for _walls call')
                    self._grid.load()

    def on_key_release(self, symbol: int, modifiers: int):
        """main key-releasing method"""
        # long press ending:
        match symbol:
            case arcade.key.RIGHT:
                self.log.info('key.RIGHT release-> .on_key_press()[1 -> _step_button_right] call')
                self._step_button_right.on_key_release()
            case arcade.key.LEFT:
                self.log.info('key.LEFT release-> .on_key_press()[12 -> _step_button_left] call')
                self._step_button_left.on_key_release()

    def on_mouse_motion(self, x: int, y: int, dx: int, dy: int):
        """main method processing the mouse motion"""
        for icon in self._icons_dict.values():
            icon.on_motion(x, y)
        for menu in self._menus_dict.values():
            if menu is not None:
                menu.on_motion(x, y)
        for manage_icon in self._manage_icons_dict.values():
            manage_icon.on_motion(x, y)
        self._arrows_menu.on_motion(x, y)
        self._guide_arrows_area.on_motion(x, y)
        self._show_mode_area.on_motion(x, y)
        # building and erasing walls:
        self._grid.build_or_erase(x, y)

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int):
        """main mouse-pressing method"""
        for icon in self._icons_dict.values():
            self.log.debug(f'{type(icon)} .on_press() call')
            if icon.on_press(x, y) is not None:
                self.clear_icons_inter_types(icon)
        for menu in self._menus_dict.values():
            if menu is not None:
                self.log.debug(f'{type(menu)} .on_press() call')
                menu.on_press(x, y)
        for manage_icon in self._manage_icons_dict.values():
            self.log.debug(f'{type(manage_icon)} .on_press() call')
            manage_icon.on_press(x, y)
        self._arrows_menu.on_press(x, y)
        self._guide_arrows_area.on_press(x, y)
        self._show_mode_area.on_press(x, y)
        # grid:
        self._grid.press(x, y, button)
        self.log.debug('.on_mouse_press() successfully -> all .on_press() is processing')

    def on_mouse_release(self, x: int, y: int, button: int, modifiers: int):
        """main mouse-releasing method"""
        self._grid.building_walls_flag = False
        for manage_icon in self._manage_icons_dict.values():
            self.log.debug(f'{type(manage_icon)} .on_release() call')
            manage_icon.on_release(x, y)

    # game mode switching by scrolling the mouse wheel:
    def on_mouse_scroll(self, x: int, y: int, scroll_x: int, scroll_y: int):
        """main mouse-scrolling method"""
        self.log.debug('_grid.scroll.scroll() call')
        self._grid.scroll()

    @logged()
    def clear_icons_inter_types(self, icon_chosen: 'Icon'):
        """clear the inter_types of settings and algo icons"""
        for icon in self._icons_dict.values():
            if icon != icon_chosen:
                icon.clear_inter_type()


class DrawLib:
    @staticmethod
    def create_line_arrow(node: 'Node', deltas: tuple[int, int] = (-1, 0),
                          grid: 'Grid' = None):
        """creates a guiding triangle-arrow in order to visualize the path and visited nodes more comprehensively,
        by default the arrow to be drawn is left sided"""
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
        """checks if the point given located in the square with the center in (cx, cy) and side that equals size"""
        return cx - size / 2 <= x <= cx + size / 2 and cy - size / 2 <= y <= cy + size / 2

    @staticmethod
    def is_point_in_circle(cx, cy, r, x, y):
        """checks if the point given located in the circle with the center in (cx, cy) and radius that equals r"""
        return (cx - x) ** 2 + (cy - y) ** 2 <= r ** 2

    @staticmethod
    def draw_icon_lock():
        """draws a lock for an icon"""
        ...


class Drawable(ABC):
    """interface for drawable element"""

    # logging.config.fileConfig('log.conf', disable_existing_loggers=True)

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
    """interface for interactable element (that can be pressed, hovered and so on)"""

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
    def on_key_press(self):
        ...

    @abstractmethod
    def on_key_release(self):
        ...


class Connected(ABC):  # Connected
    """interface for connection between two elements (class exemplars)"""

    def __init__(self):
        # object to be managed:
        self.log = logging.getLogger('Connected')
        self._obj = None

    # connects elements (one-way link):
    @abstractmethod
    def connect(self, obj):
        self.log.info('Connected')  # WITH WHAT???
        self._obj = obj


class FuncConnected(ABC):
    """interface for connection between an element (class exemplar)
    and some method(s) of another one"""

    def __init__(self):
        # function/functions connected:
        self._func = None

    # connects elements and methods:
    def connect_to_func(self, *func):
        log = logging.getLogger('FuncConnected')
        try:
            log.info(f'Connected to func {func.__name__}')
        except AttributeError:
            log.info(f'Connected to func {func}')
        self._func = func


class Grid(Drawable, FuncConnected):
    """core class for grid lines, nodes and guiding arrows display, start/end nodes and walls building and erasing,
    node choosing for info getting and saving/loading of wall-ornaments"""

    def __init__(self):
        super().__init__()
        self.log = logging.getLogger('Grid')
        # the grid itself:
        self._grid = None
        # important nodes:
        self._start_node = None
        self._end_node = None
        # game mode:
        self._mode = 0  # 0 for building the walls and erasing them afterwards, 1 for a start and end nodes choosing and 2 for info getting for every node
        self._mode_names = {0: 'BUILDING', 1: 'START&END', 2: 'DETAILS'}
        # the current node chosen (for getting info):
        self._node_chosen = None
        # guide arrows ON/OFF:
        self._guide_arrows_ind = None
        # visualization:
        self._node_sprite_list = arcade.SpriteList()
        self._grid_line_shapes = arcade.ShapeElementList()
        # algo steps visualization:
        self._triangle_shape_list = arcade.ShapeElementList()  # <<-- for more comprehensive path visualization
        self._arrow_sprite_list = arcade.SpriteList()  # <<-- for more comprehensive algorithm's visualization
        # guiding arrows presets:
        self._preset_arrows = []
        # sizes:
        self._X, self._Y = SCREEN_HEIGHT - 60, SCREEN_WIDTH - 250
        # scaling:  TODO: add AI to calculate the sizes for every resolution possible:
        self._scale = 0
        self._scale_names = {0: 10, 1: 15, 2: 22, 3: 33, 4: 45, 5: 66, 6: 90,
                             7: 110}  # {0: 5, 1: 10, 2: 15, 3: 22, 4: 33, 5: 45, 6: 66, 7: 90, 8: 110, 9: 165, 10: 198}  # factors of 990 num
        # pars:
        self._tiles_q = None
        self._line_width = None
        self._tile_size, self._hor_tiles_q = self.get_pars()
        # WALLS MANAGER:
        self._building_walls_flag = False
        self._build_or_erase = True  # True for building and False for erasing
        # memoization for undo/redo area:
        self._walls_built_erased = [([], True)]  # TODO: swap to DICT!!!
        self._walls_index = 0
        self._walls = set()  # all walls located on the map at the time being
        # save/load:
        self._loading = False
        self._loading_ind = 0

    # INITIALIZATION AUX:
    @logged()
    def get_pars(self):
        """calculating grid visualization pars for vertical tiles number given"""
        self._Y, self._X = SCREEN_HEIGHT - 60, SCREEN_WIDTH - 250
        self._tiles_q = self._scale_names[self._scale]
        self._line_width = int(math.sqrt(max(self._scale_names.values()) / self._tiles_q))
        tile_size = self._Y // self.tiles_q
        hor_tiles_q = self._X // tile_size
        self._Y, self._X = self.tiles_q * tile_size, hor_tiles_q * tile_size
        return tile_size, hor_tiles_q

    @logged()
    def get_hor_tiles(self, i):
        return (SCREEN_WIDTH - 250) // (
                (SCREEN_HEIGHT - 30) // self._scale_names[i])  # TODO: ELIMINATE THE DEVIATION IN Y coordinates!!!

    @logged()
    def set_scale(self, ind: int):
        self._scale = ind
        self.rebuild_map()

    @logged()
    def set_guide_arrows_ind(self, ind: int):
        self._guide_arrows_ind = ind

    # AUX:
    @logged()
    def get_triangle(self, node: 'Node', point: tuple[int, int]):
        """triangle points getting"""
        scaled_point = point[0] * (self._tile_size // 2 - 2), point[1] * (self._tile_size // 2 - 2)
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
        cx, cy = 5 + node.x * self._tile_size + self._tile_size / 2, \
                 5 + node.y * self._tile_size + self._tile_size / 2
        return (cx, cy), (cx + deltas[0][0], cy + deltas[0][1]), (cx + deltas[1][0], cy + deltas[1][1])

    @logged()
    def number_repr(self, node: 'Node'):
        """gets the number representation for the node"""
        return node.y * self._hor_tiles_q + node.x

    @logged()
    def coords(self, number: int):
        """gets node's coordinates for its number representation"""
        return divmod(number, self._hor_tiles_q)

    @logged()
    def node(self, num: int) -> 'Node':
        """gets the node itself for its number representation"""
        y, x = self.coords(num)
        return self._grid[y][x]

    @logged()
    def get_node(self, mouse_x, mouse_y):
        """gets the node from the current mouse coordinates"""
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
    def arrow_sprite_list(self):
        return self._arrow_sprite_list

    @arrow_sprite_list.setter
    def arrow_sprite_list(self, arrow_sprite_list):
        self._arrow_sprite_list = arrow_sprite_list

    @property
    def triangle_shape_list(self):
        return self._triangle_shape_list

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
    def building_walls_flag(self):
        return self._building_walls_flag

    @building_walls_flag.setter
    def building_walls_flag(self, building_walls_flag):
        self._building_walls_flag = building_walls_flag

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

    @property
    def loading(self):
        return self._loading

    @logged()
    def initialize(self):
        """initializes all the nodes for _tiles_q par"""
        self._grid = [[Node(j, i, 1, NodeType.EMPTY) for i in range(self._hor_tiles_q)] for j in range(self._tiles_q)]

    @logged()
    def can_start(self):
        """checks if start and end nodes are chosen"""
        return self._start_node is not None and self._end_node is not None

    @logged()
    def clear_redo_memo(self):
        """clears redo memo list to the right from the current point if some action has been done
         while still there is some possibility of redoing"""
        if self._walls_index < len(self._walls_built_erased) - 1:
            self._walls_built_erased = self._walls_built_erased[:self._walls_index + 1]

    @logged()
    # make a node the chosen one:
    def choose_node(self, node: 'Node'):
        """chooses a node for info display"""
        self._node_chosen = node
        # draw a frame

    @logged()
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

    @logged()
    def scroll(self):
        self._mode = (self._mode + 1) % len(self._mode_names)

    def draw(self):
        # grid:
        self._grid_line_shapes.draw()
        # blocks:
        self._node_sprite_list.draw()
        # arrows:
        if self._guide_arrows_ind is not None:
            if len(self.arrow_sprite_list) > 0:
                self.arrow_sprite_list.draw()
        # path arrows:
        if self._triangle_shape_list:
            self._triangle_shape_list.draw()

    @logged()
    # creates sprites for all the nodes:
    def get_sprites(self):  # batch -->
        """nodes' and guiding arrows' sprites initialization for fast further batch-drawing"""
        for row in self.grid:
            for node in row:
                # creates a node's sprite:
                node.get_solid_colour_sprite(self)
                # creates a guiding arrow:
                node.get_guiding_arrow(self)
        self.log.info(f"{self._tiles_q * self._hor_tiles_q} grid nodes' sprites initialization")

    @logged()
    def make_grid_lines(self):
        """creates a shape element list of grid lines for fast further batch-drawing"""
        for j in range(self.tiles_q + 1):
            self._grid_line_shapes.append(
                arcade.create_line(5, 5 + self._tile_size * j, 5 + self._X, 5 + self._tile_size * j,
                                   arcade.color.BLACK,
                                   self._line_width))

        for i in range(self._hor_tiles_q + 1):
            self._grid_line_shapes.append(
                arcade.create_line(5 + self._tile_size * i, 5, 5 + self._tile_size * i, 5 + self._Y,
                                   arcade.color.BLACK,
                                   self._line_width))
        self.log.info(
            f"{self._tiles_q + self._hor_tiles_q + 2} grid lines' shapes initialization")

    # WALLS MANAGER:
    # EMPTIES -->> WALLS and BACK:
    @logged()
    def change_nodes_type(self, node_type: 'NodeType', walls_set: set or list):
        for node_num in walls_set:
            y, x = self.coords(node_num)
            self.grid[y][x].type = node_type
            self.grid[y][x].update_sprite_colour()
        self.log.info(f"{len(walls_set)} grid nodes changing their type")

    @logged(is_debug=False)
    def press(self, x, y, button):
        # MODES OF DRAWING LOGIC:
        if self._mode == 0:
            self.log.info(f'selected mode {self._mode_names[0]}')
            self._building_walls_flag = True
            if button == arcade.MOUSE_BUTTON_LEFT:
                self.log.info('MOUSE_BUTTON_LEFT -> build the walls')
                self._build_or_erase = True
            elif button == arcade.MOUSE_BUTTON_RIGHT:
                self.log.info('MOUSE_BUTTON_RIGHT -> erase the walls')
                self._build_or_erase = False
            elif button == arcade.MOUSE_BUTTON_MIDDLE:
                self.log.info('MOUSE_BUTTON_MIDDLE -> erase all linked nodes')
                self._build_or_erase = None
                n = self.get_node(x, y)
                if n:
                    self.clear_redo_memo()
                    # if self._walls_index < len(self._walls_built_erased) - 1:
                    #     self._walls_built_erased = self._walls_built_erased[: self._walls_index + 1]
                    self._walls_built_erased.append(([], False))
                    self._walls_index += 1
                    self.erase_all_linked_nodes(n)
        elif self._mode == 1:
            self.log.info(f'selected mode {self._mode_names[1]}')
            if button == arcade.MOUSE_BUTTON_LEFT:
                self.log.info('MOUSE_BUTTON_LEFT -> set START_NODE')
                sn = self.get_node(x, y)
                if sn:
                    if self._start_node:
                        self._start_node.type = NodeType.EMPTY
                        self._start_node.update_sprite_colour()
                    sn.type = NodeType.START_NODE
                    self._start_node = sn
                    self._start_node.update_sprite_colour()
            elif button == arcade.MOUSE_BUTTON_RIGHT:
                self.log.info('MOUSE_BUTTON_RIGHT -> set END_NODE')
                en = self.get_node(x, y)
                if en:
                    if self._end_node:
                        self._end_node.type = NodeType.EMPTY
                        self._end_node.update_sprite_colour()
                    en.type = NodeType.END_NODE
                    self._end_node = en
                    self._end_node.update_sprite_colour()
        elif self._mode == 2:  # a_star interactive -->> info getting:
            self.log.info(f'selected mode {self._mode_names[2]}')
            n = self.get_node(x, y)
            if n:
                if self._node_chosen == n:
                    self._node_chosen = None
                else:
                    self._node_chosen = n

    # builds/erases walls:
    @logged()
    def build_or_erase(self, x, y):
        """builds or erases a wall depending on the self._build_or_erase flag"""
        if self._building_walls_flag and self._mode == 0:
            if self._build_or_erase is not None:
                if self._build_or_erase:
                    self.log.debug(f'.build_wall({x}, {y})')
                    self.build_wall(x, y)
                else:
                    self.log.debug(f'.erase_wall({x},{y})')
                    self.erase_wall(x, y)

    def build_wall(self, x, y):
        # now building the walls:
        n = self.get_node(x, y)
        if n and n.type != NodeType.WALL:
            self.log.info(f'Building a wall')
            n.type = NodeType.WALL
            self._walls.add(self.number_repr(n))
            n.update_sprite_colour()
            self.clear_redo_memo()
            # if self._walls_index < len(self._walls_built_erased) - 1:
            #     self._walls_built_erased = self._walls_built_erased[:self._walls_index + 1]
            self._walls_built_erased.append(([self.number_repr(n)], True))
            self._walls_index += 1
            self.log.info(f'The wall successfully built')

    def erase_wall(self, x, y):
        # now erasing the walls:
        n = self.get_node(x, y)
        if n and n.type == NodeType.WALL:
            self.log.info(f'Erasing a wall')
            n.type = NodeType.EMPTY
            self._walls.remove(self.number_repr(n))
            n.update_sprite_colour()
            self.clear_redo_memo()
            # if self._walls_index < len(self._walls_built_erased) - 1:
            #     self._walls_built_erased = self._walls_built_erased[:self._walls_index + 1]
            self._walls_built_erased.append(([self.number_repr(n)], False))
            self._walls_index += 1
            self.log.info(f'The wall successfully erased')

    # erases all nodes, that are connected vertically, horizontally or diagonally to a chosen one,
    # then nodes connected to them the same way and so on recursively...
    @logged()
    def erase_all_linked_nodes(self, node: 'Node'):  # TODO: PROCESS AND LOG THIS RECURSIVE METHOD VERY CAREFULLY!!!
        @counted
        def _erase_all_linked_nodes(curr_node: 'Node'):
            curr_node.type = NodeType.EMPTY
            curr_node.update_sprite_colour()
            _number_repr = self.number_repr(curr_node)
            self._walls.remove(_number_repr)
            self._walls_built_erased[self._walls_index][0].append(_number_repr)
            for neigh in curr_node.get_extended_neighs(
                    self):  # TODO: FIT .get_extended_neighs() method in Node class!!!
                if neigh.type == NodeType.WALL:
                    _erase_all_linked_nodes(neigh)

        _erase_all_linked_nodes(node)
        self.log.info(
            f'inner recursive method .{_erase_all_linked_nodes}() called {_erase_all_linked_nodes.rec_calls} times and has max depth of {_erase_all_linked_nodes.rec_depth}')

    # CLEARING/REBUILDING:
    @lock
    def rebuild_map(self):
        self._tile_size, self._hor_tiles_q = self.get_pars()
        # grid's renewing:
        self.initialize()
        # pars resetting:
        self.aux_clear()
        self._start_node = None
        self._end_node = None
        self._node_chosen = None
        self._node_sprite_list = arcade.SpriteList()
        self._arrow_sprite_list = arcade.SpriteList()
        self._grid_line_shapes = arcade.ShapeElementList()
        self._walls = set()
        self.setup()

    @logged()
    def clear_empty_nodes(self):
        """clears all the nodes except start, end and walls"""
        # clearing the every empty node:
        for row in self._grid:
            for node in row:
                if node.type not in [NodeType.WALL, NodeType.START_NODE, NodeType.END_NODE]:
                    node.clear()
                elif node.type in [NodeType.START_NODE, NodeType.END_NODE]:
                    node.heur_clear()
        # clearing the nodes-relating pars of the game:
        self.aux_clear()

    @logged()
    @lock
    def clear_grid(self):
        """entirely clears the grid"""
        # clearing the every node:
        for row in self._grid:
            for node in row:
                node.clear()
        # clearing the nodes-relating pars of the game:
        self._start_node, self._end_node = None, None
        self.aux_clear()
        self._walls = set()
        # memoization for possible undoing:
        ...

    @logged()
    def aux_clear(self):
        # grid's pars clearing:
        self._triangle_shape_list = arcade.ShapeElementList()  # <<-- for more comprehensive path visualization
        self._arrow_sprite_list = arcade.SpriteList()  # <<-- for more comprehensive algorithm's visualization
        # builder clearing:
        self._walls_built_erased = [([], True)]
        self._walls_index = 0
        # node chosen clearing:
        self._node_chosen = None
        # aux clearing from Lastar:
        self._func[0]()

    @logged()
    def undo(self):
        """undo manager: cancels the action with the wall"""
        if not self._loading:
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

    @logged()
    def redo(self):
        """redo manager: redo an action with a wall"""
        if not self._loading:
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
    @logged()
    def save(self):
        print(f'walls: {self.walls}')
        with shelve.open(f'saved_walls {self._tiles_q}', 'c') as shelf:
            index = len(shelf)
            shelf[f'ornament {index}'] = self._walls

    @logged()
    def load(self):
        if self._loading:
            self._loading = False
            self._walls = self._walls[f'ornament {self._loading_ind}']
            self._walls_built_erased.append(([], False))
            for num in self._walls:
                self._walls_built_erased[self._walls_index][0].append(num)
            self._walls_index += 1
        elif len(self._walls) == 0:
            with shelve.open(f'saved_walls {self._tiles_q}',
                             'r') as shelf:  # TODO: flag to 'r' and check if the file exist!
                self.log.info(f'saving to a file {self._tiles_q}')
                index = len(shelf)
                if index > 0:
                    self._loading = True
                    self._walls_built_erased.append(([], True))
                    for num in self._walls:
                        self._walls_built_erased[self._walls_index][0].append(num)
                    self._walls_index += 1
                    self._walls = dict(shelf)

    @logged()
    def change_wall_ornament(self, is_next=True):
        if self._loading:
            delta = 1 if is_next else -1
            self.change_nodes_type(NodeType.EMPTY, self._grid.walls[f'ornament {self._loading_ind}'])
            self._loading_ind = (self._loading_ind + delta) % len(self._grid.walls)
            self.change_nodes_type(NodeType.WALL, self._grid.walls[f'ornament {self._loading_ind}'])

    @property
    def mode_names(self):
        return self._mode_names

    @property
    def mode(self):
        return self._mode

    @property
    def preset_arrows(self):
        return self._preset_arrows


# class for a node representation:
class Node:
    is_greedy = False
    # horizontal and vertical up and down moves:
    walk = [(dy, dx) for dx in range(-1, 2) for dy in range(-1, 2) if dy * dx == 0 and (dy, dx) != (0, 0)]
    extended_walk = [(dy, dx) for dx in range(-1, 2) for dy in range(-1, 2) if (dy, dx) != (0, 0)]
    # for a_star (accurate removing from heap):
    aux_equal_flag = False
    # guiding arrow dirs dict:
    dirs_to_angles = {(1, 0): 0, (0, 1): 90, (-1, 0): 180, (0, -1): 270}

    def __init__(self, y, x, val, node_type: 'NodeType'):
        # logger
        self.log = logging.getLogger('Node')
        # type and sprites:
        self.type = node_type
        self.sprite = None
        self.guiding_arrow_sprite = None  # for more comprehensive visualization
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

    @staticmethod
    def set_greedy(greedy_ind: int):
        Node.is_greedy = False if greedy_ind is None else True

    # COPYING/RESTORING:
    @logged(is_used=False)
    def aux_copy(self):
        """makes an auxiliary copy for a node, it is needed for a_star interactive"""
        copied_node = Node(self.y, self.x, self.type, self.val)
        copied_node.g = self.g
        copied_node.h = self.h
        copied_node.tiebreaker = self.tiebreaker
        copied_node.type = self.type
        copied_node.previously_visited_node = self.previously_visited_node
        return copied_node

    @logged(is_used=False)
    def restore(self, copied_node: 'Node'):
        """restore the node from its auxiliary copy"""
        self.g = copied_node.g
        self.h = copied_node.h
        self.tiebreaker = copied_node.tiebreaker
        self.type = copied_node.type  # NodeType.EMPTY ???
        self.update_sprite_colour()
        self.previously_visited_node = copied_node.previously_visited_node

    # SMART COPYING/RESTORING:
    @logged(is_used=False)
    def smart_copy(self, attributes: list[str]):
        copied_node = Node(self.y, self.x, self.type, self.val)
        self.smart_core(copied_node, attributes)
        return copied_node

    @logged(is_used=False)
    def smart_restore(self, other: 'Node', attributes: list[str]):
        other.smart_core(self, attributes)
        if 'type' in attributes:
            self.update_sprite_colour()

    @logged(is_used=False)
    def smart_core(self, other: 'Node', attributes: list[str]):
        for attribute in attributes:
            other.__dict__[attribute] = self.__getattribute__(attribute)

    # TYPE/SPRITE CHANGE/INIT:
    # @logged(is_used=graphic_logging)  # TODO: DANGEROUS TO LOG!!!
    def get_solid_colour_sprite(self, grid: Grid):
        """makes a solid colour sprite for a node"""
        cx, cy, size, colour = self.get_center_n_sizes(grid)
        self.sprite = arcade.SpriteSolidColor(size, size, colour)
        self.sprite.center_x, self.sprite.center_y = cx, cy
        grid.node_sprite_list.append(self.sprite)

    def get_center_n_sizes(self, grid: Grid):
        """aux calculations"""
        return (5 + grid.tile_size * self.x + grid.tile_size / 2,
                5 + grid.tile_size * self.y + grid.tile_size / 2,
                grid.tile_size - 2 * grid.line_width - (1 if grid.line_width % 2 != 0 else 0),
                self.type.value)

    # TODO: DANGEROUS TO LOG!!!
    def update_sprite_colour(self):
        """updates the sprite's color (calls after node's type switching)"""
        self.sprite.color = self.type.value

    # GUIDING ARROWS CHANGE/INIT:
    def get_guiding_arrow(self, grid: Grid):
        """makes a guiding arrow sprite for a node"""
        cx, cy = 5 + grid.tile_size * self.x + grid.tile_size / 2, 5 + grid.tile_size * self.y + grid.tile_size / 2
        size = 2 * grid.tile_size // 3
        color = arcade.color.BLACK
        self.guiding_arrow_sprite = SpriteTriangle(size, color)
        self.guiding_arrow_sprite.center_x, self.guiding_arrow_sprite.center_y = cx, cy
        # self.guiding_arrow_sprite.angle = 0
        # print(f'arrow type: {type(self.guiding_arrow_sprite)}')

    def rotate_arrow(self, delta: tuple[int, int]):
        """
        rotates the guiding arrow to the direction given

        :param tuple[int, int] delta: tuple of (dx, dy) that defines
        the direction the guiding arrow is pointed to

        """

        # here the arrow rotates:
        self.guiding_arrow_sprite.angle = self.dirs_to_angles[delta]

    def append_arrow(self, grid: Grid):
        grid.arrow_sprite_list.append(self.guiding_arrow_sprite)

    # TODO: DANGEROUS TO LOG!!!
    def remove_arrow(self, grid: Grid):
        """removes the guiding arrow sprite of the node from the arrow_sprite_list"""
        grid.arrow_sprite_list.remove(self.guiding_arrow_sprite)
        self.guiding_arrow_sprite = None

    # TODO: DANGEROUS TO LOG!!!
    def remove_arrow_from_shape_list(self, grid: Grid):
        grid.arrow_sprite_list.remove(self.guiding_arrow_sprite)

    # str and repr DUNDERS:

    def __str__(self):
        return f'{self.y, self.x} -->> {self.val}'

    def __repr__(self):
        return str(self)

    # other important DUNDERS:
    def __eq__(self, other):
        if type(self) != type(other):
            return False
        if Node.aux_equal_flag:
            return (self.y, self.x, self.g, self.h) == (other.y, other.x, other.g, other.h)  # .tiebreaker???
        else:
            return (self.y, self.x) == (other.y, other.x)

    # this is needed for using Node objects in priority queue like heapq and so on
    def __lt__(self, other: 'Node'):
        if self.is_greedy:
            return (self.h, self.tiebreaker) < (other.h, other.tiebreaker)
        else:
            return (self.g + self.h, self.tiebreaker) < (other.g + other.h, other.tiebreaker)

    def __hash__(self):
        return hash((self.y, self.x))

    # CLEARING:
    # TODO: DANGEROUS TO LOG!!!
    # @logged()
    def clear(self):
        """entirely clears the node, returning it to the initial state it came from"""
        self.heur_clear()
        self.type = NodeType.EMPTY
        self.update_sprite_colour()
        # self.guiding_arrow_sprite = None

    # TODO: DANGEROUS TO LOG!!!
    # @logged()
    def heur_clear(self):
        """clears the node heuristically"""
        self.g = np.Infinity
        self.h = 0
        self.tiebreaker = None
        self.previously_visited_node = None
        self.times_visited = 0
        # wave lee:
        self.val = 1

    # HEURISTICS: # TODO: DANGEROUS TO LOG!!!
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
    # TODO: DANGEROUS TO LOG!!!
    def get_neighs(self, grid: Grid, forbidden_node_types: list['NodeType']):  # has become smarter
        """gets neighs of the node, now can be set up"""
        for dy, dx in self.walk:
            ny, nx = self.y + dy, self.x + dx
            if 0 <= ny < grid.tiles_q and 0 <= nx < grid.hor_tiles_q:
                # by default can visit the already visited nodes
                if grid.grid[ny][nx].type not in forbidden_node_types:
                    yield grid.grid[ny][nx]

    # TODO: DANGEROUS TO LOG!!!
    def get_extended_neighs(self, grid: Grid) -> list['Node']:
        """gets extended neighs (with diagonal ones) of the node, generator"""
        for dy, dx in self.extended_walk:
            ny, nx = self.y + dy, self.x + dx
            if 0 <= ny < grid.tiles_q and 0 <= nx < grid.hor_tiles_q:
                yield grid.grid[ny][nx]


# class representing an algo:
class Algorithm(Connected):

    def __init__(self, name: str):
        super().__init__()
        # logger
        self.log = logging.getLogger('Algorithm')
        # info:
        self._name = name
        # path:
        self._path = None
        self._path_index = 0
        # iterations and time:
        self._iterations = 0
        self._time_elapsed = 0

    def connect(self, grid: Grid):
        self._obj = grid

    @property
    def path(self):
        return self._path

    @property
    def name(self):
        return self._name

    @property
    def iterations(self):
        return self._iterations

    @property
    def time_elapsed(self):
        return self._time_elapsed

    @time_elapsed.setter
    def time_elapsed(self, time_elapsed):
        self._time_elapsed = time_elapsed

    @abstractmethod
    def get_nodes_visited_q(self):
        ...

    def base_clear(self):
        # visualization:
        self._obj.triangle_shape_list = arcade.ShapeElementList()  # <<-- for more comprehensive path visualization
        self._obj.arrow_sprite_list = arcade.ShapeElementList()  # <<-- for more comprehensive algorithm's visualization
        # path:
        self._path = None
        self._path_index = 0
        # iterations and time:
        self._iterations = 0
        self._time_elapsed = 0

    @abstractmethod
    def clear(self):
        ...

    @abstractmethod
    def prepare(self):
        ...

    def get_current_state(self):
        """returns the important pars of the current algo state as f-string"""
        return f"{self._name}'s iters: {self._iterations}, path's length:" \
               f" {len(self._path) if self._path else 'no path found still'}, " \
               f"nodes visited: {self.get_nodes_visited_q()}, time elapsed: {self._time_elapsed} ms"

    @abstractmethod
    def get_details(self):
        """returns the important details (like heur, val and so on) for the node chosen"""
        ...

    @logged()
    def path_up(self):
        if self._path_index < len(self.path) - 1:
            if (path_node := self._path[self._path_index]).type not in [NodeType.START_NODE, NodeType.END_NODE]:
                path_node.type = NodeType.PATH_NODE
                path_node.update_sprite_colour()
            # arrows:
            p = -self._path[self._path_index + 1].x + self._path[self._path_index].x, \
                -self._path[self._path_index + 1].y + self._path[self._path_index].y
            p1, p2, p3 = self._obj.get_triangle(self._path[self._path_index + 1], p)
            triangle_shape = arcade.create_triangles_filled_with_colors(
                [p1, p2, p3],
                [arcade.color.WHITE, arcade.color.RED, arcade.color.RED])
            self._obj.triangle_shape_list.append(triangle_shape)  # NOT A BUG!!!
            # line arrows removing:
            node = self._path[self._path_index + 1]
            # if self.inter_types[2] == InterType.PRESSED:
            if node not in [self._obj.start_node, self._obj.end_node]:
                node.remove_arrow_from_shape_list(self._obj)
            # index's step up:
            self._path_index += 1

    @logged()
    def recover_path(self):
        """path-recovering process for interactive a_star"""
        # start point of path restoration (here we begin from the end node of the shortest path found):
        node = self._obj.end_node
        shortest_path = []
        # path restoring (here we get the reversed path):
        while node.previously_visited_node:
            shortest_path.append(node)
            node = node.previously_visited_node
        shortest_path.append(self._obj.start_node)
        # returns the result:
        self._path = shortest_path

    @logged()
    def visualize_path(self):
        for i, node in enumerate(self._path):
            if node.type not in [NodeType.START_NODE, NodeType.END_NODE]:
                node.type = NodeType.PATH_NODE
                node.update_sprite_colour()
            if node.arrow_shape is not None:
                node.remove_arrow(self._obj)
            if i + 1 < len(self.path):
                p = -self.path[i + 1].x + self.path[i].x, \
                    -self.path[i + 1].y + self.path[i].y
                p1, p2, p3 = self._obj.get_triangle(self.path[i + 1], p)
                triangle_shape = arcade.create_triangles_filled_with_colors(
                    [p1, p2, p3],
                    [arcade.color.WHITE, arcade.color.RED, arcade.color.RED])
                self._obj.triangle_shape_list.append(triangle_shape)

    @logged()
    def path_down(self):
        if self._path_index > 0:
            if (path_node := self._path[self._path_index]).type not in [NodeType.START_NODE, NodeType.END_NODE]:
                path_node.type = NodeType.VISITED_NODE
                path_node.update_sprite_colour()
                # arrows:
            self._obj.triangle_shape_list.remove(self._obj.triangle_shape_list[self._path_index - 1])  # NOT A BUG!!!
            # line arrows restoring:
            if path_node not in [self._obj.start_node, self._obj.end_node]:
                path_node.append_arrow(self._obj)
            self._path_index -= 1
        else:
            self._path = None
            self.algo_down()

    @abstractmethod
    def algo_up(self):
        ...

    @abstractmethod
    def algo_down(self):
        ...

    @abstractmethod
    def full_algo(self):
        ...


class Astar(Algorithm, FuncConnected):

    def __init__(self):
        super().__init__('Astar')
        # logger
        self.log = logging.getLogger('Astar')
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

    def get_nodes_visited_q(self):
        return len(self._nodes_visited)

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

    def get_details(self):
        node_chosen = self._obj.node_chosen
        return f"Node: {node_chosen.y, node_chosen.x}, val: {node_chosen.val}, g: {node_chosen.g}, h: {node_chosen.h}, " \
               f"f=g+h: {node_chosen.g + node_chosen.h}, t: {node_chosen.tiebreaker}, " \
               f"times visited: {node_chosen.times_visited}"  # , type: {node_chosen.type}"

    @logged()
    def prepare(self):
        # heap:
        self._nodes_to_be_visited = [self._obj.start_node]
        hq.heapify(self._nodes_to_be_visited)
        # heur/cost:
        self._obj.start_node.g = 0
        # transmitting the greedy flag to the Node class: TODO: fix this strange doing <<--
        self._func[0](self._greedy_ind)
        # important pars and dicts:
        self._iterations = 0
        self._time_elapsed = 0
        self._neighs_added_to_heap_dict = {0: [self._obj.start_node]}
        self._curr_node_dict = {0: None}
        self._max_times_visited_dict = {0: 0}
        # path nullifying:
        self._path = None
        # SETTINGS menu should be closed during the algo's interactive phase!!!
        # arrows list renewal:
        self._obj.arrow_sprite_list = arcade.SpriteList()

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
            self.recover_path()
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
                    # operations with arrows:
                    print(f'neigh: {neigh}, arrow type: {neigh.guiding_arrow_sprite}')
                    neigh.rotate_arrow((neigh.x - curr_node.x, neigh.y - curr_node.y))
                    # arrow = DrawLib.create_line_arrow(neigh, (neigh.x - curr_node.x, neigh.y - curr_node.y), self._obj)
                    # here the arrow rotates (re-estimating of neigh g-cost):
                    if neigh.guiding_arrow_sprite not in self._obj.arrow_sprite_list:
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
                # operations with arrows:
                if node.type not in [NodeType.START_NODE, NodeType.END_NODE, NodeType.NEIGH]:
                    if node.guiding_arrow_sprite in self._obj.arrow_sprite_list:
                        node.remove_arrow_from_shape_list(self._obj)
                if node.type == NodeType.NEIGH:
                    # here the arrow rotates backwards:
                    node.rotate_arrow(
                        (node.x - node.previously_visited_node.x, node.y - node.previously_visited_node.y))
                    # arrow = DrawLib.create_line_arrow(node, (
                    #     node.x - node.previously_visited_node.x, node.y - node.previously_visited_node.y), self._obj)
                    # node.arrow_shape = arrow
                    # node.append_arrow(self._obj)
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

    @timer
    def full_algo(self):
        # False if game.greedy_ind is None else True
        # transmitting the greedy flag to the Node class: TODO: fix this strange doing <<--
        self._func[0](self._greedy_ind)
        self._nodes_to_be_visited = [self._obj.start_node]
        self._obj.start_node.g = 0
        hq.heapify(self._nodes_to_be_visited)
        self._max_times_visited = 0
        # the main cycle:
        while self._nodes_to_be_visited:
            self._iterations += 1
            curr_node = hq.heappop(self._nodes_to_be_visited)
            curr_node.times_visited += 1
            if curr_node not in [self._obj.start_node, self._obj.end_node]:
                curr_node.type = NodeType.TWICE_VISITED if curr_node.times_visited > 1 else NodeType.VISITED_NODE
                curr_node.update_sprite_colour()
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
                        neigh.tiebreaker = self._obj.start_node.tiebreakers[self._tiebreaker](self._obj.start_node,
                                                                                              self._obj.end_node,
                                                                                              neigh)
                    neigh.previously_visited_node = curr_node

                    # if neigh.type not in [NodeType.START_NODE, NodeType.END_NODE]:
                    #     arrow = DrawLib.create_line_arrow(neigh, (neigh.x - curr_node.x, neigh.y - curr_node.y),
                    #                                       self._obj)
                    #     # here the arrow rotates (re-estimating of neigh g-cost):
                    #     if neigh.arrow_shape is not None:
                    #         neigh.remove_arrow(self._obj)
                    #     neigh.arrow_shape = arrow
                    #     neigh.append_arrow(self._obj)

                    # shape = arcade.create_triangles_filled_with_colors(
                    #     [(9, 98), (98, 989), (989, 98989)],
                    #     [arcade.color.BLACK, arcade.color.BLACK, arcade.color.BLACK]
                    # )

                    if neigh.type not in [NodeType.START_NODE, NodeType.END_NODE]:
                        y, x = neigh.y, neigh.x
                        arrow = self._obj.preset_arrows[y][x][(neigh.x - curr_node.x, neigh.y - curr_node.y)]
                        # here the arrow rotates (re-estimating of neigh g-cost):
                        neigh.arrow_shape = arrow

                    hq.heappush(self._nodes_to_be_visited, neigh)
        # for all neighs that left in the heap we must define the nodetype:
        for neigh in self._nodes_to_be_visited:
            neigh.type = NodeType.NEIGH
            neigh.update_sprite_colour()
        # now adding the guiding arrow shapes to the Shape list:
        for neigh in self._nodes_visited.keys() | self._nodes_to_be_visited:
            if neigh.arrow_shape is not None:
                neigh.append_arrow(self._obj)


class WaveLee(Algorithm):

    def __init__(self):
        super().__init__('WaveLee')
        # logger
        self.log = logging.getLogger('WaveLee')
        # wave lee algo's important pars:
        self._front_wave_lee = None
        self._next_wave_lee = None
        self._fronts_dict = None
        self._nodes_visited_q = 0

    def get_nodes_visited_q(self):
        return self._nodes_visited_q

    def clear(self):
        self.base_clear()
        # wave lee algo's important pars:
        self._front_wave_lee = None
        self._next_wave_lee = None
        self._fronts_dict = None
        self._nodes_visited_q = 0

    def get_details(self):
        node_chosen = self._obj.node_chosen
        return f"Node: {node_chosen.y, node_chosen.x}, wave_num: {node_chosen.val}, " \
               f"times visited: {node_chosen.times_visited}, type: {node_chosen.type}"

    @logged()
    def prepare(self):
        # starting attributes' values:
        self._front_wave_lee = []
        self._next_wave_lee = [self._obj.start_node]
        self._obj.start_node.val = 1  # node.val must not be changed during the algo's interactive phase!!!
        self._iterations = 0
        self._fronts_dict = {}
        self._nodes_visited_q = 0

    def algo_up(self):
        self._iterations += 1
        self._front_wave_lee = self._next_wave_lee[:]
        self._fronts_dict[self._iterations] = self._front_wave_lee
        self._nodes_visited_q += len(self._front_wave_lee)
        self._next_wave_lee = []
        for curr_node in self._front_wave_lee:
            curr_node.val = self._iterations
            if curr_node not in [self._obj.end_node, self._obj.start_node]:
                curr_node.type = NodeType.VISITED_NODE
                curr_node.update_sprite_colour()
            if curr_node == self._obj.end_node:
                self._obj.end_node.val = self._iterations
                self.recover_path()
                print(f'PATH RECOVERED')
                break
            for neigh in curr_node.get_neighs(self._obj, [NodeType.START_NODE, NodeType.WALL, NodeType.VISITED_NODE]):
                if neigh.val == 1:  # it is equivalent to if neigh.type == NodeType.EMPTY, TODO: decide if is it needed???
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
                            neigh.append_arrow(self._obj)
                        self._next_wave_lee.append(neigh)
                        neigh.previously_visited_node = curr_node
        print(f'iteration: {self._iterations}, CURRENT FRONT: {self._front_wave_lee}')
        self.log.debug(f'iteration: {self._iterations}, CURRENT FRONT: {self._front_wave_lee}')

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
                    neigh.remove_arrow(self._obj)
            if self._iterations != 0:
                # the front nodes have become NEIGHS:
                for node in self._front_wave_lee:
                    if node != self._obj.start_node:
                        if node != self._obj.end_node:
                            node.type = NodeType.NEIGH
                        node.update_sprite_colour()
                        node.val = 1
                # current and next fronts stepping back:
                self._next_wave_lee = self._front_wave_lee[:]
                self._nodes_visited_q -= len(self._next_wave_lee)
                self._front_wave_lee = self._fronts_dict[self._iterations]
            else:
                # the starting point:
                self._next_wave_lee = [self._obj.start_node]
                self._front_wave_lee = []
            print(f'iteration: {self._iterations}, CURRENT FRONT: {self._front_wave_lee}')
            self.log.debug(f'iteration: {self._iterations}, CURRENT FRONT: {self._front_wave_lee}')

    @timer
    def full_algo(self):
        # other.get_neighs(game)  # Why is it necessary???
        front_wave = {self._obj.start_node}
        iteration = 0
        # wave-spreading:
        while front_wave:
            iteration += 1
            new_front_wave = set()
            for front_node in front_wave:
                front_node.val = iteration  # N
                if front_node not in [self._obj.start_node, self._obj.end_node]:
                    front_node.type = NodeType.VISITED_NODE
                    front_node.update_sprite_colour()
                if front_node == self._obj.end_node:
                    self.recover_path()
                    return
                for front_neigh in front_node.get_neighs(
                        self._obj,
                        [NodeType.START_NODE, NodeType.VISITED_NODE, NodeType.WALL]):
                    if front_neigh not in new_front_wave:
                        front_neigh.previously_visited_node = front_node
                        new_front_wave.add(front_neigh)
            front_wave = set() | new_front_wave


class BfsDfs(Algorithm):

    def __init__(self):
        super().__init__('Bfs/Dfs')
        # logger
        self.log = logging.getLogger('Bfs/Dfs')
        # important algo's attributes:
        self._queue = None
        self._is_bfs = True
        # dicts:
        self._curr_node_dict = {}
        self._neighs_added_to_heap_dict = {}

    @property
    def bfs_dfs_ind(self):
        return 0 if self._is_bfs else 1

    def get_nodes_visited_q(self):
        return len(self._curr_node_dict)

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

    def get_details(self):
        node_chosen = self._obj.node_chosen
        return f"Node: {node_chosen.y, node_chosen.x}, val: {node_chosen.val}, " \
               f"times visited: {node_chosen.times_visited}, type: {node_chosen.type}"

    @logged()
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
            self.recover_path()
        self._neighs_added_to_heap_dict[self._iterations + 1] = set()
        for neigh in curr_node.get_neighs(self._obj, [NodeType.START_NODE, NodeType.VISITED_NODE, NodeType.WALL] + (
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
                    neigh.remove_arrow(self._obj)
                neigh.arrow_shape = arrow
                neigh.append_arrow(self._obj)
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

    @timer
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


class Menu(Drawable, Interactable, FuncConnected):

    def __init__(self):
        super().__init__()
        self._areas = []

    @property
    def areas(self):
        return self._areas

    def is_hidden(self):
        return not self._func[0]()

    def append_area(self, area: 'Area'):
        """appends an area to the menu:"""
        self._areas.append(area)

    def multiple_append(self, *areas: 'Area'):
        """appends several areas to menu at once"""
        for area in areas:
            self.append_area(area)

    def lock(self):
        """prevents menu from changes"""
        for area in self._areas:
            area.lock()

    def unlock(self):
        """makes the menu interactable again"""
        for area in self._areas:
            area.unlock()

    def update(self):
        pass

    def setup(self):
        for area in self._areas:
            area.setup()

    def draw(self):
        if self._func[0]():
            for area in self._areas:
                area.draw()

    def on_motion(self, x, y):
        pass

    def on_press(self, x, y):
        if self._func[0]():
            for area in self._areas:
                area.on_press(x, y)

    def on_release(self, x, y):
        pass

    def on_key_press(self):
        pass

    def on_key_release(self):
        pass


class Area(Drawable, Interactable, FuncConnected):

    def __init__(self, cx, cy, delta, sq_size, sq_line_w, header: str, fields: dict[int, str], no_choice=False):
        super().__init__()
        # logger
        self.log = logging.getLogger('Area')
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

    @logged()
    def choose_field(self, field_chosen_ind: int = 0):
        if 0 <= field_chosen_ind < len(self._fields):
            self._field_chosen = field_chosen_ind
        else:
            self.log.error(
                f"Wrong field's index: {field_chosen_ind} for an area, expected to be in range: [{0},{len(self._fields)})")
            raise IndexError(
                f"Wrong field's index: {field_chosen_ind} for an area, expected to be in range: [{0},{len(self._fields)})")

    def lock(self):
        self._is_locked = True

    def unlock(self):
        self._is_locked = False

    # presets:
    @logged()
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

    @logged()
    def on_press(self, x, y):
        for i in range(len(self._fields)):
            if not self._is_locked and DrawLib.is_point_in_square(
                    self._cx + 10,
                    self._cy - 30 - self._delta * i,
                    self._sq_size,
                    x, y):  # 36 366 98 989
                if self._field_chosen is not None:
                    if i == self._field_chosen:
                        if self._no_choice:
                            self._field_chosen = None
                    else:
                        self._field_chosen = i
                else:
                    self._field_chosen = i
                # set the index in the algo:
                self._func[0](self._field_chosen)

    def on_release(self, x, y):
        pass

    def on_key_press(self):
        pass

    def on_key_release(self):
        pass

    # draws a lock for right window part: 36 366 98 989
    @staticmethod
    def draw_lock(center_x: int, center_y: int, size=14, line_w=2):

        arcade.draw_rectangle_filled(center_x, center_y, size, size, arcade.color.RED)
        arcade.draw_rectangle_outline(center_x, center_y + size / 2, size / 2 + size / 10, size + size / 5,
                                      arcade.color.RED, border_width=line_w)

    # draws the cross of forbiddance:
    @staticmethod
    def draw_cross(center_x: int, center_y: int, delta=9, line_w=2):
        arcade.draw_line(center_x - delta, center_y + delta, center_x + delta, center_y - delta, arcade.color.BLACK,
                         line_width=line_w)
        arcade.draw_line(center_x + delta, center_y + delta, center_x - delta,
                         center_y - delta, arcade.color.BLACK, line_width=line_w)


# class for design element:
class Icon(ABC):
    """abstract class for Icon representation"""

    def __init__(self, cx, cy):
        self._cx = cx
        self._cy = cy
        self._incrementer = 0
        self._inter_type = InterType.NONE
        self._vertices = []

    def clear_inter_type(self):
        self._inter_type = InterType.NONE

    def set_pressed(self):
        self._inter_type = InterType.PRESSED

    def is_pressed(self):
        return self._inter_type == InterType.PRESSED


class PlayButton(Icon, Drawable, Interactable, FuncConnected):
    DELTAS = [0.5, 0.015]  # pixels/radians

    def __init__(self, cx, cy, r, line_w):
        super().__init__(cx, cy)
        # logger
        self.log = logging.getLogger('PlayButton')
        self._incrementer = [0, 0]
        self._r = r
        self._line_w = line_w
        self._multiplier = 1

    @logged()
    def setup(self):
        pass

    def update(self):
        if self._inter_type == InterType.PRESSED:
            self.log.debug('PRESSED play button')
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

    def press(self):
        if self._inter_type == InterType.PRESSED:
            self._inter_type = InterType.HOVERED
            self.log.debug(f'{self._func[0].__name__} call')
            self._func[0](True)
        else:
            self._inter_type = InterType.PRESSED
            self.log.debug(f'{self._func[0].__name__} call')
            self._func[0]()

    def on_press(self, x, y):
        # if self._inter_type != InterType.PRESSED:
        if DrawLib.is_point_in_circle(self._cx, self._cy, self._r, x, y):
            self.press()

    def on_release(self, x, y):
        pass

    def on_key_press(self):
        pass

    def on_key_release(self):
        pass


@class_logged()
@long_pressable
class StepButton(Icon, Drawable, Interactable, FuncConnected):
    DELTAS = [0.15, 0.1, 0.05]
    THRESHOLD = 8
    TICKS_THRESHOLD = 12
    MAGNITUDE = 3

    def __init__(self, cx, cy, a, dh, line_w, is_right=True):
        super().__init__(cx, cy)
        # logger
        self.log = logging.getLogger('StepButton')
        self._incrementer = [0, 0, 0, 0]  # horizontal movement, sin oscillating, blinking and ticks
        self._a = a
        self._dh = dh
        self._line_w = line_w  # TODO: MAY BE SHOULD BE INITIALIZED ONCE FOR THE ENTIRE GAME???
        self._is_right = is_right
        # self._cycle_breaker = False  # TODO: SHOULD BE REWORKED!!! HOW SHOULD THE ALGO KNOW IF IT IS CHANGED???

    # @logged()
    def setup(self):
        pass

    def update(self):
        # print(f'ticks: {self._ticks}')
        # long press logic:
        if self._cycle_breaker:
            self._ticks += 1
            if self._ticks >= self.TICKS_THRESHOLD:
                self._func[0]()
                if self._interactive_incr <= self.THRESHOLD:
                    self._interactive_incr += self.DELTAS[0]
                self._incrementer[1] = (self._incrementer[1] + self.DELTAS[1]) % 3
                self._incrementer[2] += self.DELTAS[2]
        else:  # 36 366 98 989
            if self._interactive_incr > 0:
                self._interactive_incr -= self.DELTAS[0]
            else:
                self._interactive_incr = 0
            self._incrementer[1] = 0
            self._incrementer[2] = 0

    @property
    def cycle_breaker(self):
        return self._cycle_breaker

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
            self._cx += self.multiplier * self._interactive_incr  # self._incrementer[0]
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

            if self._ticks >= self.TICKS_THRESHOLD:
                # if self._incrementer[3] >= self.TICKS_THRESHOLD:
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

    # @logged()
    def press(self):
        self._inter_type = InterType.PRESSED
        if Lastar.is_in_interaction():
            self._cycle_breaker = True
            self._func[0]()
        else:
            self._func[1](self._is_right)

    # @logged()
    def on_press(self, x, y):
        if len(self._vertices) == 3:
            if reduce(lambda a, b: a or arcade.is_point_in_polygon(x, y, self._vertices[b]), list(range(3)), False):
                self.press()

    # @logged()
    def release(self):
        self._inter_type = InterType.NONE
        self._cycle_breaker = False
        self._ticks = 0
        # self._incrementer[3] = 0

    def on_release(self, x, y):
        self.release()

    # keys:
    def on_key_press(self):
        self.press()

    def on_key_release(self):
        self.release()


class Eraser(Icon, Drawable, Interactable, FuncConnected):
    """Icon for clearing grid entirely"""

    def __init__(self, cx, cy, h, w, r, line_w=2):
        super().__init__(cx, cy)
        self.log = logging.getLogger('Eraser')
        self._h = h
        self._w = w
        self._r = r
        self._line_w = line_w
        self._centers = []

    @logged()
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
            self._func[0]()

    def on_release(self, x, y):
        self._inter_type = InterType.NONE

    def on_key_press(self):
        pass

    def on_key_release(self):
        pass


@long_pressable
class Undo(Icon, Drawable, Interactable, FuncConnected):
    """Icon for operations with walls:
     undoing -->> is_right = False
     redoing -->> is_right = True"""

    DELTA = 0.5
    THRESHOLD = 12 - 1
    TICKS_THRESHOLD = 12
    MAGNITUDE = 3

    def __init__(self, cx, cy, a, dh, r, line_w=2, is_right=False):
        super().__init__(cx, cy)
        self.log = logging.getLogger('Undo')
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

    @logged()
    def setup(self):
        ...

    def update(self):
        # long press logic:
        if self._cycle_breaker:
            self._ticks += 1
            if self._ticks >= self.TICKS_THRESHOLD:
                self._func[0]()
                if self._interactive_incr <= self.THRESHOLD:
                    self._interactive_incr += self.DELTA
        else:  # 36 366 98 989
            if self._interactive_incr > 0:
                self._interactive_incr -= self.DELTA
            else:
                self._interactive_incr = 0

    def draw(self):
        start_angle, end_angle = self.angles

        arcade.draw_arc_outline(self._cx, self._cy, 2 * self._r, 2 * self._r, arcade.color.BLACK, start_angle,
                                end_angle, 2 * self.line_w)
        arcade.draw_arc_outline(self._cx, self._cy, 2 * (self._r + self._dh), 2 * (self._r + self._dh),
                                arcade.color.BLACK, start_angle, end_angle,
                                2 * self.line_w)
        arcade.draw_line(self._cx - self.multiplier * self._r, self._cy,
                         self._cx - self.multiplier * (self._r + self._dh), self._cy, arcade.color.BLACK, self.line_w)

        x_shift = self.multiplier * self._interactive_incr

        vs = self._vertices = [
            (
                self._cx - self.multiplier * self._a * math.sqrt(3) / 2 - x_shift,
                self._cy + self._r + self._dh / 2
            ),
            (
                self._cx - x_shift,
                self._cy + self._r + self._dh / 2 + self._a / 2
            ),
            (
                self._cx - x_shift,
                self._cy + self._r + self._dh / 2 - self._a / 2
            )
        ]

        if self._interactive_incr > 0:
            # upper_line:
            arcade.draw_line(
                self._cx,
                self._cy + self._r + self._dh,
                self._cx - x_shift,
                self._cy + self._r + self._dh,
                arcade.color.BLACK,
                self.line_w
            )
            # lower line:
            arcade.draw_line(
                self._cx,
                self._cy + self._r,
                self._cx - x_shift,
                self._cy + self._r,
                arcade.color.BLACK,
                self.line_w
            )

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
            if not Lastar.is_in_interaction():
                self._cycle_breaker = True
                self._func[0]()

    def on_release(self, x, y):
        self._inter_type = InterType.NONE
        self._cycle_breaker = False
        self._ticks = 0

    def on_key_press(self):
        pass

    def on_key_release(self):
        pass


class GearWheelButton(Icon, Drawable, Interactable):
    # does it make sense to log???

    DELTA = 0.02

    def __init__(self, cx: int, cy: int, r: int, cog_size=8, multiplier=1.5, line_w=2, clockwise=True):
        super().__init__(cx, cy)
        self.log = logging.getLogger('GearWheelButton')
        self._r = r
        self._cog_size = cog_size
        self._multiplier = multiplier
        self._line_w = line_w
        self._clockwise = clockwise

    @logged()
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

    def on_key_press(self):
        pass

    def on_key_release(self):
        pass


class ArrowsMenu(Icon, Drawable, Interactable, FuncConnected):  # cx, cy = (1755, 785)
    # does it make sense to log???

    arrows_indices = []
    walk_index = 0
    choosing_arrows = True
    arrows = []
    arrows_reset = None

    def __init__(self, cx, cy, arrow_length, arrow_height):
        super().__init__(cx + 70, cy - 75)
        self.log = logging.getLogger('ArrowsMenu')
        self._inter_types_arrows = [InterType.NONE for _ in range(4)]
        self._inter_type_reset_square = InterType.NONE
        self._arrow_length = arrow_length
        self._arrow_height = arrow_height

        self.elements_setup()

    def elements_setup(self):
        for ind, (dx, dy) in enumerate(Arrow.walk):
            self.arrows.append(
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
        self.arrows_reset = ArrowReset(self._cx, self._cy, self._arrow_height)

    @property
    def inter_types_arrows(self):
        return self._inter_types_arrows

    @inter_types_arrows.setter
    def inter_types_arrows(self, inter_types_arrows):
        self._inter_types_arrows = inter_types_arrows

    @logged()
    def setup(self):
        for arrow in self.arrows:
            arrow.setup()
        self.arrows_reset.setup()

    def update(self):
        pass

    def draw(self):
        if self._func[0]():
            # text:
            arcade.draw_text(f'Directions priority: ', SCREEN_WIDTH - 235, SCREEN_HEIGHT - 70 - 120, arcade.color.BLACK,
                             bold=True)
            # elements:
            for arrow in self.arrows:
                arrow.draw()
            self.arrows_reset.draw()

    def on_motion(self, x, y):
        for arrow in self.arrows:
            arrow.on_motion(x, y)
        self.arrows_reset.on_motion(x, y)

    def on_press(self, x, y):
        if self.choosing_arrows:
            for arrow in self.arrows:
                arrow.on_press(x, y)
        self.arrows_reset.on_press(x, y)

    def on_release(self, x, y):
        pass

    def on_key_press(self):
        pass

    def on_key_release(self):
        pass


class Arrow(Icon, Drawable, Interactable):  # part of an arrow menu
    # does it make sense to log???

    # initial directions priority for all algorithms:
    walk = [(1, 0), (0, -1), (-1, 0), (0, 1)]

    def __init__(self, cx, cy, index, arrow_length, arrow_height, line_w, point: tuple[int, int],
                 colour: tuple[int, int, int]):
        super().__init__(cx, cy)
        self.log = logging.getLogger('Arrow')
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

    @logged()
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

    @logged()
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

    def on_key_press(self):
        pass

    def on_key_release(self):
        pass


class ArrowReset(Icon, Drawable, Interactable):
    # does it make sense to log???

    def __init__(self, cx, cy, arrow_height):
        super().__init__(cx, cy)
        self.log = logging.getLogger('ArrowReset')
        self._arrow_height = arrow_height

    @logged()
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

    @logged()
    def on_press(self, x, y):
        if DrawLib.is_point_in_square(self._cx, self._cy, self._arrow_height, x, y):
            ArrowsMenu.choosing_arrows = True
            ArrowsMenu.walk_index = 0
            ArrowsMenu.arrows_indices = []
            for arrow in ArrowsMenu.arrows:
                arrow._inter_type = InterType.NONE

    def on_release(self, x, y):
        pass

    def on_key_press(self):
        pass

    def on_key_release(self):
        pass


class AstarIcon(Icon, Drawable, Interactable, FuncConnected):
    # does it make sense to log???

    DELTA = 0.05

    def __init__(self, cx, cy, size_w, size_h, line_w=2, clockwise=True):
        super().__init__(cx, cy)
        self.log = logging.getLogger('AstarIcon')
        self._size_w = size_w
        self._size_h = size_h
        self._line_w = line_w
        self._clockwise = clockwise

    @logged()
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

    @logged()
    def on_press(self, x, y):
        if arcade.is_point_in_polygon(x, y, self._vertices[0]):
            if self._inter_type != InterType.PRESSED:
                self._inter_type = InterType.PRESSED
                self._func[0]()
                return self

    def on_release(self, x, y):
        pass

    def on_key_press(self):
        pass

    def on_key_release(self):
        pass


class Waves(Icon, Drawable, Interactable, FuncConnected):
    # does it make sense to log???
    DELTA = 0.25

    def __init__(self, cx, cy, size=32, waves_q=5, line_w=2):
        super().__init__(cx, cy)
        self.log = logging.getLogger('Waves')
        self._size = size
        self._waves_q = waves_q
        self._line_w = line_w

    @logged()
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

    @logged()
    def on_press(self, x, y):
        if DrawLib.is_point_in_circle(self._cx, self._cy, self._size, x, y):
            if self._inter_type != InterType.PRESSED:
                self._inter_type = InterType.PRESSED
                self._func[0]()
                return self

    def on_release(self, x, y):
        pass

    def on_key_press(self):
        pass

    def on_key_release(self):
        pass


class BfsDfsIcon(Icon, Drawable, Interactable, FuncConnected):
    # does it make sense to log???

    DELTA = 0.15

    def __init__(self, cx, cy, size, line_w):
        super().__init__(cx, cy)
        self.log = logging.getLogger('BfsDfsIcon')
        self._size = size
        self._line_w = line_w

    @logged()
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

    @logged()
    def on_press(self, x, y):
        if DrawLib.is_point_in_square(self._cx, self._cy, self._size, x, y):
            if self._inter_type != InterType.PRESSED:
                self._inter_type = InterType.PRESSED
                self._func[0]()
                return self

    def on_release(self, x, y):
        pass

    def on_key_press(self):
        pass

    def on_key_release(self):
        pass


# INFO CLASSES:
class Info(Drawable, FuncConnected):
    """displays some information like heuristical and other important node's pars and so on"""

    def __init__(self, cx, cy, height, line_w=2):  # (cx, cy) -->> left bottom vertex
        super().__init__()
        self.log = logging.getLogger('Info')
        self._cx, self._cy = cx, cy
        self._width, self._height = 0, height
        self._line_w = line_w
        self._text = None

    @logged()
    def setup(self):
        self._text = arcade.Text(
            '',
            self._cx + self._height // 4,
            self._cy - self._height // 4,
            arcade.color.BLACK,
            self._height / 2,
            italic=True, bold=True
        )

    def update(self):
        # text updating per frame:
        self._text.text = self._func[0]()
        self._width = self._text.content_width + self._height / 2

    def draw(self):
        self._text.draw()
        arcade.draw_rectangle_outline(
            self._cx + self._width / 2, self._cy,
            self._width, self._height,
            arcade.color.BLACK, self._line_w
        )

        arcade.draw_rectangle_outline(
            self._cx + self._width / 2, self._cy,
            self._width + 4 * self._line_w, self._height + 4 * self._line_w,
            arcade.color.BLACK, self._line_w
        )

        for j, i in [(1, 1), (-1, 1), (-1, -1), (1, -1)]:
            arcade.draw_line(
                self._cx + self._width / 2 + j * self._width / 2,
                self._cy + i * self._height / 2,
                self._cx + self._width / 2 + j * (self._width / 2 + 2 * self._line_w),
                self._cy + i * (self._height / 2 + 2 * self._line_w),
                arcade.color.BLACK, self._line_w
            )


# LEVI GIN AREA:
class MessageBox(Drawable, Interactable):
    # TODO: to implement after UI

    def setup(self):
        pass

    def update(self):
        pass

    def draw(self):
        pass

    def on_motion(self, x, y):
        pass

    def on_press(self, x, y):
        pass

    def on_release(self, x, y):
        pass

    def on_key_press(self):
        pass

    def on_key_release(self):
        pass


class SpriteTriangle(arcade.Sprite):
    """
    This sprite is just a triangle sprite of one solid color. No need to
    use an image file.


    :param float size: base size of the triangle
    :param Color color: Color of the triangle

    """

    def __init__(self, size: int, color: arcade.Color):
        super().__init__()

        # determine the texture's cache name
        cache_name = self._build_cache_name("triangle_texture", size, color[0], color[1], color[2])

        # use the named texture if it was already made
        if cache_name in load_texture.texture_cache:  # type: ignore
            texture = load_texture.texture_cache[cache_name]  # type: ignore

        # generate the texture if it's not in the cache
        else:
            texture = self._make_triangle_texture(size, color, name=cache_name)
            load_texture.texture_cache[cache_name] = texture  # type: ignore

        # apply results to the new sprite
        self.texture = texture
        self._points = self.texture.hit_box_points

    def _build_cache_name(*args: Any, separator: str = "-") -> str:
        """
        Generate cache names from the given parameters

        This is mostly useful when generating textures with many parameters

        :param args: params to format
        :param separator: separator character or string between params

        :return: Formatted cache string representing passed parameters
        """
        return separator.join([f"{arg}" for arg in args])

    def _make_triangle_texture(self, base: int, color: arcade.Color = arcade.color.BLACK,
                               name: str = None) -> arcade.Texture:
        # name for cashing:
        name = name or self._build_cache_name("triangle_texture", base, color[0], color[1], color[2])
        # img creating:
        bg_color = (0, 0, 0, 0)  # fully transparent
        img = PIL.Image.new("RGBA", (base, base), bg_color)
        draw = PIL.ImageDraw.Draw(img)
        # 3 sides for triangle:
        b = base - 1
        _h = b // 6
        # TODO: regular polygon -->> polygon (triangle must be right for correct visualization)...
        # draw.regular_polygon((r, r, r), n_sides=3, fill=color)
        draw.polygon(((5 * _h, 3 * _h), (2 * _h, 0), (2 * _h, 6 * _h)), fill=color)
        # draw.ellipse((0, 0, diameter - 1, diameter - 1), fill=color)
        # new Texture returning:
        return arcade.Texture(name, img)


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

# TODO: LAYERS OF DRAWING
# TODO: LOGGING OF MOUSE MOVEMENT, KEYS
# TODO: BETTER VISUALIZATION FOR FULL ALGO!!!
# TODO: GUIDING ARROWS OPTION FOR FULL ALGO!!!
# TODO: TRY SOMETHING WITH GETATTR OVERRIDING...
# TODO: PRESETS FOR ARROWS
# TODO: MULTIPROCESSING FOR GUIDING ARROWS INITIALIZATION
# TODO: SPRITE-GUIDING_ARROWS AND THEIR FURTHER ROTATION/DRAWING
# TODO: BACKGROUND SAVING DAEMON PROCESS, BACKGROUND INITIALIZATION