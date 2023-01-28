import math
from typing import Sequence

import arcade
import arcade.gui
import numpy


class MainWindow(arcade.Window):  # 36 366 98 989
    def __init__(self):
        super().__init__(1920, 1080, 'Game')
        self.set_update_rate(1 / 20)
        arcade.set_background_color(arcade.color.AQUAMARINE)
        arcade.timings_enabled()
        # diff views:
        self.menu = Menu(self)
        self.a_star = Astar(self)
        self.wave_lee = WaveLee(self)
        # the start view:
        self.show_view(self.menu)
        # pars:

    def setup(self):
        ...


class Menu(arcade.View):
    def __init__(self, window: MainWindow):
        super().__init__(window)
        self.window = window
        # arcade.set_background_color(arcade.color.BLUEBERRY)
        # self.camera = arcade.Camera()
        self.HUD_camera = arcade.Camera()
        self.manager = arcade.gui.UIManager()
        self.manager.enable()
        self.settings_button = arcade.gui.UIFlatButton(950, 700, 200, 25, 'SETTINGS')
        self.manager.add(self.settings_button)
        self.a_star_button = arcade.gui.UIFlatButton(950, 600, 200, 25, 'A_STAR')
        self.manager.add(self.a_star_button)
        self.wave_lee_button = arcade.gui.UIFlatButton(950, 500, 200, 25, 'WAVE_LEE')
        self.manager.add(self.wave_lee_button)
        self.test_button = arcade.gui.UITextureButton(950, 300, 200, 25, text='TEST')
        self.manager.add(self.test_button)
        self.test_button.on_click = self.test_button_clicked
        self.twist_angle = 0

        # self.test_button.event()

        @self.test_button.event
        def on_mouse_enter(self):
            ...

        @self.test_button.event
        def on_hide_view(self):
            ...

        self.test_button.move()  # running away button
        # click:
        self.settings_button.event()
        ...

    def update(self, delta_time: float):
        self.twist_angle += 0.02

    def test_button_clicked(self, event):
        ...

    def on_draw(self):
        self.clear()
        self.HUD_camera.use()
        self.manager.draw()
        # for i in range(15):
        #     for j in range(15):
        #         self.draw_gear_wheel(50 + i * (60 + 3), 100 + j * (60 + 3), 28, 7, 2, (i + j) % 2 == 0,
        #                              (i + j) % 2 == 0)

        # self.draw_star(330, 365, 5, 32, 2, is_vortex=True)

        # self.draw_vortex(330, 365, 32, 1, 3)


        # self.draw_gear_wheel(330, 365, 256, 64, 5, True, False)
        # self.draw_gear_wheel(330, 365, 128, 32, 4)
        # self.draw_gear_wheel(330, 365, 64, 16, 3, True, False)
        # self.draw_gear_wheel(250, 365, 32, 32, 8, 2, True, False)
        # self.draw_gear_wheel(330, 365, 16, 4, 1, True, False)
        # self.draw_gear_wheel(50, 100, 28, 7, 2, True, False)


    def draw_start(self):
        # arcade.draw_circle_outline()
        # arcade.draw_circle_outline()
        ...

    def draw_star(self, cx, cy, vertices=5, r=32, line_w=2, is_vortex=False, clockwise=True, max_angle=123):
        delta_angle = 2 * math.pi / vertices
        max_phi = max_angle * 2 * math.pi
        d = vertices // 2
        angle = self.twist_angle if clockwise else -self.twist_angle  # in radians
        for i in range(int(max_angle * vertices) if is_vortex else vertices):
            # _a = (i - d) * delta_angle + angle
            da = d * delta_angle
            m, m_ = (angle / (2 * d * math.pi), (angle + da) / (2 * d * math.pi)) if is_vortex else (1, 1)
            arcade.draw_line(cx + m * r * math.cos(angle),
                             cy + m * r * math.sin(angle),
                             cx + m_ * r * math.cos(angle + da),
                             cy + m_ * r * math.sin(angle + da),
                             arcade.color.BLACK, line_w)
            angle += da
            # angle = (angle + da) % max_phi

    def draw_vortex(self, cx, cy, r=32, cog_size=6, line_w=2, is_vortex=True, shift=False, clockwise=True,
                    max_angle=7.5, linear_incr=1.75):
        arcade.draw_circle_outline(cx, cy, r - cog_size, arcade.color.BLACK, line_w)
        circumference = 2 * math.pi * r  # approximately if cog_size << radius
        angular_size = (2 * math.pi) * cog_size / circumference
        max_cogs_fit_in_the_gear_wheel = int(circumference / cog_size)
        cogs_q = max_cogs_fit_in_the_gear_wheel // 2
        fit_angular_size = (2 * math.pi - cogs_q * angular_size) / cogs_q
        max_phi = max_angle * 2 * math.pi
        k = math.log(linear_incr) / (2 * math.pi)
        colour_list = [arcade.color.BLACK,
                       arcade.color.GRAY]  # arcade.color.BLUEBERRY, arcade.color.PURPLE, arcade.color.BLUE, arcade.color.YELLOW, arcade.color.GREEN, arcade.color.RED]
        # power = math.log(r, cog_size)
        # print(f'circumference: {circumference}, angular_size: {angular_size}, \nmax_cogs_fit_in_the_gear_wheel: {max_cogs_fit_in_the_gear_wheel}, cogs_q: {cogs_q}')
        angle = (angular_size if shift else 0) + (self.twist_angle if clockwise else -self.twist_angle)  # in radians
        for i in range(int(max_angle * cogs_q) + 1):
            _a, a_ = (angle - angular_size / 2), (angle + angular_size / 2)
            _c, c_ = (math.e ** (k * _a), math.e ** (k * a_)) if is_vortex else (1, 1)
            _r, r_ = _c * r, c_ * r
            _rx, _ry, rx_, ry_ = _r * math.cos(_a), _r * math.sin(_a), r_ * math.cos(a_), r_ * math.sin(a_)
            _dx, _dy = _c * cog_size * math.cos(angle), _c * cog_size * math.sin(angle)
            dx_, dy_ = c_ * cog_size * math.cos(angle), c_ * cog_size * math.sin(angle)
            arcade.draw_polygon_filled(
                [
                    [cx + _rx, cy + _ry],
                    [cx + _rx + 12 * _dx, cy + _ry - 12 * _dy],
                    [cx + rx_ + 12 * dx_, cy + ry_ - 12 * dy_],
                    [cx + rx_, cy + ry_]
                ],
                colour_list[0] if i == int(max_angle * cogs_q) else colour_list[1])
            # arcade.draw_line(cx + _rx,
            #                  cy + _ry,
            #                  cx + _rx + dx,
            #                  cy + _ry + dy,
            #                  arcade.color.BLACK, line_w)
            # arcade.draw_line(cx + _rx + dx,
            #                  cy + _ry + dy,
            #                  cx + rx_ + dx,
            #                  cy + ry_ + dy,
            #                  arcade.color.BLACK, line_w)
            # arcade.draw_line(cx + rx_ + dx,
            #                  cy + ry_ + dy,
            #                  cx + rx_,
            #                  cy + ry_,
            #                  arcade.color.BLACK, line_w)
            arcade.draw_line(cx + _rx,
                             cy + _ry,
                             cx + rx_,
                             cy + ry_,
                             arcade.color.BLACK, line_w)
            # arcade.draw_arc_filled(cx, cy, 2 * r, 2 * r, arcade.color.BLACK, math.degrees(angle + angular_size / 2), math.degrees(angle + angular_size / 2 + fit_angular_size))
            _a_ = a_ + fit_angular_size
            c = math.e ** (k * _a_) if is_vortex else 1
            _r_ = c * r
            arcade.draw_line(cx + rx_,
                             cy + ry_,
                             cx + _r_ * math.cos(_a_),
                             cy + _r_ * math.sin(_a_),
                             arcade.color.BLACK, line_w)
            # arcade.draw_points() -->> instead of line in order to fit the circle's arcs more accurate
            angle = (angle + angular_size + fit_angular_size) % max_phi

    def draw_gear_wheel(self, cx, cy, rx=32, ry=32, cog_size=8, line_w=2, shift=False, clockwise=True):
        circumference = math.pi * (rx + ry)  # approximately if cog_size << radius
        angular_size = (2 * math.pi) * cog_size / circumference
        max_cogs_fit_in_the_gear_wheel = int(circumference / cog_size)
        cogs_q = max_cogs_fit_in_the_gear_wheel // 2
        fit_angular_size = (2 * math.pi - cogs_q * angular_size) / cogs_q
        angle = (angular_size if shift else 0) + (self.twist_angle if clockwise else -self.twist_angle)  # in radians
        upper_vertices_list = []
        for i in range(cogs_q):
            # aux pars:
            _a, a_ = (angle - angular_size / 2), (angle + angular_size / 2)
            _rx, _ry, rx_, ry_ = rx * math.cos(_a), ry * math.sin(_a), rx * math.cos(a_), ry * math.sin(a_)
            _dx, _dy = cog_size * math.cos(angle), cog_size * math.sin(angle)
            dx_, dy_ = cog_size * math.cos(angle), cog_size * math.sin(angle)
            # polygon's points:
            upper_vertices_list.append([cx + _rx, cy + _ry])
            upper_vertices_list.append([cx + _rx + _dx, cy + _ry + _dy])
            upper_vertices_list.append([cx + rx_ + dx_, cy + ry_ + dy_])
            upper_vertices_list.append([cx + rx_, cy + ry_])
            # angle incrementation:
            angle += angular_size + fit_angular_size
        # upper gear wheel:
        arcade.draw_polygon_filled(upper_vertices_list, arcade.color.GRAY)
        arcade.draw_polygon_outline(upper_vertices_list, arcade.color.BLACK, line_w)
        # hole:
        arcade.draw_ellipse_filled(cx, cy, 2 * (rx - 1.5 * cog_size), 2 * (ry - 1.5 * cog_size), arcade.color.AQUAMARINE)
        arcade.draw_ellipse_outline(cx, cy, 2 * (rx - 1.5 * cog_size), 2 * (ry - 1.5 * cog_size), arcade.color.BLACK, line_w)

    def draw_gear_wheel_3d(self, cx, cy, rx=32, ry=32, h=8, cog_size=8, line_w=2, shift=False, clockwise=True):
        circumference = math.pi * (rx + ry)  # approximately if cog_size << radius
        angular_size = (2 * math.pi) * cog_size / circumference
        max_cogs_fit_in_the_gear_wheel = int(circumference / cog_size)
        cogs_q = max_cogs_fit_in_the_gear_wheel // 2
        fit_angular_size = (2 * math.pi - cogs_q * angular_size) / cogs_q
        angle = (angular_size if shift else 0) + (self.twist_angle if clockwise else -self.twist_angle)  # in radians
        upper_vertices_list, lower_shapes_list = [], arcade.ShapeElementList()
        shapes = arcade.ShapeElementList()
        for i in range(cogs_q):
            _a, a_ = (angle - angular_size / 2), (angle + angular_size / 2)
            _rx, _ry, rx_, ry_ = rx * math.cos(_a), ry * math.sin(_a), rx * math.cos(a_), ry * math.sin(a_)
            _dx, _dy = cog_size * math.cos(angle), cog_size * math.sin(angle)
            dx_, dy_ = cog_size * math.cos(angle), cog_size * math.sin(angle)
            upper_vertices_list.append([cx + _rx, cy + _ry])
            upper_vertices_list.append([cx + _rx + _dx, cy + _ry + _dy])
            upper_vertices_list.append([cx + rx_ + dx_, cy + ry_ + dy_])
            upper_vertices_list.append([cx + rx_, cy + ry_])
            if math.pi <= angle % (2 * math.pi) <= 2 * math.pi:
                if 3 / 2 * math.pi <= angle % (2 * math.pi) <= 2 * math.pi:
                    arcade.create_line(cx + _rx,
                                       cy + _ry - h,
                                       cx + _rx + _dx,
                                       cy + _ry + _dy - h,
                                       arcade.color.BLACK, line_w).draw()
                    arcade.create_line(cx + _rx + _dx,
                                       cy + _ry + _dy - h,
                                       cx + rx_ + dx_,
                                       cy + ry_ + dy_ - h,
                                       arcade.color.BLACK, line_w).draw()
                if math.pi <= angle % (2 * math.pi) <= 3 / 2 * math.pi:
                    arcade.create_line(cx + rx_ + dx_,
                                       cy + ry_ + dy_ - h,
                                       cx + rx_,
                                       cy + ry_ - h,
                                       arcade.color.BLACK, line_w).draw()
                _a_ = a_ + fit_angular_size
                arcade.create_line(cx + rx_,
                                   cy + ry_ - h,
                                   cx + rx * math.cos(_a_),
                                   cy + ry * math.sin(_a_) - h,
                                   arcade.color.BLACK, line_w).draw()

                arcade.draw_polygon_filled([
                    [cx + _rx + _dx, cy + _ry + _dy - h],
                    [cx + rx_ + dx_, cy + ry_ + dy_ - h],
                    [cx + rx_ + dx_, cy + ry_ + dy_],
                    [cx + _rx + _dx, cy + _ry + _dy]
                ],
                    arcade.color.GRAY)

                arcade.draw_polygon_filled([
                    [cx + rx_, cy + ry_ - h],
                    [cx + rx * math.cos(_a_), cy + ry * math.sin(_a_) - h],
                    [cx + rx * math.cos(_a_), cy + ry * math.sin(_a_)],
                    [cx + rx_, cy + ry_]
                ],
                    arcade.color.GRAY)

            if math.pi <= angle % (2 * math.pi) < 3 / 2 * math.pi:
                arcade.draw_polygon_filled([
                    [cx + rx_ + dx_, cy + ry_ + dy_ - h],
                    [cx + rx_, cy + ry_ - h],
                    [cx + rx_, cy + ry_],
                    [cx + rx_ + dx_, cy + ry_ + dy_]
                ],
                    arcade.color.GRAY)

            elif 3 / 2 * math.pi <= angle % (2 * math.pi) <= 2 * math.pi:
                arcade.draw_polygon_filled([
                    [cx + _rx, cy + _ry - h],
                    [cx + _rx + _dx, cy + _ry + _dy - h],
                    [cx + _rx + _dx, cy + _ry + _dy],
                    [cx + _rx, cy + _ry]
                ],
                    arcade.color.GRAY)

            if 3 / 2 * math.pi <= angle % (2 * math.pi) <= 2 * math.pi:
                shapes.append(arcade.create_line(cx + _rx,
                                                 cy + _ry,
                                                 cx + _rx,
                                                 cy + _ry - h,
                                                 arcade.color.BLACK, line_w))
            if 0 <= angle % (2 * math.pi) <= math.pi / 2 or math.pi <= angle % (2 * math.pi) <= 2 * math.pi:
                shapes.append(arcade.create_line(cx + _rx + _dx,
                                                 cy + _ry + _dy,
                                                 cx + _rx + _dx,
                                                 cy + _ry + _dy - h,
                                                 arcade.color.BLACK, line_w))
            if math.pi / 2 <= angle % (2 * math.pi) <= math.pi or math.pi <= angle % (2 * math.pi) <= 2 * math.pi:
                shapes.append(arcade.create_line(cx + rx_ + dx_,
                                                 cy + ry_ + dy_,
                                                 cx + rx_ + dx_,
                                                 cy + ry_ + dy_ - h,
                                                 arcade.color.BLACK, line_w))
            if math.pi <= angle % (2 * math.pi) <= 3 / 2 * math.pi:
                shapes.append(arcade.create_line(cx + rx_,
                                                 cy + ry_,
                                                 cx + rx_,
                                                 cy + ry_ - h,
                                                 arcade.color.BLACK, line_w))
            angle += angular_size + fit_angular_size
            # heights:
            # shapes.append(arcade.create_line())
        shapes.draw()
        # upper gear wheel:
        arcade.draw_polygon_filled(upper_vertices_list, arcade.color.GRAY)
        arcade.draw_polygon_outline(upper_vertices_list, arcade.color.BLACK, line_w)
        # lower gear wheel:
        lower_shapes_list.draw()
        # hole:
        arcade.draw_ellipse_filled(cx, cy, rx - cog_size, ry - cog_size, arcade.color.AQUAMARINE)
        arcade.draw_ellipse_outline(cx, cy, rx - cog_size, ry - cog_size, arcade.color.BLACK, line_w)

    @staticmethod
    def get_degrees(radians):
        return radians * 180 / math.pi

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int):
        print(f'y, x: {y, x}, button: {button}')
        self.window.show_view(self.window.a_star)


class Astar(arcade.View):
    def __init__(self, window: MainWindow):
        super().__init__(window)
        self.HUD_camera = arcade.Camera()

    def on_draw(self):
        self.clear()
        arcade.set_background_color(arcade.color.PURPLE)
        self.HUD_camera.use()
        arcade.draw_rectangle_filled(100, 100, 500, 500, arcade.color.BLACK)

    def on_key_press(self, symbol: int, modifiers: int):
        if symbol == arcade.key.ESCAPE:
            ...


class WaveLee(arcade.View):
    def __init__(self, window):
        super().__init__(window)
        self.HUD_camera = arcade.Camera()

    def on_draw(self):
        self.clear()
        arcade.set_background_color(arcade.color.GREEN)
        self.HUD_camera.use()
        arcade.draw_rectangle_filled(1800, 800, 100, 100, arcade.color.RED)


def main():
    game = MainWindow()
    game.setup()
    arcade.run()


if __name__ == '__main__':
    main()

# precessing vortex:

# def draw_gear_wheel(self, cx, cy, r, cog_size, line_w, is_vortex=True, shift=False, clockwise=True, max_angle=5.5, linear_incr=1.78):
#     arcade.draw_circle_outline(cx, cy, r - cog_size, arcade.color.BLACK, line_w)
#     circumference = 2 * math.pi * r  # approximately if cog_size << radius
#     angular_size = (2 * math.pi) * cog_size / circumference
#     max_cogs_fit_in_the_gear_wheel = int(circumference / cog_size)
#     cogs_q = max_cogs_fit_in_the_gear_wheel // 2
#     fit_angular_size = (2 * math.pi - cogs_q * angular_size) / cogs_q
#     max_phi = max_angle * 2 * math.pi
#     k = math.log(linear_incr) / (2 * math.pi)
#     colour_list = [arcade.color.BLACK, arcade.color.GRAY]  # arcade.color.BLUEBERRY, arcade.color.PURPLE, arcade.color.BLUE, arcade.color.YELLOW, arcade.color.GREEN, arcade.color.RED]
#     # power = math.log(r, cog_size)
#     # print(f'circumference: {circumference}, angular_size: {angular_size}, \nmax_cogs_fit_in_the_gear_wheel: {max_cogs_fit_in_the_gear_wheel}, cogs_q: {cogs_q}')
#     angle = (angular_size if shift else 0) + (self.twist_angle if clockwise else -self.twist_angle)  # in radians
#     for i in range(int(max_angle * cogs_q) + 1):
#         _a, a_ = (angle - angular_size / 2), (angle + angular_size / 2)
#         _c, c_ = (math.e ** (k * _a), math.e ** (k * a_)) if is_vortex else (1, 1)
#         _r, r_ = _c * r, c_ * r
#         _rx, _ry, rx_, ry_ = _r * math.cos(_c), _r * math.sin(_c), r_ * math.cos(c_), r_ * math.sin(c_)
#         _dx, _dy = _c * cog_size * math.cos(angle), _c * cog_size * math.sin(angle)
#         dx_, dy_ = c_ * cog_size * math.cos(angle), c_ * cog_size * math.sin(angle)
#         arcade.draw_polygon_filled(
#             [
#                 [cx + _rx, cy + _ry],
#                 [cx + _rx - _dx, cy + _ry - _dy],
#                 [cx + rx_ - dx_, cy + ry_ - dy_],
#                 [cx + rx_, cy + ry_]
#             ],
#             colour_list[0] if i == int(max_angle * cogs_q) else colour_list[1])
#         # arcade.draw_line(cx + _rx,
#         #                  cy + _ry,
#         #                  cx + _rx + dx,
#         #                  cy + _ry + dy,
#         #                  arcade.color.BLACK, line_w)
#         # arcade.draw_line(cx + _rx + dx,
#         #                  cy + _ry + dy,
#         #                  cx + rx_ + dx,
#         #                  cy + ry_ + dy,
#         #                  arcade.color.BLACK, line_w)
#         # arcade.draw_line(cx + rx_ + dx,
#         #                  cy + ry_ + dy,
#         #                  cx + rx_,
#         #                  cy + ry_,
#         #                  arcade.color.BLACK, line_w)
#         arcade.draw_line(cx + _rx,
#                          cy + _ry,
#                          cx + rx_,
#                          cy + ry_,
#                          arcade.color.BLACK, line_w)
#         # arcade.draw_arc_filled(cx, cy, 2 * r, 2 * r, arcade.color.BLACK, math.degrees(angle + angular_size / 2), math.degrees(angle + angular_size / 2 + fit_angular_size))
#         _a_ = a_ + fit_angular_size
#         c = math.e ** (k * _a_) if is_vortex else 1
#         _r_ = c * r
#         arcade.draw_line(cx + rx_,
#                          cy + ry_,
#                          cx + _r_ * math.cos(c),
#                          cy + _r_ * math.sin(c),
#                          arcade.color.BLACK, line_w)
#         # arcade.draw_points() -->> instead of line in order to fit the circle's arcs more accurate
#         angle = (angle + angular_size + fit_angular_size) % max_phi
