# class Magic:
#
#     walk = [(dx, dy) for dy in range(-1, 2) for dx in range(-1, 2) if (dx, dy) != (0, 0)]
#
#     num = 99
#     _secret_num = 999
#
#     def __init__(self, x, y, val):
#         self.x, self.y = x, y
#         self.val = val
#         self.type = 'EMPTY'
#
#     def __eq__(self, other):
#         return (self.x, self.y, self.val) == (other.x, other.y, other.val)
#
#     def _secret_func(self):
#         ...
#
#     ...
#
#
# magic = Magic(98, 989, 98989)
#
# print(f"Magic's dict: {magic.__class__.__dict__}")
# print(f"magic's dict: {magic.__dict__}")
# print(f"magic's attribute val: {magic.__getattribute__('val')}")
#
# magic.__dict__['val'] = 99999
#
# print(f"magic's attribute val: {magic.__getattribute__('val')}")
import functools
import logging
from logging import config
from abc import ABC


class Logger:
    def __init__(self, func):
        self.func = func
        self.cl = func.__class__
        self.doc = func.__doc__()

    def __call__(self, *args, **kwargs):
        ...


def logged(func):
    @functools.wraps(func)
    def _wrapper(*args, **kwargs):
        obj = args[0]
        obj.log.debug(f'method .{func.__name__}() of {obj.__class__} started')
        obj.log.debug(f"method's description: {func.__doc__}")
        f = func(*args, **kwargs)
        obj.log.debug(f'method .{func.__name__}() of {obj.__class__} successfully finished')
        return f
    return _wrapper


class Ex:
    logging.config.fileConfig('log.conf', disable_existing_loggers=True)

    def __init__(self, name: str, val: int):
        self.log = logging.getLogger('Ex')
        self._name = name
        self._val = val

    @property
    def info(self):
        return f'name: {self._name}, val: {self._val}'

    @logged
    def func(self):
        """lala"""
        squared_val = self._val ** 2
        return squared_val + 1

#
# new_ex = Ex('example', 89)
# print(f'res: {new_ex.func()}')
# print(f'info: [name: {new_ex.func.__name__}, doc: {new_ex.func.__doc__}, {new_ex.func.__class__}]')


def counter(func):
    def reset():
        wrapped.rec_depth = 0
        wrapped.rec_calls = 0

    def wrapped(*args, **kwargs):
        nonlocal depth
        # what for?..
        if depth == 0:
            print(f'fafa')
            reset()
        # depth and calls incrementation:
        depth += 1
        wrapped.rec_calls += 1
        # max depth defining:
        wrapped.rec_depth = max(wrapped.rec_depth, depth)
        f = func(*args, **kwargs)
        # depth backtracking:
        depth -= 1
        return f
    # starts a wrapper:
    depth = 0
    return wrapped


@counter
def rec_seeker(val: int):
    if val == 0:
        return 0
    elif val == 1:
        return 1
    return rec_seeker(val - 1) + rec_seeker(val - 2) + 1


@counter
def function():
    ...

#
# print(f'res: {rec_seeker(15)}')
# print(f'f depth: {rec_seeker.rec_depth}, f calls: {rec_seeker.rec_calls}')
#
# print(f'res: {function()}')
# print(f'f depth: {function.rec_depth}, f calls: {function.rec_calls}')
#
#
#


def long_pressable(cls):
    long_press_attributes = {
        '_cycle_breaker': False,
        '_ticks': 0,
        '_interactive_incr': 0
    }

    for attr_name, attr_val in long_press_attributes.items():
        setattr(cls, attr_name, attr_val)

    k = cls.boo

    def s_foo(self, val: int):
        print(f'Upper covering...')
        print(f'val: {val}')
        k(self)
        print(f'Lower covering...')

    setattr(cls, "boo", s_foo)

    return cls


@long_pressable
class Cl(Ex):
    def __init__(self, name):
        super().__init__('Lala', 98)
        self.name = name

    def boo(self):
        print(f'Boo, {self.name}!')


cl = Cl('class')
cl.boo(98)
print(f'subclasses: {Cl.__subclasses__()}')
print(f'is subclass: {issubclass(Cl, Ex)}')

print(f'dict: {Cl.__dict__}')
print(f'dir: {dir(Cl)}')





























class S:
    ...








