class Magic:

    walk = [(dx, dy) for dy in range(-1, 2) for dx in range(-1, 2) if (dx, dy) != (0, 0)]

    num = 99
    _secret_num = 999

    def __init__(self, x, y, val):
        self.x, self.y = x, y
        self.val = val
        self.type = 'EMPTY'

    def __eq__(self, other):
        return (self.x, self.y, self.val) == (other.x, other.y, other.val)

    def _secret_func(self):
        ...

    ...


magic = Magic(98, 989, 98989)

print(f"Magic's dict: {magic.__class__.__dict__}")
print(f"magic's dict: {magic.__dict__}")
print(f"magic's attribute val: {magic.__getattribute__('val')}")

magic.__dict__['val'] = 99999

print(f"magic's attribute val: {magic.__getattribute__('val')}")
