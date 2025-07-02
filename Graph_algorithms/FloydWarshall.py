import math

from new_Lastar import Algorithm, NodeType


class FloydWarshall(Algorithm):

    def __init__(self):
        super().__init__('FloydWarshall')
        self.adjacency_matrix = None
        self.size_matrix = None
        self._marker = None
        self.list_node = []
        self._mini_iterations = None
        self._global_iterations = None
        self._dict_iters = dict()
        self._flag = True

    def clear(self):
        ...

    def prepare(self):
        ...

    def get_details(self):
        node_chosen = self._obj.node_chosen
        return f"Node: {node_chosen.y, node_chosen.x}, val: {node_chosen.val}, g: {node_chosen.g} " \
               f"times visited: {node_chosen.times_visited}"

    def get_current_state(self):
        """returns the important pars of the current algo state as f-string"""
        return f"{self._name}'s iters: {sum(self._dict_iters.values())}, " \
               f"global iters: {self._global_iterations}, path's length:" \
               f" {len(self._path) if self._path else 'no path'}, " \
               f"nodes visited: {self.get_nodes_visited_q()}, time elapsed: {self._time_elapsed} ms"  # found still

    def get_nodes_visited_q(self):
        """Floyd-Warshall algorithm visits all the EMPTY nodes (not the WALLS) at least once,
        therefore nodes visited quantity equals the number of empty nodes on the grid"""
        return self._obj.tiles_q * self._obj.hor_tiles_q - len(self._obj.walls)

    def algo_up(self):
        ...

    def algo_down(self):
        ...

    def full_algo(self):
        def perform_edge_relaxation(from_y: int, to_x: int, intermediate_n: int):
            if self.list_node[to_x].type and self.list_node[intermediate_n].type not in [NodeType.START_NODE,
                                                                                         NodeType.END_NODE]:
                self.list_node[to_x].type, self.list_node[intermediate_n].type = NodeType.VISITED_NODE if \
                    self.list_node[to_x].times_visited == 0 and self.list_node[
                        intermediate_n].times_visited == 0 else NodeType.TWICE_VISITED
                self.list_node[to_x].update_sprite_colour()
                self.list_node[intermediate_n].update_sprite_colour()
            self.adjacency_matrix[from_y][to_x] = self.adjacency_matrix[intermediate_n][to_x] + \
                                                  self.adjacency_matrix[from_y][intermediate_n]
            self.list_node[to_x].previously_visited_node = self.list_node[intermediate_n]
            self.list_node[to_x].times_visited += 1
            self.list_node[intermediate_n].times_visited += 1
            self._iterations += 1

        # preparation:
        self._global_iterations = 0
        self._obj.start_node.g = 0
        self._mini_iterations = 0

        # инициализируем матрицу смежности с неизвестными расстояниями (число нод = число строк = число столбцов)
        # как вариант можно попробовать сразу создать матрицу, исключая из подсчёта стены
        self.size_matrix = self._obj.hor_tiles_q * self._obj.tiles_q
        self.adjacency_matrix = [[math.inf for _ in range(self.size_matrix)] for _ in
                                 range(self.size_matrix)]

        # neighs' initialization:
        self._marker = 0
        for row in self._obj.grid:
            for node in row:
                self.list_node.append(node)
                node.g = self._marker
                node.neighs = list(node.get_neighs(self._obj, [NodeType.WALL]))
                self._marker += 1

        # вносим в матрицу расстояния между соседями (заполняем её)
        for row in self._obj.grid:
            for node in row:
                self.adjacency_matrix[node.g][node.g] = 0
                for neighs in node:
                    self.adjacency_matrix[node.g][neighs.g] = neighs.val

        #  Работа алгоритма по поиску и релаксации расстояний:
        #  это тоже та ещё дичь, O(n3)...
        #  когда мой ноут представляет, что в цикле будет 22000^3 значений, у него потеют ладошки
        for n in range(self.size_matrix):
            self._global_iterations += 1
            self._iterations = 0
            self._flag = False
            for y in range(len(self.adjacency_matrix)):
                for x in range(len(self.adjacency_matrix)):
                    self._mini_iterations += 1
                    if self.list_node[x].type != NodeType.WALL and self.list_node[y].type != NodeType.WALL:
                        if self.adjacency_matrix[y][x] < self.adjacency_matrix[n][x] + self.adjacency_matrix[y][n]:
                            perform_edge_relaxation(y, x, n)
                            self._flag = True
            self._dict_iters[self._global_iterations] = self._iterations

        if not self._flag:
            self.recover_path()
            print(f'dict: ')
            for key, val in self._dict_iters.items():
                print(f'outer_iteration, inner_iterations {key, val}')
            print(f'self._mini_iterations: {self._mini_iterations}')
            return





