import math

from new_Lastar import Algorithm, NodeType, GridNode, timer


# Witch Doctor's algo (Maxim Vedernikov):
class BellmanFord(Algorithm):

    def __init__(self):
        super().__init__('BellmanFord')
        self._flag = True
        self._negative = False
        self._global_iterations = None
        self._iter_dict = dict()

    def clear(self):
        self.base_clear()

    def get_details(self):
        node_chosen = self._obj.node_chosen
        return f"Node: {node_chosen.y, node_chosen.x}, val: {node_chosen.val}, g: {node_chosen.g} " \
               f"times visited: {node_chosen.times_visited}"

    def get_current_state(self):
        """returns the important pars of the current algo state as f-string"""
        if not self._negative:
            return f"{self._name}'s iters: {sum(self._iter_dict.values())}, " \
                   f"global iters: {self._global_iterations}, path's length:" \
                   f" {len(self._path) if self._path else 'no path'}, " \
                   f"nodes visited: {self.get_nodes_visited_q()}, time elapsed: {self._time_elapsed} ms"  # found still
        else:
            return f"{self._name}'s iters: {self._iterations}, number of full passes: {self._global_iterations}, " \
                   f"no path, negative cycle detected"

    def get_nodes_visited_q(self):
        """Bellman-Ford algorithm visits all the EMPTY nodes (not the WALLS) at least once,
        therefore nodes visited quantity equals the number of empty nodes on the grid"""
        return self._obj.tiles_q * self._obj.hor_tiles_q - len(self._obj.walls)

    def prepare(self):
        # algo pars:
        self._obj.start_node.g = 0
        # iters counters:
        self._iterations = 0
        self._global_iterations = 0
        # neighs initialization:
        self.initialize_neighs()

    def perform_edge_relaxation(self, from_: GridNode, to: GridNode):
        """relaxes an edge on the grid in the direction from 'from_' node to 'to' node"""
        print(f'relaxes the edge of {from_, to}')
        to.g = from_.g + from_.val
        if to.type not in [NodeType.START_NODE, NodeType.END_NODE]:
            to.type = NodeType.VISITED_NODE if to.times_visited == 0 else NodeType.TWICE_VISITED
            to.update_sprite_colour()
        to.previously_visited_node = from_
        to.times_visited += 1
        self._iterations += 1

    def initialize_neighs(self):
        for row in self._obj.grid:
            for node in row:
                node.neighs = list(node.get_neighs(self._obj, [NodeType.WALL]))

    def algo_up(self):
        # one global iteration = a step-up (upper DP steps by curr_path_length var)...
        self._iterations = 0
        self._global_iterations += 1
        self._flag = False
        # inner cycle (iterations, lower DP steps by node vars)
        for row in self._obj.grid:
            for current_node in row:
                if current_node.type != NodeType.WALL:
                    # relaxing all the edges linked to the current node:
                    for neigh in current_node.neighs:
                        self._iterations += 1
                        # TODO: ??? optimization ???
                        # necessary condition for the edge relaxation:
                        if current_node.g != math.inf and current_node.g + current_node.val < neigh.g:
                            self.perform_edge_relaxation(current_node, neigh)
                            # if at least one edge relaxation has been performed -> we can proceed to the next global iteration
                            # after the inner cycle has been completed:
                            self._flag = True
        self._iter_dict[self._global_iterations] = self._iterations
        if not self._flag:
            self.recover_path()
            print(f'dict: ')
            for key, val in self._iter_dict.items():
                print(f'outer_iteration, inner_iterations {key, val}')
            print(f'self._mini_iterations: {self._iterations}')

    def algo_down(self):
        # one global iteration down...
        self._iterations = 0
        self._global_iterations -= 1
        # TODO: difficult to implement, is this worth it?..
        ...

    # APPROVED!!!
    @timer
    def full_algo(self):

        # preparation:
        self._obj.start_node.g = 0

        self._global_iterations = 0
        self._iterations = 0
        # neighs' initialization:
        self.initialize_neighs()
        # here the dynamical programming starts, DP[from_node, to_node, curr_path_length]
        # main (outer) cycle (global iterations, upper DP steps by curr_path_length var)
        for i in range(self._obj.tiles_q * self._obj.hor_tiles_q):
            print(f'GLOBAL ITERATION: {self._global_iterations}')
            self._iterations = 0
            self._global_iterations += 1
            self._flag = False
            # inner cycle (iterations, lower DP steps by node vars)
            for row in self._obj.grid:
                for current_node in row:
                    if current_node.type != NodeType.WALL:
                        # relaxing all the edges linked to the current node:
                        for neigh in current_node.neighs:
                            self._iterations += 1
                            # print(f'LALA')
                            # TODO: ??? optimization ???
                            # necessary condition for the edge relaxation:
                            if current_node.g != math.inf and current_node.g + current_node.val < neigh.g:
                                self.perform_edge_relaxation(current_node, neigh)
                                # if at least one edge relaxation has been performed -> we can proceed to the next global iteration
                                # after the inner cycle has been completed:
                                self._flag = True
            self._iter_dict[self._global_iterations] = self._iterations
            if not self._flag:
                self.recover_path()
                print(f'dict: ')
                for key, val in self._iter_dict.items():
                    print(f'outer_iteration, inner_iterations {key, val}')
                print(f'self._mini_iterations: {self._iterations}')
                return

        # negative cycle check:
        ...

        self._negative = True
