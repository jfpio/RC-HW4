from queue import PriorityQueue
from typing import Dict, Tuple

import cv2
import numpy as np
from matplotlib import pyplot as plt
from environment import Environment
from utils import generate_maze


from dataclasses import dataclass, field
from typing import Any


@dataclass(eq=True, frozen=True)
class Position:
    x: int
    y: int

    def __add__(self, other):
        return Position(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Position(self.x - other.x, self.y - other.y)


@dataclass(order=True)
class PrioritizedItem:
    priority: float
    item: Any = field(compare=False)


def a_star_search(occ_map: np.ndarray, start: Position, end: Position) -> Dict[
    Position, Position]:
    """
    Implements the A* search with heuristic function being distance from the goal position.
    :param occ_map: Occupancy map, 1 – field is occupied, 0 – is not occupied.
    :param start: Start position from which to perform search
    :param end: Goal position to which we want to find the shortest path
    :return: The dictionary containing at least the optimal path from start to end in the form:
        {start: intermediate, intermediate: ..., almost: goal}
    """
    """ TODO: your code goes here """
    assert occ_map[start.x, start.y] == 0, "cannot start from occupied place"
    assert occ_map[start.x, start.y] == 0, "cannot end in occupied place"

    def heuristic_function(a: Position, b: Position) -> float:
        return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5

    def check_if_position_is_valid(position: Position, grid: np.ndarray) -> bool:
        return 0 <= position.x < grid.shape[0] and 0 <= position.y < grid.shape[1] and grid[position.x, position.y] == 0

    g_score_map = {start: 0}
    came_from = {}

    open_queue = PriorityQueue()
    open_queue.put(PrioritizedItem(heuristic_function(start, end), start))

    while not open_queue.empty():
        current_node = open_queue.get().item
        if current_node == end:
            break

        children_positions = [current_node + Position(x, y) for x, y in ((1, 0), (-1, 0), (0, 1), (0, -1))]
        valid_children_positions = [position for position in children_positions if check_if_position_is_valid(position, occ_map)]

        for new_node in valid_children_positions:
            tentative_g_score = g_score_map[current_node] + 1
            if new_node not in g_score_map or tentative_g_score < g_score_map[new_node]:
                came_from[new_node] = current_node
                g_score_map[new_node] = tentative_g_score
                open_queue.put(PrioritizedItem(heuristic_function(new_node, end), new_node))

    path = [end]
    while path[-1] != start:
        path.append(came_from[path[-1]])
    path = list(reversed(path))
    return {path[i]: path[i + 1] for i in range(len(path) - 1)}

class LocalizationMap:
    def __init__(self, environment):
        """ TODO: your code goes here """

    def position_update_by_motion_model(self, delta: np.ndarray) -> None:
        """
        :param delta: Movement taken by agent in the previous turn.
        It should be one of [[0, 1], [0, -1], [1, 0], [-1, 0]]
        """
        """ TODO: your code goes here """

    def position_update_by_measurement_model(self, distances: np.ndarray) -> None:
        """
        Updates the probabilities of agent position using the lidar measurement information.
        :param distances: Noisy distances from current agent position to the nearest obstacle.
        """
        """ TODO: your code goes here """

    def position_update(self, distances: np.ndarray, delta: np.ndarray = None):
        self.position_update_by_motion_model(delta)
        self.position_update_by_measurement_model(distances)


class LocalizationAgent:
    def __init__(self, environment):
        """ TODO: your code goes here """
        self.environment = environment
        self.position_probabilities = np.zeros_like(self.environment.gridmap, dtype=float)

    def step(self) -> None:
        """
        Localization agent step, which should (but not have to) consist of the following:
            * reading the lidar measurements
            * updating the agent position probabilities
            * choosing and executing the next agent action in the environment
        """
        """ TODO: your code goes here """
        lidar_measurements = self.environment.lidar()

        start_position = Position(*self.environment.xy_to_rowcol(self.environment.position()))
        end_position = Position(*self.environment.xy_to_rowcol(self.environment.goal_position))
        mapping = a_star_search(
            occ_map=self.environment.gridmap,
            start=start_position,
            end=end_position
        )
        next_position = mapping[start_position]
        delta = next_position - start_position
        self.environment.step((delta.x, delta.y))


    def visualize(self) -> np.ndarray:
        """
        :return: the matrix of probabilities of estimation of current agent position
        """
        """ TODO: your code goes here """


if __name__ == "__main__":
    maze = generate_maze((11, 11))
    env = Environment(
        maze,
        lidar_angles=3,
        resolution=1 / 11 / 10,
        agent_init_pos=None,
        goal_position=(0.87, 0.87),
        position_stochasticity=0.5
    )
    agent = LocalizationAgent(env)

    while not env.success():
        agent.step()

        # if env.total_steps % 10 == 0:
        #     plt.imshow(agent.visualize())
        #     plt.colorbar()
        #     plt.savefig('/tmp/map.png')
        #     plt.close(plt.gcf())
        #
        #     cv2.imshow('map', cv2.imread('/tmp/map.png'))
        #     cv2.waitKey(1)

    print(f"Total steps taken: {env.total_steps}, total lidar readings: {env.total_lidar_readings}")
