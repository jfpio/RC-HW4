import math
import time
from queue import PriorityQueue
from typing import Dict, Optional

import cv2
import numpy as np
from matplotlib import pyplot as plt
from environment import Environment
from utils import generate_maze
from scipy import stats

from dataclasses import dataclass, field
from typing import Any

LIDAR_STOCHASTICITY = 0
MOTION_STOCHASTICITY = 0


@dataclass(eq=True, frozen=True)
class Position:
    x: int
    y: int

    def __add__(self, other):
        return Position(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Position(self.x - other.x, self.y - other.y)

    def to_tuple(self):
        return self.x, self.y


@dataclass(order=True)
class PrioritizedItem:
    priority: float
    item: Any = field(compare=False)


def check_if_position_is_valid(position: Position, grid: np.ndarray) -> bool:
    return 0 <= position.x < grid.shape[0] and 0 <= position.y < grid.shape[1] and grid[position.x, position.y] == 0


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

    g_score_map = {start: 0}
    came_from = {}

    open_queue = PriorityQueue()
    open_queue.put(PrioritizedItem(heuristic_function(start, end), start))

    while not open_queue.empty():
        current_node = open_queue.get().item
        if current_node == end:
            break

        children_positions = [current_node + Position(x, y) for x, y in ((1, 0), (-1, 0), (0, 1), (0, -1))]
        valid_children_positions = [position for position in children_positions if
                                    check_if_position_is_valid(position, occ_map)]

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
        self.environment = environment

        probability_for_cell = 1 / np.sum(self.environment.gridmap == 0)
        self.probability_map = np.zeros_like(self.environment.gridmap, dtype=float)
        self.probability_map[self.environment.gridmap == 0] = probability_for_cell

        results = []
        for x in range(self.environment.gridmap.shape[0]):
            for y in range(self.environment.gridmap.shape[1]):
                if self.environment.gridmap[x, y] == 0:
                    _, distances = self.environment.ideal_lidar(self.environment.rowcol_to_xy((x, y)))
                    results.append(np.clip(distances, 1e-25, 1000))
                else:
                    results.append(np.full(shape=(3,), fill_value=np.NaN))

        self.ideal_measurements = np \
            .concatenate(results) \
            .reshape((self.environment.gridmap.shape[0], self.environment.gridmap.shape[1], 3))

    def position_update_by_motion_model(self, delta: Optional[Position]) -> None:
        """
        :param delta: Movement taken by agent in the previous turn.
        It should be one of [[0, 1], [0, -1], [1, 0], [-1, 0]]
        """
        """ TODO: your code goes here """

        if delta is None:
            return

        probability_of_move = 1 - self.environment.position_stochasticity
        new_probability_map = np.zeros_like(self.probability_map)

        for x in range(self.probability_map.shape[0]):
            for y in range(self.probability_map.shape[1]):
                new_position = Position(x, y) + delta

                if self.environment.gridmap[x, y] == 1: # Current cell is obstacle
                    continue

                elif not check_if_position_is_valid(new_position, self.environment.gridmap) \
                        or self.environment.gridmap[new_position.to_tuple()] == 1:
                    new_probability_map[x, y] = self.probability_map[x, y]

                else:
                    new_probability_map[x, y] = (1 - probability_of_move) * self.probability_map[x, y]
                    new_probability_map[new_position.to_tuple()] = probability_of_move * self.probability_map[x, y]

        probability_map_normalization_coefficient = np.sum(new_probability_map)
        new_probability_map = new_probability_map / probability_map_normalization_coefficient

        assert math.isclose(np.sum(new_probability_map), 1.0, abs_tol=0.01) # Too many zeros, error
        self.probability_map = new_probability_map

    def position_update_by_measurement_model(self, distances: np.ndarray) -> None:
        """
        Updates the probabilities of agent position using the lidar measurement information.
        :param distances: Noisy distances from current agent position to the nearest obstacle.
        """
        """ TODO: your code goes here """

        X = distances / self.ideal_measurements
        norm = stats.norm(loc=1, scale=self.environment.lidar_stochasticity)
        weights = norm.pdf(X)
        weights = np.prod(weights, axis=-1)
        weights[self.environment.gridmap == 1] = 0
        normalization_coefficient = np.sum(weights)
        weights = weights / normalization_coefficient

        new_probability_map = self.probability_map * weights
        probability_map_normalization_coefficient = np.sum(new_probability_map)

        new_probability_map = new_probability_map / probability_map_normalization_coefficient

        assert math.isclose(np.sum(new_probability_map), 1.0, abs_tol=0.01) # Too many zeros, error
        self.probability_map = new_probability_map

    def position_update(self, distances: np.ndarray, delta: Optional[Position] = None):
        self.position_update_by_motion_model(delta)
        self.position_update_by_measurement_model(distances)


class LocalizationAgent:
    def __init__(self, environment):
        """ TODO: your code goes here """
        self.environment = environment
        self.localization_map = LocalizationMap(environment)
        self.previous_move = None

    def step(self) -> None:
        """
        Localization agent step, which should (but not have to) consist of the following:
            * reading the lidar measurements
            * updating the agent position probabilities
            * choosing and executing the next agent action in the environment
        """
        """ TODO: your code goes here """

        angles, distances = self.environment.lidar()
        self.localization_map.position_update(distances, self.previous_move)

        positions = []
        probabilities = []
        for x in range(self.localization_map.probability_map.shape[0]):
            for y in range(self.localization_map.probability_map.shape[1]):
                if self.localization_map.probability_map[x, y] > 0:
                    positions.append(Position(x, y))
                    probabilities.append(self.localization_map.probability_map[x, y])

        assert len(positions) > 0
        chosen_position = np.random.choice(positions, p=probabilities)

        mapping = a_star_search(
            occ_map=self.environment.gridmap,
            start=chosen_position,
            end=Position(*self.environment.xy_to_rowcol(self.environment.goal_position))
        )

        next_position = mapping[chosen_position]
        delta = next_position - chosen_position
        delta = Position(0, 1)
        if delta == Position(1, 0):
            print("Down")
        elif delta == Position(-1, 0):
            print("Up")
        elif delta == Position(0, 1):
            print("Right")
        elif delta == Position(0, -1):
            print("Left")

        self.previous_move = delta
        self.environment.step(delta.to_tuple())

    def visualize(self) -> np.ndarray:
        """
        :return: the matrix of probabilities of estimation of current agent position
        """
        """ TODO: your code goes here """
        new_map = self.localization_map.probability_map.copy()
        new_map[self.environment.gridmap == 1] = -0.001
        return new_map


if __name__ == "__main__":
    np.random.seed(420)
    maze = generate_maze((11, 11))
    env = Environment(
        maze,
        lidar_angles=3,
        resolution=1 / 11 / 10,
        agent_init_pos=(0.10, 0.15),
        goal_position=(0.87, 0.87),
        position_stochasticity=0.5,
        lidar_stochasticity=0.1
    )
    agent = LocalizationAgent(env)

    while not env.success():
        agent.step()

        if env.total_steps % 1 == 0:
            plt.imshow(agent.visualize())
            plt.colorbar()
            plt.show()
            time.sleep(1)
            # plt.savefig('/tmp/map.png')
            # plt.close(plt.gcf())
            #
            # cv2.imshow('map', cv2.imread('/tmp/map.png'))
            # cv2.waitKey(0)

    print(f"Total steps taken: {env.total_steps}, total lidar readings: {env.total_lidar_readings}")
