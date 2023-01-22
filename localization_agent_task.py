from queue import PriorityQueue
from typing import Dict, Tuple

import cv2
import numpy as np
from matplotlib import pyplot as plt
from environment import Environment
from utils import generate_maze


def a_star_search(occ_map: np.ndarray, start: Tuple[int, int], end: Tuple[int, int]) -> Dict[Tuple[int, int], Tuple[int, int]]:
    """
    Implements the A* search with heuristic function being distance from the goal position.
    :param occ_map: Occupancy map, 1 – field is occupied, 0 – is not occupied.
    :param start: Start position from which to perform search
    :param end: Goal position to which we want to find the shortest path
    :return: The dictionary containing at least the optimal path from start to end in the form:
        {start: intermediate, intermediate: ..., almost: goal}
    """
    """ TODO: your code goes here """


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

    def step(self) -> None:
        """
        Localization agent step, which should (but not have to) consist of the following:
            * reading the lidar measurements
            * updating the agent position probabilities
            * choosing and executing the next agent action in the environment
        """
        """ TODO: your code goes here """

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
        resolution=1/11/10,
        agent_init_pos=None,
        goal_position=(0.87, 0.87),
        position_stochasticity=0.5
    )
    agent = LocalizationAgent(env)

    while not env.success():
        agent.step()

        if env.total_steps % 10 == 0:
            plt.imshow(agent.visualize())
            plt.colorbar()
            plt.savefig('/tmp/map.png')
            plt.close(plt.gcf())

            cv2.imshow('map', cv2.imread('/tmp/map.png'))
            cv2.waitKey(1)

    print(f"Total steps taken: {env.total_steps}, total lidar readings: {env.total_lidar_readings}")
