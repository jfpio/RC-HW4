import itertools
from typing import Tuple, Optional

import cv2

import numpy as np
from matplotlib import pyplot as plt

from environment import Environment
from localization_agent_solution import a_star_search
from utils import generate_maze, bresenham


class OccupancyMap:
    def __init__(self, environment):
        """ TODO: your code goes here """

    def point_update(self, pos: Tuple[int, int], distance: Optional[float], total_distance: Optional[float], occupied: bool) -> None:
        """
        Update regarding noisy occupancy information inferred from lidar measurement.
        :param pos: rowcol grid coordinates of position being updated
        :param distance: optional distance from current agent position to the :param pos: (your solution don't have to use it)
        :param total_distance: optional distance from current agent position to the final cell from current laser beam (your solution don't have to use it)
        :param occupied: whether our lidar reading tell us that a cell on :param pos: is occupied or not
        """
        """ TODO: your code goes here """

    def map_update(self, pos: Tuple[float, float], angles: np.ndarray, distances: np.ndarray) -> None:
        """
        :param pos: current agent position in xy in [0; 1] x [0; 1]
        :param angles: angles of the beams that lidar has returned
        :param distances: distances from current agent position to the nearest obstacle in directions :param angles:
        """
        """ TODO: your code goes here """


class MappingAgent:
    def __init__(self, environment):
        """ TODO: your code goes here """

    def step(self) -> None:
        """
        Mapping agent step, which should (but not have to) consist of the following:
            * reading the lidar measurements
            * updating the occupancy map beliefs/probabilities about their state
            * choosing and executing the next agent action in the environment
        """
        """ TODO: your code goes here """

    def visualize(self) -> np.ndarray:
        """
        :return: the matrix of probabilities of estimation of given cell occupancy
        """
        """ TODO: your code goes here """


if __name__ == "__main__":
    maze = generate_maze((11, 11))

    env = Environment(
        maze,
        resolution=1/11/10,
        agent_init_pos=(0.136, 0.136),
        goal_position=(0.87, 0.87),
        lidar_angles=256
    )
    agent = MappingAgent(env)

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
