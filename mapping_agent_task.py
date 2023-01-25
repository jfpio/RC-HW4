from typing import Optional

import cv2

import numpy as np
from matplotlib import pyplot as plt

from environment import Environment
from localization_agent_task import a_star_search, Position, check_if_position_is_valid
from utils import generate_maze, bresenham

def check_if_in_bounds(position: Position, grid: np.ndarray) -> bool:
    return 0 <= position.x < grid.shape[0] and 0 <= position.y < grid.shape[1]

class OccupancyMap:
    def __init__(self, environment):
        """ TODO: your code goes here """
        self.environment = environment
        self.probability_map = np.full_like(environment.gridmap, fill_value=0.5)

        self.l0 = np.log(1)
        self.log_odds_map = np.full_like(environment.gridmap, fill_value=self.l0)

    def point_update(self, pos: Position, distance: Optional[float], total_distance: Optional[float],
                     occupied: bool) -> None:
        """
        Update regarding noisy occupancy information inferred from lidar measurement.
        :param pos: rowcol grid coordinates of position being updated
        :param distance: optional distance from current agent position to the :param pos: (your solution don't have to use it)
        :param total_distance: optional distance from current agent position to the final cell from current laser beam (your solution don't have to use it)
        :param occupied: whether our lidar reading tell us that a cell on :param pos: is occupied or not
        """
        """ TODO: your code goes here """
        if not check_if_in_bounds(pos, self.log_odds_map):
            return

        constant = np.log(8 if occupied else 0.1)
        self.log_odds_map[pos.to_tuple()] += constant

        if occupied:
            surrounding_cells = [pos + move for move in [
                Position(1, 0), Position(-1, 0), Position(0, 1), Position(0, -1),
                Position(1, 1), Position(-1, -1), Position(1, -1), Position(-1, 1)]
            ]
            for cell in surrounding_cells:
                self.log_odds_map[cell.to_tuple()] += np.log(4)

    def map_update(self, pos: Position, angles: np.ndarray, distances: np.ndarray) -> None:
        """
        :param pos: current agent position in xy in [0; 1] x [0; 1]
        :param angles: angles of the beams that lidar has returned
        :param distances: distances from current agent position to the nearest obstacle in directions :param angles:
        """
        """ TODO: your code goes here """
        for angle, distance in zip(angles, distances):
            scaled_distance = distance / self.environment.resolution
            end = self.calculate_end_position(pos, angle, scaled_distance)
            cells_to_update = bresenham(pos.to_tuple(), end.to_tuple())
            for cell in cells_to_update:
                cell_position = Position(cell[0], cell[1])
                self.point_update(pos=cell_position, occupied=cell_position == end, distance=None, total_distance=None)

        self.probability_map = self.log_odds_map_to_probability_map(self.log_odds_map)

    @staticmethod
    def log_odds_map_to_probability_map(log_odds_map: np.ndarray) -> np.ndarray:
        """
        :param log_odds_map: log odds map of probabilities
        :return: probability map
        """
        def log_odds_to_probability(log_odds):
            return 1 - (1/(1 + np.exp(log_odds)))

        return log_odds_to_probability(log_odds_map)

    @staticmethod
    def calculate_end_position(position: Position, angle: float, distance: float) -> Position:
        d_x = distance * np.cos(angle)
        d_y = distance * np.sin(angle)
        return position + Position(int(d_x), int(d_y))


class MappingAgent:
    def __init__(self, environment):
        """ TODO: your code goes here """
        self.occupancy_map = OccupancyMap(environment)
        self.environment = environment

    def step(self) -> None:
        """
        Mapping agent step, which should (but not have to) consist of the following:
            * reading the lidar measurements
            * updating the occupancy map beliefs/probabilities about their state
            * choosing and executing the next agent action in the environment
        """
        """ TODO: your code goes here """
        angles, distances = self.environment.lidar()
        position = Position(*self.environment.xy_to_rowcol(self.environment.position()))

        self.occupancy_map.map_update(position, angles, distances)

        casted_occupancy_map = self.occupancy_map.probability_map.copy()
        casted_occupancy_map[casted_occupancy_map < 0.6] = 0
        casted_occupancy_map[casted_occupancy_map >= 0.6] = 1

        my_position = Position(*self.environment.xy_to_rowcol(self.environment.position()))
        a_star_mapping = a_star_search(
            occ_map=casted_occupancy_map,
            start=my_position,
            end=Position(*self.environment.xy_to_rowcol(self.environment.goal_position)),
        )

        next_position = a_star_mapping[my_position]
        delta = next_position - my_position
        self.environment.step(delta.to_tuple())


    def visualize(self) -> np.ndarray:
        """
        :return: the matrix of probabilities of estimation of given cell occupancy
        """
        """ TODO: your code goes here """
        return self.occupancy_map.probability_map


if __name__ == "__main__":
    maze = generate_maze((11, 11))

    env = Environment(
        maze,
        resolution=1 / 11 / 10,
        agent_init_pos=(0.136, 0.136),
        goal_position=(0.87, 0.87),
        lidar_angles=256,
        lidar_stochasticity=0.001
    )
    agent = MappingAgent(env)

    while not env.success():
        agent.step()

        if env.total_steps % 10 == 0:
            plt.imshow(agent.visualize())
            plt.colorbar()
            plt.show()

            # Show works better for my IDE (PyCharm)
            # plt.savefig('/tmp/map.png')
            # plt.close(plt.gcf())
            #
            # cv2.imshow('map', cv2.imread('/tmp/map.png'))
            # cv2.waitKey(1)
    print(f"Total steps taken: {env.total_steps}, total lidar readings: {env.total_lidar_readings}")
