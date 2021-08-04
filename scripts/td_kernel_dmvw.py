import time

import numpy as np
import matplotlib.pyplot as plt
import matrix_operations as mo

__author__ = u'Stephan Muller'
__copyright__ = u'2017, Stephan Muller'
__license__ = u'MIT'
__credits__ = ["Achim Lilienthal", "Patrick Neumann", "Victor Hernandez"]


class TDKernelDMVW(object):
    def __init__(self, min_x, min_y, max_x, max_y, cell_size, kernel_size, wind_scale, time_scale, confidence_scale=1,
                 real_time=False, low_confidence_calculation_zero=False, evaluation_radius=0,
                 cell_interconnections=None):

        """TDKernelDMVW generates a gas distribution map with time, location, gas concentration and wind measurements.
        The model computes the distribution map for the bounding box of points (min_x, min_y) and (max_x, max_y).
        This class is not thread safe yet. Do not call any of the functions synchronously!

        :param min_x: x-coordinate of minimum point
        :param min_y: y-coordinate of minimum point
        :param max_x: x-coordinate of maximum point
        :param max_y: y-coordinate of maximum point
        :param cell_size: size of cell in meters
        :param kernel_size: kernel size of cell in meters
        :param wind_scale: heuristic parameter which influences stretching of the kernel
        :param time_scale: heuristic parameter which influences the ageing of measurements
        :param confidence_scale: heuristic parameter which influences the confidence values
        :param real_time: If set to true, comparison time for time dependent calculation equals system time. If set to
        false, comparison time for time dependent calculation equals last inserted measurement time
        :param low_confidence_calculation_zero If set to true, r_0 and v_0 are set to zero
        :param evaluation_radius: Restrict cell evaluation of single measurements for that given radius. Set zero to
        evaluate all cells.
        :rtype: TDKernelDMVW
        """
        # initialize grid dimension
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y

        # initialize heuristic algorithm factors
        self.cell_size = cell_size
        self.kernel_size = kernel_size
        self.confidence_scale = confidence_scale
        self.wind_scale = wind_scale
        self.time_scale = time_scale

        # initialize real_time
        self.real_time = real_time

        # initialize calculation method for r_0, v_0
        self.low_confidence_calculation_zero = low_confidence_calculation_zero

        # evaluation radius
        self.evaluation_radius = evaluation_radius
        self.cell_interconnections = cell_interconnections

        # always ceil max_point to the next multiple of cell_size
        self.number_of_x_cells = int(np.ceil((self.max_x - self.min_x) / self.cell_size + 1))
        self.number_of_y_cells = int(np.ceil((self.max_y - self.min_y) / self.cell_size + 1))
        # rounding is necessary because of python floating point arithmetic,
        # see https://docs.python.org/3/tutorial/floatingpoint.html
        self.max_x = self.min_x + round(self.number_of_x_cells * self.cell_size, 10)
        self.max_y = self.min_y + round(self.number_of_y_cells * self.cell_size, 10)

        # create cell_grid
        self.cell_grid_x, self.cell_grid_y = np.mgrid[self.min_x:self.max_x:self.cell_size, self.min_y:self.max_y:self.cell_size]

        # initialize maps
        # Omega
        self.importance_weight_map = None
        # R
        self.importance_weight_concentration_map = None
        # alpha
        self.confidence_map = None
        # r
        self.mean_map = None
        # v
        self.variance_map = None

        # initialize measurement vectors
        self.measurement_positions_x = None
        self.measurement_positions_y = None
        self.measurement_concentrations = None
        self.measurement_wind_directions = None
        self.measurement_wind_speeds = None
        self.measurement_timestamps = None

        # initialize gaussian factors
        self.norm_fact = 1 / (np.sqrt(2 * np.pi) * self.kernel_size)
        self.exp_fact = 1 / (2 * self.kernel_size ** 2)

        # initialize confidence scaling
        self.sigma_omega = self.confidence_scale * self.norm_fact

    def set_measurements(self, x, y, concentration, timestamp, wind_speed, wind_direction):

        """Put new measurements into measurement vectors. Either all parameters can be single numbers or vectors with same length.
        :param x: x position
        :param y: y position
        :param concentration: concentration
        :param wind_speed: wind speed in m/s
        :param wind_direction: wind direction in degree. 0 is along positive x-axis, 90 is along positive y-axis.
        :param timestamp: time as timestamp
        """
        self.measurement_positions_x = x
        self.measurement_positions_y = y
        self.measurement_concentrations = concentration
        self.measurement_wind_speeds = wind_speed
        self.measurement_wind_directions = wind_direction
        self.measurement_timestamps = timestamp

    def calculate_maps(self):

        """Calculate all maps based on measurements
        """
        # set current timestamp
        if self.real_time:
            current_timestamp = time.time()
        else:
            current_timestamp = self.measurement_timestamps[self.get_measurements_length() - 1]

        # Omega
        self.importance_weight_map = np.zeros((self.number_of_x_cells, self.number_of_y_cells))

        # R
        self.importance_weight_concentration_map = np.zeros((self.number_of_x_cells, self.number_of_y_cells))

        for i in range(0, self.get_measurements_length()):
            x = self.measurement_positions_x[i]
            y = self.measurement_positions_y[i]

            a = self.kernel_size + self.wind_scale * self.measurement_wind_speeds[i]
            b = self.kernel_size / (1 + (self.wind_scale * self.measurement_wind_speeds[i]) / self.kernel_size)

            covariance = np.array([(a * a, 0), (0, b * b)])
            #print("covariance:", covariance)

            wind_direction_rad = np.deg2rad(self.measurement_wind_directions[i])
            #print("wind_direction_rad:", wind_direction_rad)
            rotation = np.array([(np.cos(wind_direction_rad), -np.sin(wind_direction_rad)),
                                 (np.sin(wind_direction_rad), np.cos(wind_direction_rad))])
            #print("rotation:", rotation)
            covariance_matrix = np.dot(np.dot(rotation, covariance), np.transpose(rotation))
            #print("covariance_matrix:", covariance_matrix)

            sigma_x_sq = covariance_matrix[0][0]
            sigma_x = np.sqrt(sigma_x_sq)
            sigma_y_sq = covariance_matrix[1][1]
            sigma_y = np.sqrt(sigma_y_sq)
            sigma_xy = covariance_matrix[0][1]

            rho = sigma_xy / (sigma_x * sigma_y)

            norm_fact = 1 / (2 * np.pi * sigma_x * sigma_y * np.sqrt(1 - rho ** 2))

            min_x = round((x - self.min_x) / self.cell_size, 0)
            min_y = round((y - self.min_y) / self.cell_size, 0)

            count_cells = self.evaluation_radius * self.kernel_size / self.cell_size
            low_index_x = int(np.max([0, min_x - count_cells]))
            high_index_x = int(np.min([self.number_of_x_cells - 1, min_x + count_cells]))
            low_index_y = int(np.max([0, min_y - count_cells]))
            high_index_y = int(np.min([self.number_of_y_cells - 1, min_y + count_cells]))

            cell_grid_x = self.cell_grid_x[low_index_x:high_index_x, low_index_y:high_index_y]
            cell_grid_y = self.cell_grid_y[low_index_x:high_index_x, low_index_y:high_index_y]

            c_sq = (((cell_grid_x - x) ** 2 / sigma_x_sq) + ((cell_grid_y - y) ** 2 / sigma_y_sq) -
                    ((2 * rho * (cell_grid_x - x) * (cell_grid_y - y)) / (sigma_x * sigma_y)))

            omega = norm_fact * np.exp(-(0.5 / (1 - rho ** 2)) * c_sq)

            # time dependency
            omega = omega * np.exp(-self.time_scale * (current_timestamp - self.measurement_timestamps[i]))

            self.importance_weight_map[low_index_x:high_index_x, low_index_y:high_index_y] += omega

            self.importance_weight_concentration_map[low_index_x:high_index_x, low_index_y:high_index_y] \
                += omega * self.measurement_concentrations[i]

        # p_key = (x, y)

        # if self.cell_interconnections and self.cell_interconnections[p_key]:
        #     interc = self.cell_interconnections[p_key]
        #
        #     open_set = interc
        #     closed_set = set()
        #     while len(open_set) > 0:
        #         e1 = open_set.pop()
        #
        #         dist = np.linalg.norm(np.array(e1) - np.array(p_key))
        #         if dist <= self.evaluation_radius:
        #             closed_set.add(e1)
        #
        #             for e2 in self.cell_interconnections[e1]:
        #                 if e2 not in closed_set:
        #                     open_set.add(e2)
        #
        #     closed_set.add(p_key)
        #     print("len closed_set:", len(closed_set))
        #     x_np, y_np = zip(*closed_set)
        #     x_np = np.array(x_np)
        #     y_np = np.array(y_np)
        #
        #     # for p in interc:
        #     #     px, py = p
        #     #     c_sq = (((self.cell_grid_x[px] - x) ** 2 / sigma_x_sq) + ((self.cell_grid_y[py] - y) ** 2 / sigma_y_sq) -
        #     #             ((2 * rho * (self.cell_grid_x[px] - x) * (self.cell_grid_y[py] - y)) / (sigma_x * sigma_y)))
        #     #
        #     #     print("c_sq:", c_sq)
        #     #
        #     #     omega = norm_fact * np.exp(-(0.5 / (1 - rho ** 2)) * c_sq)
        #     #
        #     #     print("omega1:", omega)
        #     #
        #     #     # time dependency
        #     #     omega = omega * np.exp(-self.time_scale * (current_timestamp - self.measurement_timestamps[i]))
        #     #
        #     #     print("omega:", omega)
        #     #     print("self.importance_weight_map[px, py]:", self.importance_weight_map[px, py])
        #     #     self.importance_weight_map[px, py] += omega
        #     #
        #     #     self.importance_weight_concentration_map[px, py] \
        #     #         += omega * self.measurement_concentrations[i]
        #
        #     cell_grid_x = self.cell_grid_x[x_np[:, None], y_np]
        #     cell_grid_y = self.cell_grid_y[x_np[:, None], y_np]
        #
        #     # plt.scatter(cell_grid_x, cell_grid_y, c="blue")
        #     # plt.scatter([x], [y], c="red")
        #     # plt.scatter([self.min_x, self.max_x], [self.min_y, self.max_y], c="black")
        #     # plt.show()
        #
        #     c_sq = (((cell_grid_x - x) ** 2 / sigma_x_sq) + ((cell_grid_y - y) ** 2 / sigma_y_sq) -
        #             ((2 * rho * (cell_grid_x - x) * (cell_grid_y - y)) / (sigma_x * sigma_y)))
        #
        #     omega = norm_fact * np.exp(-(0.5 / (1 - rho ** 2)) * c_sq)
        #
        #     # time dependency
        #     omega = omega * np.exp(-self.time_scale * (current_timestamp - self.measurement_timestamps[i]))
        #
        #     self.importance_weight_map[x_np[:, None], y_np] += omega
        #
        #     self.importance_weight_concentration_map[x_np[:, None], y_np] \
        #         += omega * self.measurement_concentrations[i]

        # alpha
        self.confidence_map = 1 - np.exp(-self.importance_weight_map / (self.sigma_omega * self.sigma_omega))

        # r_0
        if self.low_confidence_calculation_zero:
            r_0 = 0
        else:
            numerator = 0
            denominator = 0

            for i in range(0, self.get_measurements_length()):
                # phi(t_*,t_i)
                phi = np.exp(-self.time_scale * (current_timestamp - self.measurement_timestamps[i]))
                numerator += phi * self.measurement_concentrations[i]
                denominator += phi

            r_0 = numerator / denominator

        # r
        self.mean_map = (self.confidence_map *
                         (self.importance_weight_concentration_map / self.importance_weight_map)) + \
                        ((1 - self.confidence_map) * r_0)

        # initialize numerator and denominator
        numerator = 0
        denominator = 1

        # V
        variance_map_temp = np.zeros((self.number_of_x_cells, self.number_of_y_cells))

        for i in range(0, self.get_measurements_length()):
            x = self.measurement_positions_x[i]
            y = self.measurement_positions_y[i]
            concentration = self.measurement_concentrations[i]
            timestamp = self.measurement_timestamps[i]

            min_x = int(round((self.measurement_positions_x[i] - self.min_x) / self.cell_size, 0))
            min_y = int(round((self.measurement_positions_y[i] - self.min_y) / self.cell_size, 0))

            p_key = (x, y)
            count_cells = self.evaluation_radius * self.kernel_size / self.cell_size
            low_index_x = int(np.max([0, min_x - count_cells]))
            high_index_x = int(np.min([self.number_of_x_cells - 1, min_x + count_cells]))
            low_index_y = int(np.max([0, min_y - count_cells]))
            high_index_y = int(np.min([self.number_of_y_cells - 1, min_y + count_cells]))

            cell_grid_x = self.cell_grid_x[low_index_x:high_index_x, low_index_y:high_index_y]
            cell_grid_y = self.cell_grid_y[low_index_x:high_index_x, low_index_y:high_index_y]

            distance_matrix = mo.get_distance_matrix(cell_grid_x, cell_grid_y, x, y)

            # (r_i - r^{(k(i))})^2
            variance_factor = (concentration - self.mean_map[min_x][min_y]) ** 2

            # V
            variance_map_temp[low_index_x:high_index_x, low_index_y:high_index_y] += self.norm_fact * np.exp(-(distance_matrix ** 2) * self.exp_fact) * variance_factor

            # phi(t_*,t_i)
            phi = np.exp(-self.time_scale * (current_timestamp - timestamp))

            # v_0
            numerator += phi * np.sqrt(variance_factor)
            denominator += phi

        # v_0
        if self.low_confidence_calculation_zero:
            v_0 = 0
        else:
            v_0 = numerator / denominator

        # v
        self.variance_map = self.confidence_map * variance_map_temp / self.importance_weight_map + (1 - self.confidence_map) * v_0

    def get_measurements_length(self):
        return len(self.measurement_positions_x)
