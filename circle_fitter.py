#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np

class CircleFitter(object):
    """Основной класс для нахождение окружностей RANSAC-like алгоритмом."""
    def __init__(self, N, M, points):
        self.points = np.array(points, dtype=np.float64)
        self.N = N
        self.M = M

    def fit_circles(self):
        """Нахождение окружностей."""
        fitted_circles, points_mask = 0, np.zeros(self.N)

        while fitted_circles < self.M:
            bool_points_mask = (points_mask == 0)

            if bool_points_mask.sum() <= 2:
                fitted_circles, points_mask = 0, np.zeros(self.N)
                continue

            circle = self._create_circle(bool_points_mask)
            while not circle:
                circle = self._create_circle(bool_points_mask)

            circle_points = np.isclose(np.square(self.points - circle[0]).sum(axis=1), circle[1])

            if np.any(np.logical_and(circle_points, bool_points_mask)[np.invert(bool_points_mask)]): # have the points that are layed in more then 1 circle
                                                                                                     # => it contradicts with precondition
                fitted_circles, points_mask = 0, np.zeros(self.N)
                continue

            points_mask = np.where(
                                   circle_points,
                                   fitted_circles + 1, points_mask
                                  )

            fitted_circles += 1

            if fitted_circles == self.M and np.any(points_mask == 0): # find M circles, but there are uncovered points remaining
                fitted_circles, points_mask = 0, np.zeros(self.N)

        return points_mask.astype(int)

    def _take_random_three(self, bool_points_mask):
        """Сэмплирование 3 случайных точек."""
        sampled_points = self.points[bool_points_mask][np.random.choice(np.arange(bool_points_mask.sum(), dtype=int), 3, replace=False), :]

        return np.hstack(((sampled_points[:, 0]**2 + sampled_points[:, 1]**2).reshape(-1, 1), sampled_points, np.ones([3, 1])))

    def _create_circle(self, bool_points_mask):
        """Генерация окружности по точкам."""
        circle_equation_matrix = self._take_random_three(bool_points_mask)

        m11_minor = np.linalg.det(circle_equation_matrix[:, 1:])

        if not m11_minor: # points on the same line
            return None

        m12_minor = np.linalg.det(circle_equation_matrix[:, [0, 2, 3]])
        m13_minor = np.linalg.det(circle_equation_matrix[:, [0, 1, 3]])
        m14_minor = np.linalg.det(circle_equation_matrix[:, [0, 1, 2]])

        center_x = (1/2)*(m12_minor / m11_minor)
        center_y = -(1/2)*(m13_minor / m11_minor)
        radius_sq = center_x**2 + center_y**2 + (m14_minor / m11_minor)

        return np.array([center_x, center_y]), radius_sq

def get_input_line(cast_type):
    return list(map(cast_type, sys.stdin.readline().split()))

def main():
    [N, M] = get_input_line(int)

    points = [get_input_line(float) for _ in range(N)]

    circle_fitter = CircleFitter(N, M, points)
    result = circle_fitter.fit_circles()
    [print(points_id) for points_id in result]

if __name__ == "__main__":
    main()
