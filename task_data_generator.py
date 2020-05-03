#!/usr/bin/env python
# coding: utf-8

import numpy as np
from collections import namedtuple

def create_input_from_random_circles():
    """
    Создает точки из M случайных окружностей (где M в {1, 2, 3}).
    """
    PointBounds = namedtuple("PointBounds", ["lower_bound", "upper_bound"])
    pb = PointBounds(-10e6, 10e6)

    number_of_circles = np.random.randint(1, 4)

    circle_points_num = np.random.randint(3, 3333, number_of_circles)

    points_mask = np.concatenate([np.full(pts_num, i + 1) for i, pts_num in enumerate(circle_points_num)])

    circles_stack = []
    while len(circles_stack) < number_of_circles:
        circle_center = (pb.upper_bound - pb.lower_bound) * np.random.rand(2) + pb.lower_bound

        max_radius = np.min((np.abs(pb.upper_bound - circle_center), np.abs(pb.lower_bound - circle_center)))

        radius =  max_radius * np.random.rand()

        if any([check_circle_intersection(circle_center, radius, other_c, other_r) for other_c, other_r in circles_stack]):
            continue

        circles_stack.append((circle_center, radius))

    all_points = np.concatenate([sample_points_from_circle(N, center, radius) for N, (center, radius)
                                 in zip(circle_points_num, circles_stack)])

    return len(all_points), number_of_circles, all_points, points_mask, circles_stack

def check_circle_intersection(first_center, first_r, second_center, second_r):
    """Проверка окружностей на пересечение."""
    center_dist = np.linalg.norm(first_center - second_center)
    if (center_dist > first_r + second_r) or (np.abs(first_r - second_r) > center_dist):  # circles are distinct enough or one circle inside another
        return False
    else:
        return True

def sample_points_from_circle(N, center, radius):
    """Сэмплирование точек из окружности."""
    return np.array([(radius*np.array([np.cos(angle), np.sin(angle)])) + center
                     for angle in 2*np.pi*np.random.rand(N)])

def compare_masks(first_mask, second_mask):
    """Сравнение ground truth масок и полученными алгоритмом."""
    _, first_mask_counts = np.unique(first_mask, return_counts=True)
    _, second_mask_counts = np.unique(second_mask, return_counts=True)
    return np.array_equal(np.sort(first_mask_counts), np.sort(second_mask_counts))
