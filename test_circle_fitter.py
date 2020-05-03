#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pytest
import task_data_generator
from circle_fitter import CircleFitter

def test_easy_3_points_circle():
    """Базовый тест из формулировки задачи."""
    N, M = 3, 1
    points = [[0, 0], [1, 0], [0, 1]]
    circle_fitter = CircleFitter(N, M, points)
    result = circle_fitter.fit_circles()
    assert np.array_equal(result, [1, 1, 1])

@pytest.mark.parametrize('seed', np.arange(1, 256, 1))
def test_random_circles(seed):
    """Тесты на случайных данных."""

    np.random.seed(seed)

    N, M, points, true_mask, circle_stack = task_data_generator.create_input_from_random_circles()

    circle_fitter = CircleFitter(N, M, points)

    result = circle_fitter.fit_circles()

    assert task_data_generator.compare_masks(result, true_mask)
