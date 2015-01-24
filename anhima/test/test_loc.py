# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import


import unittest
import numpy as np
from anhima.loc import locate_position, locate_interval, locate_positions, \
    locate_intervals


class TestLocatePositions(unittest.TestCase):

    def test_locate_position(self):
        pos = np.array([3, 6, 11])
        self.assertEqual(0, locate_position(pos, 3))
        self.assertEqual(1, locate_position(pos, 6))
        self.assertEqual(2, locate_position(pos, 11))
        self.assertIsNone(locate_position(pos, 1))
        self.assertIsNone(locate_position(pos, 12))

    def test_locate_positions(self):
        pos1 = np.array([3, 6, 11, 20, 35])
        pos2 = np.array([4, 6, 20, 39])
        expect_cond1 = np.array([False, True, False, True, False])
        expect_cond2 = np.array([False, True, True, False])
        cond1, cond2 = locate_positions(pos1, pos2)
        assert np.array_equal(expect_cond1, cond1)
        assert np.array_equal(expect_cond2, cond2)

    def test_locate_interval(self):
        pos = np.array([3, 6, 11, 20, 35])
        self.assertEqual(slice(0, 5), locate_interval(pos, 2, 37))
        self.assertEqual(slice(1, 5), locate_interval(pos, 4, 37))
        self.assertEqual(slice(0, 4), locate_interval(pos, 2, 32))
        self.assertEqual(slice(1, 4), locate_interval(pos, 4, 32))
        self.assertEqual(slice(1, 3), locate_interval(pos, 4, 19))
        self.assertEqual(slice(2, 4), locate_interval(pos, 7, 32))
        self.assertEqual(slice(2, 3), locate_interval(pos, 7, 19))
        self.assertEqual(slice(3, 3), locate_interval(pos, 17, 19))
        self.assertEqual(slice(0, 0), locate_interval(pos, 0, 0))
        self.assertEqual(slice(5, 5), locate_interval(pos, 1000, 2000))

    def test_locate_intervals(self):
        pos = np.array([3, 6, 11, 20, 35])
        intervals = np.array([[0, 2], [6, 17], [12, 15], [31, 35], [100, 120]])
        expect_cond1 = np.array([False, True, True, False, True])
        expect_cond2 = np.array([False, True, False, True, False])
        cond1, cond2 = locate_intervals(pos, intervals[:, 0], intervals[:, 1])
        assert np.array_equal(expect_cond1, cond1)
        assert np.array_equal(expect_cond2, cond2)
        assert np.array_equal([6, 11, 35], pos[cond1])
        assert np.array_equal([[6, 17], [31, 35]], intervals[cond2])
