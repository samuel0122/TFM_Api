
import sys
import unittest
import numpy as np

sys.path.append("src")
from metrics import calculate_F1, calculate_iou

# Suponiendo que las funciones calculate_iou y calculate_F1 han sido importadas

class TestIoU(unittest.TestCase):
        
    def test_iou_no_overlap(self):
        box1 = [0, 0, 1, 1]
        box2 = [2, 2, 3, 3]
        self.assertAlmostEqual(calculate_iou(box1, box2), 0.0, places=7)

    def test_iou_complete_overlap(self):
        box1 = [0, 0, 2, 2]
        box2 = [0, 0, 2, 2]
        self.assertAlmostEqual(calculate_iou(box1, box2), 1.0, places=7)

    def test_iou_partial_overlap(self):
        box1 = [0, 0, 2, 2]
        box2 = [1, 1, 3, 3]
        expected_iou = 1 / 7  # area_intersection=1, area_union=7
        self.assertAlmostEqual(calculate_iou(box1, box2), expected_iou, places=7)

    def test_iou_touching_edges(self):
        box1 = [0, 0, 1, 1]
        box2 = [1, 1, 2, 2]
        self.assertAlmostEqual(calculate_iou(box1, box2), 0.0, places=7)

    def test_iou_one_inside_another(self):
        box1 = [0, 0, 3, 3]
        box2 = [1, 1, 2, 2]
        expected_iou = 1 / 9  # area_intersection=1, area_union=9
        self.assertAlmostEqual(calculate_iou(box1, box2), expected_iou, places=7)


class TestF1(unittest.TestCase):
    def test_f1_no_predictions_no_truths(self):
        y_true = []
        y_pred = []
        f1, _ = calculate_F1(y_true, y_pred)
        self.assertEqual(f1, 1.0)

    def test_f1_perfect_predictions(self):
        y_true = [[0, 0, 1, 1], [2, 2, 3, 3]]
        y_pred = [[0, 0, 1, 1], [2, 2, 3, 3]]
        f1, matched_ious = calculate_F1(y_true, y_pred)
        self.assertEqual(f1, 1.0)
        self.assertEqual(len(matched_ious), 2)
        self.assertTrue(all(iou == 1.0 for iou in matched_ious))

    def test_f1_one_perfect_predictions(self):
        y_true = [[0, 0, 1, 1], [2, 2, 3, 3]]
        y_pred = [[0, 0, 1, 1]]
        f1, matched_ious = calculate_F1(y_true, y_pred)
        expected_f1 = 2 / 3  # precision=1/1, recall=1/2, f1=2/3
        self.assertAlmostEqual(f1, expected_f1, places=7)
        self.assertEqual(len(matched_ious), 1)
        self.assertEqual(matched_ious[0], 1.0)

    def test_f1_one_extra_predictions(self):
        y_true = [[2, 2, 3, 3]]
        y_pred = [[0, 0, 1, 1], [2, 2, 3, 3]]
        f1, matched_ious = calculate_F1(y_true, y_pred)
        expected_f1 = 2 / 3  # precision=1/2, recall=1/1, f1=2/3
        self.assertAlmostEqual(f1, expected_f1, places=7)
        self.assertEqual(len(matched_ious), 1)
        self.assertEqual(matched_ious[0], 1.0)

    def test_f1_some_predictions_matched(self):
        y_true = [[0, 0, 1, 1], [2, 2, 3, 3]]
        y_pred = [[0, 0, 1, 1], [3, 3, 4, 4]]
        f1, matched_ious = calculate_F1(y_true, y_pred)
        expected_f1 = 1 / 2  # precision=1/2, recall=1/2, f1=0.5
        self.assertAlmostEqual(f1, expected_f1, places=7)
        self.assertEqual(len(matched_ious), 1)
        self.assertEqual(matched_ious[0], 1.0)

    def test_f1_no_predictions(self):
        y_true = [[0, 0, 1, 1]]
        y_pred = []
        f1, _ = calculate_F1(y_true, y_pred)
        self.assertEqual(f1, 0.0)

    def test_f1_no_truths(self):
        y_true = []
        y_pred = [[0, 0, 1, 1]]
        f1, _ = calculate_F1(y_true, y_pred)
        self.assertEqual(f1, 0.0)

    def test_f1_partial_overlap(self):
        y_true = [[0, 0, 2, 2]]
        y_pred = [[1, 1, 3, 3]]
        f1, matched_ious = calculate_F1(y_true, y_pred)
        expected_f1 = 0  # precision=0/1, recall=0/1, f1=0
        self.assertAlmostEqual(f1, expected_f1, places=7)
        self.assertEqual(len(matched_ious), 0)

    def test_f1_iou_threshold(self):
        y_true = [[0, 0, 2, 2]]
        y_pred = [[1, 1, 3, 3]]
        f1, matched_ious = calculate_F1(y_true, y_pred, iou_threshold=0.1)
        expected_f1 = 1  # precision=1, recall=1, f1=1
        self.assertAlmostEqual(f1, expected_f1, places=7)
        self.assertEqual(len(matched_ious), 1)
        self.assertAlmostEqual(matched_ious[0], 1/7, places=7)

    def test_f1_iou_threshold_no_match(self):
        y_true = [[0, 0, 2, 2]]
        y_pred = [[1, 1, 3, 3]]
        f1, matched_ious = calculate_F1(y_true, y_pred, iou_threshold=0.5)
        expected_f1 = 0.0  # No match
        self.assertEqual(f1, expected_f1)
        self.assertEqual(len(matched_ious), 0)

class TestF1WithMultipleBoundingBoxes(unittest.TestCase):

    def test_f1_multiple_perfect_matches(self):
        y_true = [[0, 0, 1, 1], [1, 1, 2, 2], [2, 2, 3, 3]]
        y_pred = [[0, 0, 1, 1], [1, 1, 2, 2], [2, 2, 3, 3]]
        f1, matched_ious = calculate_F1(y_true, y_pred)
        self.assertEqual(f1, 1.0)
        self.assertEqual(len(matched_ious), 3)
        self.assertTrue(all(iou == 1.0 for iou in matched_ious))

    def test_f1_multiple_partial_overlaps(self):
        y_true = [[0, 0, 2, 2], [2, 2, 4, 4]]
        y_pred = [[1, 1, 3, 3], [3, 3, 5, 5]]
        f1, matched_ious = calculate_F1(y_true, y_pred, iou_threshold=0.5)
        # Expected precision=0/2, recall=0/2 -> F1=0
        expected_f1 = 0
        self.assertAlmostEqual(f1, expected_f1, places=7)
        self.assertEqual(len(matched_ious), 0)

    def test_f1_mixed_matches_and_non_matches(self):
        y_true = [[0, 0, 1, 1], [1, 1, 2, 2], [2, 2, 3, 3]]
        y_pred = [[0, 0, 1, 1], [3, 3, 4, 4]]
        f1, matched_ious = calculate_F1(y_true, y_pred, iou_threshold=0.5)
        # Expected precision=1/2, recall=1/3 -> F1=2/5
        expected_f1 = 2 / 5
        self.assertAlmostEqual(f1, expected_f1, places=7)
        self.assertEqual(len(matched_ious), 1)
        self.assertTrue(all(iou == 1.0 for iou in matched_ious))

    def test_f1_no_matches(self):
        y_true = [[0, 0, 1, 1], [1, 1, 2, 2], [2, 2, 3, 3]]
        y_pred = [[3, 3, 4, 4], [4, 4, 5, 5]]
        f1, matched_ious = calculate_F1(y_true, y_pred, iou_threshold=0.5)
        # No matches -> F1=0
        self.assertEqual(f1, 0.0)
        self.assertEqual(len(matched_ious), 0)

    def test_f1_some_matches_below_threshold(self):
        y_true = [[0, 0, 2, 2], [2, 2, 4, 4]]
        y_pred = [[1, 1, 3, 3], [3, 3, 5, 5]]
        f1, matched_ious = calculate_F1(y_true, y_pred, iou_threshold=0.75)
        # Expected precision=0, recall=0 -> F1=0 (since IoU<0.75)
        self.assertEqual(f1, 0.0)
        self.assertEqual(len(matched_ious), 0)

    def test_f1_all_predictions_true_but_some_missed(self):
        y_true = [[0, 0, 1, 1], [1, 1, 2, 2], [2, 2, 3, 3]]
        y_pred = [[0, 0, 1, 1], [1, 1, 2, 2]]
        f1, matched_ious = calculate_F1(y_true, y_pred, iou_threshold=0.5)
        # Expected precision=1, recall=2/3 -> F1=4/5
        expected_f1 = 4 / 5
        self.assertAlmostEqual(f1, expected_f1, places=7)
        self.assertEqual(len(matched_ious), 2)
        self.assertTrue(all(iou == 1.0 for iou in matched_ious))
        
    def test_f1_repeated_matches(self):
        y_true = [[0, 0, 1, 1], [2, 2, 3, 3]]
        y_pred = [[0, 0, 1, 1], [0, 0, 1, 1]]
        f1, matched_ious = calculate_F1(y_true, y_pred, iou_threshold=0.5)
        # Expected precision=1/2, recall=1/2 -> F1=1/2
        expected_f1 = 1 / 2
        self.assertAlmostEqual(f1, expected_f1, places=7)
        self.assertEqual(len(matched_ious), 1)
        self.assertEqual(matched_ious[0], 1.0)
        
if __name__ == "__main__":
    unittest.main()
