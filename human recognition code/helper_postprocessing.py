"""
Authors: DIPteam
E-mail: dipteam42@gmail.com
Course: Letna skola za multimediski tehnologii, FEEIT, September 2021
Date: 10.09.2022

Description: function library
             data postprocessing operations: intersection over union, non-maximum suppression
Python version: 3.6

TODO: update nms function
"""

# python imports
import numpy as np
import os
import cv2
def calc_iou(box1, box2):
    """
    calculate intersection over union for two rectangles
    :param box1: list of coordinates: row1, col1, row2, col2 [list]
    :param box2: list of coordinates: row1, col1, row2, col2 [list]
    :return: iou value
    """

    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    # respective area of the two boxes
    boxAArea = (box1[2] - box1[0]) * (box1[3] - box1[1])
    boxBArea = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # overlap area
    interArea = max(xB - xA, 0) * max(yB - yA, 0)

    # IoU
    iou = interArea / (boxAArea + boxBArea - interArea)

    return iou


def line_segments_intersect(x1, x2, y1, y2):
    """
    calculate intersection over union for two 1-d segments
    # Assumes x1 <= x2 and y1 <= y2; if this assumption is not safe, the code
    # can be changed to have x1 being min(x1, x2) and x2 being max(x1, x2) and
    # similarly for the ys.
    :param x1: min_point of first segment
    :param x2: max_point of first segment
    :param y1: min_point of second segment
    :param y2: max_point of second segment
    :return: IoU value
    """

    if x2 >= y1 and y2 >= x1:
        # the segments overlap
        intersection = min(x2, y2) - max(y1, x1)
        union = max(x2, y2) - min(x1, y1)
        iou = float(intersection) / float(union)

        return iou

    return 0


def calc_iou_partwise(box1, box2):
    """
    calculate intersection over union for height and width separately
    :param box1: list of coordinates: row1, col1, row2, col2 [list]
    :param box2: list of coordinates: row1, col1, row2, col2 [list]
    :return: iou_height: iou value by height [float]
             iou_width: iou value by width [float]
    """

    iou_height = line_segments_intersect(box1[0], box1[2], box2[0], box2[2])
    iou_width = line_segments_intersect(box1[1], box1[3], box2[1], box2[3])

    return iou_height, iou_width


def nms_v1(coords, iou_thr):
    """
    return bounding boxes after non-maximum suppression
    all window sizes are considered with the same IOU threshold
    combination strategy: mean value of the coordinates of twoopposite corners
    :param coords: list of coordinates of all bounding boxes [ndarray]
    :param iou_thr: minimum overlap threshold to combine the windows [int]
    :return: list of coordinates of the merged bounding boxes
    """

    new_windows = []  # list of coordinates of output windows
    deletedIndices = []  # indices of bounding boxes which have been merged

    for ind1 in range(np.shape(coords)[1]):  # get initial bbox (ind1)

        # --- find all bboxes which overlap with the initial bbox---

        to_combine = [coords[ind1]]  # initialize list of bounding boxes overlapping with the initial bbox

        for ind2 in range(ind1 + 1, np.shape(coords)[1]):  # iterate through all remaining bounding boxes
            if ind2 not in deletedIndices:

                iou = calc_iou(coords[ind1], coords[ind2])
                

                if iou > iou_thr:
                    print("["+str(ind1)+"]"+str(ind2)+":"+str(iou))
                    to_combine.append(coords[ind2])
                    deletedIndices.append(ind2)

        deletedIndices.append(ind1)  # mark initial bounding box as merged
        to_combine = np.array(to_combine)

        if len(to_combine) > 1:  # if at least one window overlaps with the initial (delete isolated windows)
            # form new window with mean coordinates for the two opposite points
            new = [np.mean(to_combine[:, 0]), np.mean(to_combine[:, 1]), np.mean(to_combine[:, 2]),
                   np.mean(to_combine[:, 3])]
            new_windows.append(new)

    new_windows = np.array(new_windows).astype(np.int)  # coordinates in integer format is needed for the drawing function

    return new_windows
