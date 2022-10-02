import os

import cv2
import numpy as np

def func(im_path, an_path):
    imgs = []
    anns = []
    # cita sliki dodava vo lista, od txt file cita anotacii i dodava vo lista
    filenames = [x for x in os.listdir(an_path)]
    for txt in filenames:
        st1 = os.path.join(an_path, txt)
        txt = txt[:-4]
        txt = txt + '.png'
        st = os.path.join(im_path, txt)
        img = cv2.imread(st, 0)
        x = np.loadtxt(st1, delimiter=',', ndmin=2).astype(np.int)
        imgs.append(img)
        for row in range(x.shape[0]):
            x[row, 2] = x[row, 2] + x[row, 0]
            x[row, 3] = x[row, 1] + x[row, 3]
        anns.append(x)

    return imgs, anns


sliki, anotacii = func(r"C:\Users\stef\Desktop\diogen_feit\team1\\", r"C:\Users\stef\Desktop\diogen_feit\annotations\\")
for img in sliki:
    for anx in anotacii.pop(0):
        cv2.rectangle(img, (anx[1], anx[0]), (anx[1] + anx[3], anx[0] + anx[2]), color=(255, 0, 0))
        cv2.imshow('proba', img)
        cv2.waitKey(0)



"""
Authors: DIPteam
E-mail: dipteam42@gmail.com
Course: Letna skola za multimediski tehnologii, FEEIT, September 2021
Date: 10.09.2022

Description: function library
             data operations: load, save, process
Python version: 3.6

TODO:
partwise IOU
"""

# python imports
import os
import numpy as np
import cv2
import random
from tqdm import tqdm

import xml.etree.ElementTree as ET

# custom imports
import helper_postprocessing


def read_data_rpn(im_path, im_size, im_depth, annot_path, exclude_empty, shuffle):
    """
    load, resize, and normalize image data
    loads images and annotations from one folder
    :param im_path: global path of folder containing images of a data subset [string]
    :param im_size: output dimensions of the images (cols, rows) [tuple]
    :param im_depth: required depth of the loaded images (value: 1 or 3) [int]
    :param shuffle: whether to shuffle input data order [bool]
    :return: images_list - array of normalized depth maps [ndarray]
             object_annotations_list - annotated bounding boxes min_row, min_col, max_row, max_col [list]
    """

    images_list = []       # array of normalized images
    object_annotations_list = []       # array of array of bounding boxes for each image

    # list images in source folder
    for im_name in tqdm(os.listdir(os.path.join(im_path))):

        # --- load image ---
        if not im_name[-4:] != '.bmp':  # exclude system files
            continue

        if im_depth == 3:
            image = cv2.imread(os.path.join(im_path, im_name))
        else:
            image = cv2.imread(os.path.join(im_path, im_name), 0)

        if im_size != (image.shape[1], image.shape[0]):
            image = cv2.resize(image, im_size, interpolation=cv2.INTER_AREA)

        image = image.reshape(image.shape[0], image.shape[1], im_depth)

        # --- load annotations ---
        annot_name = str(int(im_name[5:11].lstrip('0')) - 1) + '.xml'

        root = ET.parse(os.path.join(annot_path, annot_name)).getroot()

        objects = []    # list of all objects in the image

        for object in root.findall('object'):

            cl = object.find('class').text

            bb_xml = object.find('bndbox')
            bb = [np.int(bb_xml.find('xmin').text),     # min_col
                  np.int(bb_xml.find('xmax').text),     # max_col
                  np.int(bb_xml.find('ymin').text),     # min_row
                  np.int(bb_xml.find('ymax').text),     # max_row
                  ]

            annot = [bb[2], bb[0], bb[3], bb[1]]    # min_row, min_col, max_row, max_col

            if cl == 'car' and bb[3] - bb[2] + 1 > 25:  # select positive car samples, height > 25px
                objects.append(annot)

        if exclude_empty:
            if len(objects) > 0:
                images_list.append(image)
                object_annotations_list.append(objects)

        else:
            images_list.append(image)
            object_annotations_list.append(objects)

    if len(images_list) == 0:
        print("No images were read.")
        exit(100)

    if shuffle:
        data = list(zip(images_list, object_annotations_list))
        random.shuffle(data)
        images_list, object_annotations_list = zip(*data)

    images_list = np.array(images_list).astype(np.uint8)

    return images_list, object_annotations_list


def get_anchor_data_ssd(bboxes, anchor_dims, img_dims, anchor_stride, iou_low, iou_high, num_negs_ratio):
    """
    generate ground truth output matrices
    multi-output, classifier and regressor branch
    :param bboxes: annotated bounding boxes [min_row, min_col, max_row, max_col] [ndarray]
                   # NOTE: ensure the coordinates are integers
    :param anchor_dims: tuple of anchor dimensions - (height, width) [tuple]
    :param img_dims: (rows, cols, depth) [tuple]
    :param anchor_stride: stride along rows and columns [int]
    :param iou_low: [int]
    :param iou_high: [int]
    :param num_negs_ratio: select X times more negative than positive samples in a frame [int]
    :return:
    """

    output_class_list = []  # output for classifier branch
    output_reg_list = []    # output for regressor branch
    valid_inds = []     # indices of images containing at least one object

    num_anchors = len(anchor_dims)

    for img_ind, img_bboxes in tqdm(enumerate(bboxes)):

        output_dims_class = (np.int(img_dims[0] / anchor_stride), np.int(img_dims[1] / anchor_stride), num_anchors + 1)     # each depth-wise matrix contains classes for one anchor size, last matrix contains negative samples
        output_dims_reg = (np.int(img_dims[0] / anchor_stride), np.int(img_dims[1] / anchor_stride), num_anchors * 4)   # regressor output - 4 values: delta_r, delta_c, delta_h, delta_w

        # output matrices for one image
        output_class = np.zeros(output_dims_class).astype(np.int)
        output_reg = np.zeros(output_dims_reg).astype(np.int)

        # first position of an anchor center
        start_r = np.int(np.round(anchor_stride / 2))
        start_c = np.int(np.round(anchor_stride / 2))

        for output_row, center_row in enumerate(range(start_r, img_dims[0], anchor_stride)):  # iterate through rows of centers
            for output_col, center_col in enumerate(range(start_c, img_dims[1], anchor_stride)):  # iterate through columns of centers

                for anchor_ind, anchor_dim in enumerate(anchor_dims):  # iterate through different anchor dimensions

                    half_anchor_dim_h = np.int(np.round(anchor_dim[0] / 2))
                    half_anchor_dim_w = np.int(np.round(anchor_dim[1] / 2))

                    for bbox in img_bboxes:  # iterate through annotated bounding boxes

                        # --- assign classes: calculate IOU, place 1 or 0 at the required position ---
                        anchor = [max(0, center_row - half_anchor_dim_h),
                                  max(0, center_col - half_anchor_dim_w),
                                  min(center_row - half_anchor_dim_h + anchor_dim[0], img_dims[0]),
                                  min(center_col - half_anchor_dim_w + anchor_dim[1], img_dims[1])]
                        # min_row, min_col, max_row, max_col

                        iou = helper_postprocessing.calc_iou(bbox, anchor)

                        if iou >= iou_high:

                            # positive sample: set class, calculate deltas
                            output_class[output_row, output_col, anchor_ind] = 1

                            # --- set deltas ---
                            # current location minus correct location
                            delta_r = bbox[0] - anchor[0]
                            delta_c = bbox[1] - anchor[1]
                            delta_h = (bbox[2] - bbox[0]) - anchor_dim[0]
                            delta_w = (bbox[3] - bbox[1]) - anchor_dim[1]

                            output_reg[output_row, output_col, anchor_ind * 4 + 0] = delta_r
                            output_reg[output_row, output_col, anchor_ind * 4 + 1] = delta_c
                            output_reg[output_row, output_col, anchor_ind * 4 + 2] = delta_h
                            output_reg[output_row, output_col, anchor_ind * 4 + 3] = delta_w

                        if (iou < iou_high) and (iou > iou_low):
                            # IOU between iou_min and iou_max
                            # class - marked 2, deltas - 0
                            output_class[output_row, output_col, anchor_ind] = 2

        # assign background
        for out_row in range(output_class.shape[0]):  # iterate through rows of output
            for out_col in range(output_class.shape[1]):  # iterate through columns of output

                if sum(output_class[out_row, out_col, :]) == 0:
                    # print(out_row, out_col)
                    output_class[out_row, out_col, -1] = 1  # last matrix contains non-object class

        # replace 2s with 0s
        output_class = np.where(output_class == 2, 0, output_class)

        # remove border objects (object is not fully into frame
        for anchor_dim in anchor_dims:
            border_padding_h = np.int((anchor_dim[0] / anchor_stride) / 2) + 1
            border_padding_w = np.int((anchor_dim[1] / anchor_stride) / 2) + 1

            output_class[0:border_padding_h, :, :] = 0
            output_class[output_class.shape[0] - border_padding_h:, :, :] = 0
            output_class[:, 0:border_padding_w, :] = 0
            output_class[:, output_class.shape[1] - border_padding_w:, :] = 0

            output_reg[0:border_padding_h, :, :] = 0
            output_reg[output_class.shape[0] - border_padding_h:, :, :] = 0
            output_reg[:, 0:border_padding_w, :] = 0
            output_reg[:, output_class.shape[1] - border_padding_w:, :] = 0


        # --- select negative samples ---
        num_positives = np.sum(output_class[:, :, 0:-1])    # count positive samples (exclude last matrix)

        # find negative samples
        negs = output_class[:, :, -1]
        [r, c] = np.where(negs == 1)

        # select negatives to remove
        ind_to_remove = np.arange(len(r))
        np.random.shuffle(ind_to_remove)    # shuffle sample order

        num_neg = min(len(r), num_positives * num_negs_ratio)   # number of positive to negative samples ratio: 1 to 10
        num_to_remove = len(r) - num_neg
        ind_to_remove = ind_to_remove[:num_to_remove]

        # remove negatives
        for ind in ind_to_remove:
            output_class[r[ind], c[ind], :] = 0

        if num_positives > 0:
            output_class_list.append(output_class)
            output_reg_list.append(output_reg)
            valid_inds.append(img_ind)

    # output_class_list = np.array(output_class_list)
    # output_reg_list = np.array(output_reg_list)

    return output_class_list, output_reg_list, valid_inds


def plot_gt_annotations(images, bboxes, dst_path):
    """
    plot annotated ground truth rectangles onto photos
    :param images: grayscale photos [ndarray]
    :param bboxes: list of bounding box coordinates tor all images (# min_row, min_col, max_row, max_col) [list]
    :param dst_path: path of the destination folder of the annotated images [string]
    :return: None
    """

    for ind, img in enumerate(images):
        coords = bboxes[ind]  # min_row, min_col, max_row, max_col

        for rect in coords:
            cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), color=(0, 0, 0), thickness=1)

        cv2.imwrite(os.path.join(dst_path, str(ind).zfill(4) + '.bmp'), img)

from copy import deepcopy

def save_results(results_path,results_path_nms, images, output_cls, output_reg, anchor_dims, anchor_stride, prob_thr, reg_norm_coef_position_rows,reg_norm_coef_position_cols,reg_norm_coef_size_height,reg_norm_coef_size_width, thr_clustering, colors_list, flag_save_coords):
    """
    plot bounding boxes of detected objects onto test images and save as images
    :param results_path: path of destination folder [str]
    :param images: test images [ndarray]
    :param plot_color: BGR values of the color of the annotations (tuple)
    :param output_cls: output of the classifier [ndarray]
    :param output_reg: output of the regressor [ndarray]
    :param anchor_dims: tuple of tuples of anchor dimensions (height, width) [tuple]
    :param anchor_stride: stride along rows and columns [int]
    :param prob_thr: probability threshold for object classification (range: 0 to 1) [float]
    :param norm_coef: coefficient to reverse the range normalization of regressor ground truth applied before training [int]
    :param output_branch: specifies the output branch results to be saved, accepted values are 'classifier' and 'regressor' [string]
    :return: None
    """
    # binarize classifier output probabilities
    output_cls[output_cls >= prob_thr] = 1
    output_cls[output_cls < prob_thr] = 0
    num_classes=2
    output_reg[:, :, :, 0] = output_reg[:, :, :, 0] * reg_norm_coef_position_rows  # regressor output
    output_reg[:, :, :, 1] = output_reg[:, :, :, 1] * reg_norm_coef_position_cols  # regressor output
    output_reg[:, :, :, 2] = output_reg[:, :, :, 2] * reg_norm_coef_size_height  # regressor output
    output_reg[:, :, :, 3] = output_reg[:, :, :, 3] * reg_norm_coef_size_width
    # round regressor output and cast to integer pixel values
    # output_reg = np.round(output_reg * norm_coef).astype(np.int)  # regressor output, shape = (num_images, 30, 50, 12)



    # calculate location of first (top left) anchor center - start at half of stride size
    start_r = np.int(np.round(anchor_stride / 2))
    start_c = np.int(np.round(anchor_stride / 2))

    for im_ind, image in tqdm(enumerate(images)):
        valid_bboxes=[]
        image_za_nms = deepcopy(image)

        res = output_cls[im_ind, :, :, 0:-1]     # classifier output; last matrix contains non-object class

        [r, c, d] = np.where(res > 0.5)     # get coordinate of positive anchors
                                            # d contains the indices of anchor size

        for pred_ind in range(len(r)):      # iterate over positive predictions

            anchor_dim = anchor_dims[d[pred_ind]]   # get dimension of anchor
            # calculate center of anchor
            center_row = r[pred_ind] * anchor_stride + start_r
            center_col = c[pred_ind] * anchor_stride + start_c

            # 4 dimensions are fine-tuned: r, c, h, w
            delta_r = output_reg[im_ind, r[pred_ind], c[pred_ind], d[pred_ind] * 4 + 0]
            delta_c = output_reg[im_ind, r[pred_ind], c[pred_ind], d[pred_ind] * 4 + 1]
            delta_h = output_reg[im_ind, r[pred_ind], c[pred_ind], d[pred_ind] * 4 + 2]
            delta_w = output_reg[im_ind, r[pred_ind], c[pred_ind], d[pred_ind] * 4 + 3]

            # calculate top left corner of bounding box
            min_row = np.int(center_row - np.round(anchor_dim[0] / 2))
            min_col = np.int(center_col - np.round(anchor_dim[1] / 2))

            # adjust position and size with regressor predictions
            min_row_adj = np.int(min_row + delta_r)
            min_col_adj = np.int(min_col + delta_c)
            h_adj = np.int(anchor_dim[0] + delta_h)
            w_adj = np.int(anchor_dim[1] + delta_w)

            # calculate bottom right corner of the bounding box
            max_row_adj = min_row_adj + h_adj
            max_col_adj = min_col_adj + w_adj
            if (max_row_adj - min_row_adj) > 0 and (max_col_adj - min_col_adj) > 0:

                valid_bboxes.append([min_row_adj, min_col_adj, max_row_adj, max_col_adj, d[pred_ind]])

                # cv2.circle(image, (center_col, center_row), 3, color=(255, 0, 0), thickness=3)     # plot object centers
                cv2.rectangle(image, (min_col_adj, min_row_adj), (max_col_adj, max_row_adj), color=(0,0,255), thickness=1)

            # plot bounding box onto image
            # cv2.rectangle(image, (min_col_adj, min_row_adj), (max_col_adj, max_row_adj), color=plot_color, thickness=1)


        # save test image with bounding boxes of detected objects
        cv2.imwrite(os.path.join(results_path, str(im_ind).zfill(4) + '.bmp'), image)
        # helper_postprocessing.nms_tanja(image_za_nms, os.path.join(results_path_nms,str(im_ind).zfill(4) + '.bmp'),  valid_bboxes, thr_clustering, colors_list, num_classes, flag_save_coords)

from matplotlib import pyplot as plt


def get_images(im_path):
    im_files=[x for x in os.listdir(im_path)]
    print(im_files)
    images=[]
    for im_name in im_files:
        image = cv2.imread(os.path.join(im_path,im_name), 0)
        if image is None:
            continue
        # 0 for reading the image in grayscale
        image = image.reshape(image.shape[0], image.shape[1], 1)
        images.append(image)
    images=np.array(images)
    return images
