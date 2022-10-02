"""
Authors: Elena Vasileva, Zoran Ivanovski
E-mail: elenavasileva95@gmail.com, mars@feit.ukim.edu.mk
Course: Mashinski vid, FEEIT, Spring 2021
Date: 19.03.2021

Description: function library
             training process monitoring and result statistics
Python version: 3.6
"""

# python imports
import matplotlib.pyplot as plt
import os
import numpy as np


def save_training_logs_ssd(history, dst_path):
    """
    saves graphs for the loss and accuracy of both the training and validation dataset throughout the epochs for comparison
    :param history: Keras callback object which stores accuracy information in each epoch [Keras history object]
    :param dst_path: destination for the graph images
    :return: None
    """

    # --- save combined loss graph of training and validation sets ---
    plt.figure()
    plt.plot(history.history['loss'], 'r')
    plt.plot(history.history['val_loss'], 'g')
    plt.title('Combined loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.grid()
    # plt.show()
    plt.savefig(os.path.join(dst_path, 'joint_loss.png'))
    plt.close()

    # --- save classification loss graph of training and validation sets ---
    plt.figure()
    plt.plot(history.history['rpn_out_class_loss'], 'r')
    plt.plot(history.history['val_rpn_out_class_loss'], 'g')
    plt.title('Classification loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_class', 'val_class'], loc='upper right')
    plt.grid()
    # plt.show()
    plt.savefig(os.path.join(dst_path, 'classification_loss.png'))
    plt.close()

    # --- save regression loss graph of training and validation sets ---
    plt.figure()
    plt.plot(history.history['rpn_out_regress_loss'], 'r')
    plt.plot(history.history['val_rpn_out_regress_loss'], 'g')
    plt.title('Regression loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_regr', 'val_regr'], loc='upper right')
    plt.grid()
    # plt.show()
    plt.savefig(os.path.join(dst_path, 'regression_loss.png'))
    plt.close()

    # --- save losses of training and validation sets as txt files ---
    joint_losses = np.column_stack((history.history['loss'], history.history['val_loss']))
    np.savetxt(os.path.join(dst_path, 'joint_loss.txt'), joint_losses, fmt='%.4f', delimiter='\t', header="TRAIN_LOSS\tVAL_LOSS")

    class_losses = np.column_stack((history.history['rpn_out_class_loss'], history.history['val_rpn_out_class_loss']))
    np.savetxt(os.path.join(dst_path, 'classification_loss.txt'), class_losses, fmt='%.4f', delimiter='\t', header="TRAIN_LOSS\tVAL_LOSS")

    reg_losses = np.column_stack((history.history['rpn_out_regress_loss'], history.history['val_rpn_out_regress_loss']))
    np.savetxt(os.path.join(dst_path, 'regression_loss.txt'), reg_losses, fmt='%.4f', delimiter='\t', header="TRAIN_LOSS\tVAL_LOSS")
    
def save_results(results_path, images, output_cls, output_reg, anchor_dims, anchor_stride, prob_thr, reg_norm_coef):
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

    output_reg = output_reg * reg_norm_coef  # regressor output

    # calculate location of first (top left) anchor center - start at half of stride size
    start_r = np.int(np.round(anchor_stride / 2))
    start_c = np.int(np.round(anchor_stride / 2))

    for im_ind, image in enumerate(images):

        res = output_cls[im_ind, :, :, 0:-1]  # classifier output; last matrix contains non-object class

        [r, c, d] = np.where(res > prob_thr)  # get coordinate of positive anchors
        # d contains the indices of anchor size

        for pred_ind in range(len(r)):  # iterate over positive predictions

            anchor_dim = anchor_dims[d[pred_ind]]  # get dimension of anchor
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

            # plot bounding box onto image
            if (max_row_adj - min_row_adj) > 0 and (max_col_adj - min_col_adj) > 0:
                # cv2.circle(image, (center_col, center_row), 3, color=(255, 0, 0), thickness=3)     # plot object centers
                cv2.rectangle(image, (min_col_adj, min_row_adj), (max_col_adj, max_row_adj), color=(0, 0, 255),
                              thickness=1)

        # save test image with bounding boxes of detected objects
        cv2.imwrite(os.path.join(results_path, str(im_ind).zfill(4) + '.bmp'), image)

        # helper_postprocessing.nms(image_za_nms, os.path.join(results_path_nms,str(im_ind).zfill(4) + '.bmp'),  valid_bboxes, thr_clustering, colors_list, num_classes, flag_save_coords)