"""
Authors: Elena Vasileva, Zoran Ivanovski
E-mail: elenavasileva95@gmail.com, mars@feit.ukim.edu.mk
Course: Mashinski vid, FEEIT, Spring 2021
Date: 19.03.2021

Description: design, train and evaluate a fully convolutional SSD architecture for object classification and localization
Python version: 3.6

TODO:
norm coef poseben za site kanali
partwise iou
"""

# python imports
import os
import numpy as np
from copy import deepcopy

from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import tensorflow.compat.v1 as tf 
# custom package imports
import helper_model, helper_data, helper_stats, helper_losses,helper_postprocessing
import pickle
import cv2

if __name__ == '__main__':

    # --- paths ---
    version = 'v2_baseline'
#paths for source images and annotations, model path and results path

    srcImagesPath = r'C:\Users\petar\Desktop\diogen'
    testPath=r'C:\Users\petar\Desktop\diogen\test'
    trainPath=r'C:\Users\petar\Desktop\diogen\train'
    srcAnnotationsPath = r'C:\Users\petar\Desktop\diogen\ann_topush'
    dstResultsPath = r'C:\Users\petar\Desktop\diogen\results'
    dstModelsPath = r'C:\Users\petar\Desktop\diogen\models'
    gtDstPath = r'C:\Users\petar\Desktop\diogen\gt'
    finalResPath= r'C:\Users\petar\Desktop\diogen\finalres'
    
    #destination folder for saving txt file containing the coefficient for normalization of regressor data
    file_path_reg_coef=os.path.join(dstModelsPath,version,'reg_coef')

    # --- create destination folders ---
    if not os.path.exists(os.path.join(dstResultsPath, version)):
        os.mkdir(os.path.join(dstResultsPath, version))
    # else:
    #     # to avoid overwriting training results
    #     print(f"Folder name {version} exists.")
    #     exit(1)

    resultsPath = os.path.join(dstResultsPath, version)

    if not os.path.exists(os.path.join(dstModelsPath, version)):
        os.mkdir(os.path.join(dstModelsPath, version))
    modelsPath = os.path.join(dstModelsPath, version)

    if not os.path.exists(file_path_reg_coef):
        os.mkdir(file_path_reg_coef)


    # --- variables ---
    imgDims = {'rows': 375, 'cols': 625}    # dimensions of iinput images to the CNN
    num_classes = 1     # number of object classes, negative (non-object) class is not included
    img_depth = 1    # 1 - grayscale, 3 - color

    img_dims = (imgDims['rows'], imgDims['cols'], img_depth)

    # coefficients for forming ground truth
    #NOTE da se smenat spored histogram za visini i sirini od anotaciite
    anchor_dims = [(140,40),(150,50),(180,65),(230,88),(190,75)]   # anchor dimensions (row, col)
    anchor_stride = 8
    num_negs_ratio = 3  # select X times more negative than positive samples
    #,
    # IOU thresholds for selecting positive and negative anchors
    iou_low = 0.4
    iou_high = 0.6

    # optimization hyperprameters
    epochs = 5
    lr = 0.0001     # learning rate
    batch_size = 10     # number of samples to process before updating the weights


    # --- load and format data ---
    # load full dataset into memory - image data and labels
    # x_train_orig, bboxes_train = helper_data.read_data_rpn(os.path.join(srcImagesPath, 'train'), (imgDims['cols'], imgDims['rows']), img_depth, srcAnnotationsPath, exclude_empty=True, shuffle=False)
    # x_val_orig, bboxes_val = helper_data.read_data_rpn(os.path.join(srcImagesPath, 'val'), (imgDims['cols'], imgDims['rows']), img_depth, srcAnnotationsPath, exclude_empty=True, shuffle=False)


    x_train_orig, bboxes_train = helper_data.func(os.path.join(trainPath, 'frames'), os.path.join(trainPath, 'annotations'))
    x_val_orig, bboxes_val = helper_data.func(os.path.join(testPath, 'frames'), os.path.join(testPath, 'annotations'))
    # for ind, image in enumerate(x_train_orig):
    #     annots=bboxes_train[ind]
    #     for bbox in annots:
    #         cv2.rectangle(image,(bbox[1],bbox[0]),(bbox[3],bbox[2]),color=(0,255,0))
    #     cv2.imshow('slika',image)
    #     cv2.waitKey(0)

    x_train_orig = np.array(x_train_orig)
    # x_train_orig = np.reshape(x_train_orig,x_train_orig.shape+(1,))
    x_val_orig = np.array(x_val_orig)
    # x_val_orig = np.reshape(x_val_orig,x_val_orig.shape+(1,))

    print(f'Training dataset shape: {x_train_orig.shape}')
    print(f'Number of training samples: {x_train_orig.shape[0]}')
    print(f'Number of validation samples: {x_val_orig.shape[0]}')


    # --- plot ground truth annotations ---
    gtDstPath = r'C:\Users\petar\Desktop\diogen\gt_ann'
    x_val_orig_2 = deepcopy(x_val_orig)
    #helper_data.plot_gt_annotations(x_val_orig_2, bboxes_val, gt_annotated_images_dst_path)


    # --- prepare ground truth data in required format ---
    # NOTE: images containing no objects, or objects which are not fully encased in an anchor, are discarded

    # generate ground truth output
    out_class_train, out_reg_train, valid_train = helper_data.get_anchor_data_ssd(bboxes_train, anchor_dims, img_dims, anchor_stride, iou_low, iou_high, num_negs_ratio)
    out_class_val, out_reg_val, valid_val = helper_data.get_anchor_data_ssd(bboxes_val, anchor_dims, img_dims, anchor_stride, iou_low, iou_high, num_negs_ratio)

    # normalize regression data
    # position
    out_reg_train = np.array(out_reg_train)
    out_class_train = np.array(out_class_train)
    out_class_val = np.array(out_class_val)
    out_reg_val = np.array(out_reg_val)
    print(np.sum(out_class_train[:,:,:,:-1]))


    reg_norm_coef_position_rows = np.max(np.abs(out_reg_train))
    filename = 'reg_norm_coef_position_rows.txt'
    fid1_1 = open(os.path.join(file_path_reg_coef, filename), 'wb+')
    pickle.dump(reg_norm_coef_position_rows, fid1_1)
    fid1_1.close()

    # reg_norm_coef_position_cols = np.max(np.abs(out_reg_train[:, :, :, 1]))
    # filename = 'reg_norm_coef_position_cols.txt'
    # fid1_1 = open(os.path.join(file_path_reg_coef, filename), 'wb+')
    # pickle.dump(reg_norm_coef_position_cols, fid1_1)
    # fid1_1.close()
    #
    # # size
    # reg_norm_coef_size_height = np.max(np.abs(out_reg_train[:, :, :, 2]))
    # filename = 'reg_norm_coef_size_height.txt'
    # fid1_2 = open(os.path.join(file_path_reg_coef, filename), 'wb+')
    # pickle.dump(reg_norm_coef_size_height, fid1_2)
    # fid1_2.close()
    #
    # reg_norm_coef_size_width = np.max(np.abs(out_reg_train[:, :, :, 3]))
    # filename = 'reg_norm_coef_size_width.txt'
    # fid1_2 = open(os.path.join(file_path_reg_coef, filename), 'wb+')
    # pickle.dump(reg_norm_coef_size_width, fid1_2)
    # fid1_2.close()
    prob_thr=0.5
    x_train = []
    for valid_ind in valid_train:
        x_train.append(x_train_orig[valid_ind])
    x_train = np.array(x_train)

    x_val = []
    for valid_ind in valid_val:
        x_val.append(x_val_orig[valid_ind])
    x_val = np.array(x_val)
    ground_truth_annotations_path=r'C:\Users\Koki\Desktop\LSTM\grnd_truth_annotations'
    helper_data.save_results(ground_truth_annotations_path, x_train, out_class_train, out_reg_train, anchor_dims, anchor_stride, prob_thr, reg_norm_coef_position_rows)
    print(out_class_train.shape)
    print(f'Sliki{x_train.shape}')

    x_train = np.expand_dims(x_train, -1)
    x_val= np.expand_dims(x_val, -1)
    print(f'Slikireshaped{x_train.shape}')
    # normalize training data
    # out_reg_norm = deepcopy(out_reg_train)    # NOTE: EV - ne mi teknuva oti sme troshele duplo memorija, nadole ne se koristat originalnite otkoga se premesti crtanjeto vo druga skripta
    out_reg_norm = out_reg_train

    # print(f'Max od pozicii - trening PRED: {np.max(np.abs(out_reg_norm[:, :, :, 0:2]))}')
    # print(f'Max od dimenzii - trening PRED: {np.max(np.abs(out_reg_norm[:, :, :, 2:]))}')
    # print(f'Min od pozicii - trening PRED: {np.min(np.abs(out_reg_norm[:, :, :, 0:2]))}')
    # print(f'Min od dimenzii - trening PRED: {np.min(np.abs(out_reg_norm[:, :, :, 2:]))}')

    print(f'Koeficient pozicii: {reg_norm_coef_position_rows} {reg_norm_coef_position_rows}')
    print(f'Koeficient dimenzii: {reg_norm_coef_position_rows} {reg_norm_coef_position_rows}')

    out_reg_norm[:, :, :, 0] = out_reg_train[:, :, :, 0] / reg_norm_coef_position_rows
    out_reg_norm[:, :, :, 1] = out_reg_train[:, :, :, 1] / reg_norm_coef_position_rows

    out_reg_norm[:, :, :, 2] = out_reg_train[:, :, :, 2] / reg_norm_coef_position_rows
    out_reg_norm[:, :, :, 3] = out_reg_train[:, :, :, 3] / reg_norm_coef_position_rows

    # print(f'Max od pozicii - trening: {np.max(np.abs(out_reg_norm[:, :, :, 0:2]))}')
    # print(f'Max od dimenzii - trening: {np.max(np.abs(out_reg_norm[:, :, :, 2:]))}')
    # print(f'Min od pozicii - trening: {np.min(np.abs(out_reg_norm[:, :, :, 0:2]))}')
    # print(f'Min od dimenzii - trening: {np.min(np.abs(out_reg_norm[:, :, :, 2:]))}')

    # normalize validation data
    # out_reg_val_norm = deepcopy(out_reg_val)
    out_reg_val_norm = out_reg_val

    out_reg_val_norm[:, :, :, 0] = out_reg_val[:, :, :, 0] / reg_norm_coef_position_rows
    out_reg_val_norm[:, :, :, 1] = out_reg_val[:, :, :, 1] / reg_norm_coef_position_rows

    out_reg_val_norm[:, :, :, 2] = out_reg_val[:, :, :, 2] / reg_norm_coef_position_rows
    out_reg_val_norm[:, :, :, 3] = out_reg_val[:, :, :, 3] / reg_norm_coef_position_rows


    # reg_norm_coef = np.max(np.abs(y_reg_train))
    # y_reg_train = y_reg_train / reg_norm_coef
    #
    # y_reg_val = y_reg_val / reg_norm_coef
    #
    # # save normalization coefficient
    # f = open(os.path.join(modelsPath, 'norm_coef.txt'), 'w')
    # f.write(str(reg_norm_coef))
    # f.close()

    # remove images without positive objects



    # --- plot ground truth network output ---
    x_val_2 = deepcopy(x_val)
    prob_thr = 0.5
    plot_color = (255, 255, 255)
    #path to save visualized ground truths
    ground_truth_annotations_path = r''
    output_branch = 'regressor'

    #Note: da se smeni funkcijava ??

    # helper_data.save_results(ground_truth_annotations_path, x_val_2, plot_color, out_class_val, out_reg_val, anchor_dims, anchor_stride, prob_thr, reg_norm_coef, output_branch)
    thr_clustering=0.3
    colors_list=[(0,0,255)]
    flag_save_coords=False
    # x_train=np.reshape(x_train,(x_train.shape[0],x_train[1],x_train[2],1))
    # out_class_train=np.reshape(out_class_train,(out_class_train.shape[0],out_class_train[1],out_class_train[2],1))
    # out_reg_norm=np.reshape(out_reg_norm,(out_reg_norm.shape[0],out_reg_norm[1],out_reg_norm[2],1))
    # --- construct model ---
    print(x_train.shape)
    print(out_reg_norm.shape)
    print(out_class_train.shape)
    print(out_class_val.shape)
    print(out_reg_val_norm.shape)
    # x_train=np.reshape(x_train.shape+(1,))
    # print(f'Shape after reshaping{x_train.shape}')
    epochs = 5
    with tf.device("/device:DML:0"):
        model = helper_model.construct_model_ssd(input_shape=img_dims, num_anchors=len(anchor_dims))   # build model architecture
    
        # compile model
        model.compile(loss={
            'rpn_out_class': helper_losses.rpn_loss_cls,  # loss function applied to layer named rpn_out_class
            'rpn_out_regress': helper_losses.rpn_loss_reg,
        },
            optimizer=Adam(lr=lr),
            metrics=['accuracy'])
    
        # --- fit model ---
        # save intermittent models
        model_checkpoint = ModelCheckpoint(
            filepath=os.path.join(modelsPath, 'checkpoint-{epoch:03d}-{val_loss:.4f}.hdf5'),
            # epoch number and val accuracy will be part of the weight file name
            monitor='val_loss',  # metric to monitor when selecting weight checkpoints to save
            verbose=1,
            save_best_only=False)  # True saves only the weights after epochs where the monitored value is improved
    
        history = model.fit(x_train, y={"rpn_out_class": out_class_train, "rpn_out_regress": out_reg_norm},
                            batch_size=batch_size,
                            epochs=epochs,
                            # callbacks=[model_checkpoint, early_stopping],
                            callbacks=[model_checkpoint],
                            verbose=1,
                            validation_data=(x_val, {"rpn_out_class": out_class_val, "rpn_out_regress": out_reg_val_norm}),
                            shuffle=True)

    # --- save model ---
    # save model architecture
    #NOTE da se otpakuva modelot
    print(model.summary())      # parameter info for each layer
    with open(os.path.join(modelsPath, 'modelSummary.txt'), 'w') as fh:     # save model summary
        model.summary(print_fn=lambda x: fh.write(x + '\n'))
    plot_model(model, to_file=os.path.join(modelsPath, 'modelDiagram.png'), show_shapes=True)   # save diagram of model architecture

    # save model configuration and weights
    model_json = model.to_json()  # serialize model architecture to JSON
    with open(os.path.join(os.path.join(modelsPath, 'model.json')), "w") as json_file:
        json_file.write(model_json)
    model.save_weights(os.path.join(modelsPath, 'model.h5'))  # serialize weights to HDF5
    

    # --- save training curves and logs ---
    helper_stats.save_training_logs_ssd(history=history, dst_path=modelsPath)
    boxes=helper_data.save_results(dstResultsPath, x_val, out_class_val, out_reg_val, anchor_dims, anchor_stride, prob_thr, reg_norm_coef_position_rows)
    np.shape(np.array(boxes))
    
    # selected_indices, selected_scores = tf.image.non_max_suppression_padded(
    # boxes, scores, max_output_size, iou_threshold=1.0, score_threshold=0.1,
    # soft_nms_sigma=0.5)
    # selected_boxes = tf.gather(boxes, selected_indices)
    cor=[]
    
    for ind in range(np.shape(x_val)[0]):
        new_windows=helper_postprocessing.nms_v1(boxes[ind],0)
        print(len(new_windows))
        cor.append(new_windows)
    helper_data.save_results_final(finalResPath, x_val_orig,cor)
