
from notebooks import visualization

from preprocessing import ssd_vgg_preprocessing

from nets import ssd_vgg_300, ssd_common, np_methods

import matplotlib.image as mpimg

import matplotlib.pyplot as plt

import os

import math

import random

import numpy as np

import tensorflow as tf

import cv2



slim = tf.contrib.slim



gpu_options = tf.GPUOptions(allow_growth=True)


config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)


isess = tf.InteractiveSession(config=config)



net_shape = (300, 300)


data_format = 'NHWC'


img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))



image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(


    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)


image_4d = tf.expand_dims(image_pre, 0)



reuse = True if 'ssd_net' in locals() else None


ssd_net = ssd_vgg_300.SSDNet()


with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):


    predictions, localisations, _, _ = ssd_net.net(

        image_4d, is_training=False, reuse=reuse)



ckpt_filename = './checkpoints/ssd_300_vgg.ckpt'



isess.run(tf.global_variables_initializer())


saver = tf.train.Saver()


saver.restore(isess, ckpt_filename)



ssd_anchors = ssd_net.anchors(net_shape)



def process_image(img, select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300)):


    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],

                                                              feed_dict={img_input: img})


    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(

        rpredictions, rlocalisations, ssd_anchors,

        select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)

    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)


    rclasses, rscores, rbboxes = np_methods.bboxes_sort(

        rclasses, rscores, rbboxes, top_k=400)


    rclasses, rscores, rbboxes = np_methods.bboxes_nms(

        rclasses, rscores, rbboxes, nms_threshold=nms_threshold)


    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)


    return rclasses, rscores, rbboxes


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    image_path = './a.jpg' # 图片路径


    img = mpimg.imread(image_path)


    rclasses, rscores, rbboxes = process_image(img)  # 这里传入图片


    labeled_img = visualization.bboxes_draw_on_img(

        img, rclasses, rscores, rbboxes, visualization.colors_plasma)  # 返回标注图片

    plt.imshow(labeled_img)
    plt.show()

    # visualization.plt_bboxes(img, rclasses, rscores, rbboxes)  # 展示（plt）标注图片
