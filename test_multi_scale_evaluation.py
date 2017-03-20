# -*- coding:utf-8 -*-
import sys

sys.path.append('/home/yanglu/workspace/py-faster-rcnn-0302/caffe-fast-rcnn/python')
sys.path.append('/home/yanglu/workspace/py-faster-rcnn-0302/tools/fast_rcnn/')

import caffe
import cv2
import numpy as np
import datetime
import cPickle
from config import cfg


gpu_mode = True
gpu_id = 1
val_file = '/home/yanglu/Database/VOC_PASCAL/VOC2007_test/ImageSets/Main/test.txt'  # for voc2007 test
image_root = '/home/yanglu/Database/VOC_PASCAL/VOC2007_test/JPEGImages/'
save_root = './predict_150000_ms/'
model_deploy = 'test-resnet101.prototxt'
model_weights = 'ResNet_faster_rcnn_iter_150000.caffemodel'
prob_layer = 'prob'  # output layer, normally Softmax
num_classes = 21
# base_size = 600
# crop_size = 473
mean_value = np.array([103.939, 116.779, 123.68])
# mean_value = np.array([[[102.9801, 115.9465, 122.7717]]])
# scale_array = [1.0]  # single scale

# test_scales = [600]
test_scales = [200, 400, 600, 800, 1000]
# scale_array = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]  # multi scale
cfg.TEST.HAS_RPN = True

_classes = ('__background__',  # always index 0
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor')

if gpu_mode:
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
else:
    caffe.set_mode_cpu()

t1 = datetime.datetime.now()
net = caffe.Net(model_deploy, model_weights, caffe.TEST)
t2 = datetime.datetime.now()
print 'load model:', t2 - t1

def eval_batch(max_per_image=100, thresh=0.05):
    eval_images = []
    f = open(val_file, 'r')
    for i in f:
        eval_images.append(i.strip())

    skip_num = 0
    eval_len = len(eval_images)

    all_boxes = [[[] for _ in xrange(eval_len)]
                 for _ in xrange(num_classes)]

    start_time = datetime.datetime.now()
    for i in xrange(eval_len - skip_num):
        _img = cv2.imread(image_root + eval_images[i + skip_num] + '.jpg')
        print eval_images[i]
        scores, boxes = caffe_detect_process(net, _img)
        '''test'''
        if i > 1:
            break

        for j in xrange(1, num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j * 4:(j + 1) * 4]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            keep = nms(cls_dets, thresh=0.3)
            cls_dets = cls_dets[keep, :]
            all_boxes[j][i] = cls_dets

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in xrange(1, num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        print 'Testing image: ' + str(i + 1) + '/' + str(eval_len) + '  ' + str(eval_images[i + skip_num])


    det_file = ('./detections.pkl')
    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    for cls_ind, cls in enumerate(_classes):
        if cls == '__background__':
            continue
        print 'Writing {} VOC results file'.format(cls)
        filename = save_root + 'comp4' + '_det' + '_test_' + cls + '.txt'
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(eval_images):
                dets = all_boxes[cls_ind][im_ind]
                if dets == []:
                    continue
                for k in xrange(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index, dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))

    end_time = datetime.datetime.now()
    print '\nEvaluation process ends at: {}. \nTime cost is: {}. '.format(str(end_time), str(end_time - start_time))

def bbox_transform_inv(boxes, deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes

def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """

    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes

def nms(dets, thresh):
    if dets.shape[0] == 0:
        return []
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def caffe_detect_process(net, _input):
    im_orig = _input.astype(np.float32, copy=True)
    im_orig -= mean_value

    # h, w ,d = _input.shape
    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    scaled_ims = []
    scale_array = []

    # for j in scale_array:
    #     long_size = float(base_size) * j + 1
    #     ratio = long_size / max(h, w)
    #     new_size = (int(w * ratio), int(h * ratio))
    #     _scale = cv2.resize(im_orig, new_size)
    #     # new_size = (int(w * scale_array[j]), int(h * scale_array[j]))
    #     # _scale = cv2.resize(im_orig, new_size)
    #     # _scale = cv2.resize(im_orig, None, None, fx=scale_array[j], fy=scale_array[j],
    #     #                 interpolation=cv2.INTER_LINEAR)
    #     scaled_ims.append(_scale)
    for target_size in test_scales:
        im_scale = float(target_size) / float(im_size_min)
        if np.round(im_scale * im_size_max) > 1000:
            im_scale = float(1000) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        scaled_ims.append(im)
        scale_array.append(im_scale)

    # max_shape = np.array([im.shape for im in scaled_ims]).max(axis=0)
    num_images = len(scaled_ims)
    # blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
    #                 dtype=np.float32)

    blob_scale_list = []
    scores_scale_list = []
    pred_boxes_scale_list = []

    channel_swap = (0, 3, 1, 2)
    for i in xrange(num_images):
        im = scaled_ims[i]
        blob = np.zeros((1, im.shape[0], im.shape[1], 3), dtype=np.float32)

        # blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
        blob[:, 0:im.shape[0], 0:im.shape[1], :] = im
        blob= blob.transpose(channel_swap)
        blob_scale_list.append(blob)

    # channel_swap = (0, 3, 1, 2)
    # blob = blob.transpose(channel_swap)

    blobs = {'data' : None, 'rois' : None}

    for i in xrange(num_images):
        blobs['data'] = blob_scale_list[i]
    # if not cfg.TEST.HAS_RPN:
    #     blobs['rois'] = _get_rois_blob(rois)

        if cfg.TEST.HAS_RPN:
            im_blob = blobs['data']
            blobs['im_info'] = np.array(
                [[im_blob.shape[2], im_blob.shape[3], scale_array[i]]],
                dtype=np.float32)
        net.blobs['data'].reshape(*(blobs['data'].shape))
        if cfg.TEST.HAS_RPN:
            net.blobs['im_info'].reshape(*(blobs['im_info'].shape))

        # do forward
        forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
        if cfg.TEST.HAS_RPN:
            forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)

        blobs_out = net.forward(**forward_kwargs)

        if cfg.TEST.HAS_RPN:
            # assert len(scale_array) == 1, "Only single-image batch implemented"
            rois = net.blobs['rois'].data.copy()
            # unscale back to raw image space
            boxes = rois[:, 1:5] / scale_array[i]

        '''test'''
        print 'rois is :'
        print rois
        print rois.shape

        print 'rois boxes is :'
        print boxes
        print boxes.shape

        scores = blobs_out['cls_prob']
        # Apply bounding-box regression deltas
        box_deltas = blobs_out['bbox_pred']
        '''test'''
        print 'box_deltas is :'
        print box_deltas
        print box_deltas.shape

        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, _input.shape)

        '''test'''
        print 'pred_boxes is :'
        print pred_boxes
        print pred_boxes.shape

        scores_scale_list.append(scores)
        pred_boxes_scale_list.append(pred_boxes)

    _pred_boxes = reduce(lambda x1, x2,: np.row_stack((x1, x2)),pred_boxes_scale_list)
    _scores = reduce(lambda x1, x2,: np.row_stack((x1, x2)), scores_scale_list)

    return _scores, _pred_boxes


if __name__ == '__main__':
    eval_batch(max_per_image=100, thresh=0.05)