# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an imdb (image database)."""

from fast_rcnn.config import cfg, get_output_dir
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
import argparse
from utils.timer import Timer
import numpy as np
import cv2
import caffe
from fast_rcnn.nms_wrapper import nms
import cPickle
from utils.blob import im_list_to_blob
import os

def _get_image_blob(im, y):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS
    im_orig /= cfg.STD

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    target_size = cfg.TEST.SCALES[y]
    im_scale = float(target_size) / float(im_size_min)

    if len(cfg.TEST.MAX_SIZE) == len(cfg.TEST.SCALES):
        _max_size = cfg.TEST.MAX_SIZE[y]
    else:
        _max_size = cfg.TEST.MAX_SIZE[0]
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > _max_size:
        im_scale = float(_max_size) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)

def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
    """
    rois, levels = _project_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)

def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (list): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1

        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels

def _get_blobs(im, rois, y):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None, 'rois' : None}
    blobs['data'], im_scale_factors = _get_image_blob(im, y)
    if not cfg.TEST.HAS_RPN:
        blobs['rois'] = _get_rois_blob(rois, im_scale_factors)
    return blobs, im_scale_factors

def im_detect(net, im, y, boxes=None):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals or None (for RPN)

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """
    blobs, im_scales = _get_blobs(im, boxes, y)

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(hashes, return_index=True,
                                        return_inverse=True)
        blobs['rois'] = blobs['rois'][index, :]
        boxes = boxes[index, :]

    if cfg.TEST.HAS_RPN:
        im_blob = blobs['data']
        blobs['im_info'] = np.array(
            [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
            dtype=np.float32)

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    if cfg.TEST.HAS_RPN:
        net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
    else:
        net.blobs['rois'].reshape(*(blobs['rois'].shape))

    # do forward
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    if cfg.TEST.HAS_RPN:
        forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)
    else:
        forward_kwargs['rois'] = blobs['rois'].astype(np.float32, copy=False)
    blobs_out = net.forward(**forward_kwargs)

    if cfg.TEST.HAS_RPN:
        assert len(im_scales) == 1, "Only single-image batch implemented"
        rois = net.blobs['rois'].data.copy()
        # unscale back to raw image space
        boxes = rois[:, 1:5] / im_scales[0]

    if cfg.TEST.SVM:
        # use the raw scores before softmax under the assumption they
        # were trained as linear SVMs
        scores = net.blobs['cls_score'].data
    else:
        # use softmax estimated probabilities
        scores = blobs_out['cls_prob']

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = blobs_out['bbox_pred']
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, im.shape)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        # Map scores and predictions back to the original set of boxes
        scores = scores[inv_index, :]
        pred_boxes = pred_boxes[inv_index, :]

    if cfg.TEST.ITER_BBOX_REG:
        sco = np.reshape(scores, (21*len(scores), 1))

        x1 = pred_boxes[:, 0::4]
        y1 = pred_boxes[:, 1::4]
        x2 = pred_boxes[:, 2::4]
        y2 = pred_boxes[:, 3::4]

        x11 = np.reshape(x1, (21*len(x1), 1))
        y11 = np.reshape(y1, (21*len(y1), 1))
        x22 = np.reshape(x2, (21*len(x2), 1))
        y22 = np.reshape(y2, (21*len(y2), 1))

        dets_temp = np.hstack((x11, y11, x22, y22, sco)).astype(np.float32, copy=False)

        image_scores = dets_temp[:, -1]
        if len(image_scores) > cfg.TEST.RPN_POST_NMS_TOP_N:
            image_thresh = np.sort(image_scores)[-cfg.TEST.RPN_POST_NMS_TOP_N]
            for j in xrange(len(dets_temp)):
                keep = np.where(dets_temp[:, -1] >= image_thresh)[0]
                dets_temp = dets_temp[keep, :]
        boxes_temp = dets_temp[:, 0:4] * im_scales[0]
        im_scales_temp = np.tile(im_scales[0], (len(boxes_temp), 1))
        box_roi = np.hstack((im_scales_temp, boxes_temp)).astype(np.float32, copy=False)
        blobs['rois'] = box_roi.astype(np.float32, copy=False)

        # reshape network inputs
        net.blobs['data'].reshape(*(blobs['data'].shape))
        net.blobs['rois'].reshape(*(blobs['rois'].shape))

        # do forward
        forward_kwargs_1 = {'data': blobs['data'].astype(np.float32, copy=False)}
        forward_kwargs_1['rois'] = blobs['rois'].astype(np.float32, copy=False)
        blobs_out = net.forward(**forward_kwargs_1)

        rois = net.blobs['rois'].data.copy()
        # unscale back to raw image space
        boxes = rois[:, 1:5] / im_scales[0]
        scores = blobs_out['cls_prob']

        box_deltas = blobs_out['bbox_pred']
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, im.shape)

    return scores, pred_boxes

def vis_detections(im, class_name, dets, thresh=0.3):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt
    im = im[:, :, (2, 1, 0)]
    for i in xrange(np.minimum(10, dets.shape[0])):
        bbox = dets[i, :4]
        score = dets[i, -1]
        if score > thresh:
            plt.cla()
            plt.imshow(im)
            plt.gca().add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='g', linewidth=3)
                )
            plt.title('{}  {:.3f}'.format(class_name, score))
            plt.show()

def apply_nms(all_boxes, thresh):
    """Apply non-maximum suppression to all predicted boxes output by the
    test_net method.
    """
    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    nms_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(num_classes)]
    for cls_ind in xrange(num_classes):
        for im_ind in xrange(num_images):
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue
            # CPU NMS is much faster than GPU NMS when the number of boxes
            # is relative small (e.g., < 10k)
            # TODO(rbg): autotune NMS dispatch
            keep = nms(dets, thresh, force_cpu=True)
            if len(keep) == 0:
                continue
            nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
    return nms_boxes

def test_net(net, imdb, max_per_image=100, thresh=0.05, vis=False):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    output_dir = get_output_dir(imdb, net)

    # timers
    _t = {'im_detect' : Timer(), 'im_detect_1' : Timer(), 'misc' : Timer()}

    if not cfg.TEST.HAS_RPN:
        roidb = imdb.roidb

    for i in xrange(num_images):
    #for i in xrange(1):
        # filter out any ground truth boxes
        if cfg.TEST.HAS_RPN:
            box_proposals = None
        else:
            # The roidb may contain ground-truth rois (for example, if the roidb
            # comes from the training or val split). We only want to evaluate
            # detection on the *non*-ground-truth rois. We select those the rois
            # that have the gt_classes field set to 0, which means there's no
            # ground truth.
            box_proposals = roidb[i]['boxes'][roidb[i]['gt_classes'] == 0]

        im = cv2.imread(imdb.image_path_at(i))
        if len(cfg.TEST.SCALES) > 1:
            scores_temp = [0 for _ in xrange(len(cfg.TEST.SCALES))]
            boxes_temp = [0 for _ in xrange(len(cfg.TEST.SCALES))]
            if cfg.TEST.IMAGE_FLIP:
                im_flip = im[:, ::-1]
                width = im.shape[1]
                scores_flip = [0 for _ in xrange(len(cfg.TEST.SCALES))]
                boxes_flip = [0 for _ in xrange(len(cfg.TEST.SCALES))]
            _t['im_detect'].tic()
            for y in xrange(len(cfg.TEST.SCALES)):
                _t['im_detect_1'].tic()
                scores, boxes = im_detect(net, im, y, box_proposals)
                scores_temp[y] = scores.copy()
                boxes_temp[y] = boxes.copy()
                if cfg.TEST.IMAGE_FLIP:
                    scores, boxes = im_detect(net, im_flip, y, box_proposals)
                    scores_flip[y] = scores.copy()
                    boxes_flip[y] = boxes.copy()
                _t['im_detect_1'].toc()
            _t['im_detect'].toc()
            _t['misc'].tic()
            # skip j = 0, because it's the background class
            for j in xrange(1, imdb.num_classes):
                for y in xrange(len(cfg.TEST.SCALES)):
                    inds = np.where(scores_temp[y][:, j] > thresh)[0]
                    cls_scores = scores_temp[y][inds, j]
                    cls_boxes = boxes_temp[y][inds, j*4:(j+1)*4]
                    cls_dets_temp = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                        .astype(np.float32, copy=False)
                    if cfg.TEST.IMAGE_FLIP:
                        cls_dets_flip_temp = cls_dets_temp.copy()
                        inds = np.where(scores_flip[y][:, j] > thresh)[0]
                        cls_scores = scores_flip[y][inds, j]
                        cls_boxes = boxes_flip[y][inds, j*4:(j+1)*4]
                        cls_dets_flip = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                            .astype(np.float32, copy=False)
                        oldx1 = cls_dets_flip[:, 0].copy()
                        oldx2 = cls_dets_flip[:, 2].copy()
                        cls_dets_flip[:, 0] = width - oldx2 - 1
                        cls_dets_flip[:, 2] = width - oldx1 - 1
                        cls_dets_temp = np.append(cls_dets_flip, cls_dets_flip_temp, axis=0)
                    if y == 0:
                        cls_dets = cls_dets_temp.copy()
                    else:
                        cls_dets = np.append(cls_dets, cls_dets_temp.copy(), axis=0)
                keep = nms(cls_dets, cfg.TEST.NMS)
                cls_dets = cls_dets[keep, :]
                if vis:
                    vis_detections(im, imdb.classes[j], cls_dets)
                all_boxes[j][i] = cls_dets

        else:              
            _t['im_detect'].tic()
            scores, boxes = im_detect(net, im, 0, box_proposals)
            if cfg.TEST.IMAGE_FLIP:
                im_flip = im[:, ::-1]
                width = im.shape[1]
                scores_temp = scores.copy()
                boxes_temp = boxes.copy()
                scores, boxes = im_detect(net, im_flip, 0, box_proposals)
            _t['im_detect'].toc()

            _t['misc'].tic()
            # skip j = 0, because it's the background class
            for j in xrange(1, imdb.num_classes):
                inds = np.where(scores[:, j] > thresh)[0]
                cls_scores = scores[inds, j]
                cls_boxes = boxes[inds, j*4:(j+1)*4]
                cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                    .astype(np.float32, copy=False)
                if cfg.TEST.IMAGE_FLIP:
                    cls_dets_flip = cls_dets.copy()
                    inds = np.where(scores_temp[:, j] > thresh)[0]
                    cls_scores = scores_temp[inds, j]
                    cls_boxes = boxes_temp[inds, j*4:(j+1)*4]
                    cls_dets_temp = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                        .astype(np.float32, copy=False)
                    oldx1 = cls_dets_flip[:, 0].copy()
                    oldx2 = cls_dets_flip[:, 2].copy()
                    cls_dets_flip[:, 0] = width - oldx2 - 1
                    cls_dets_flip[:, 2] = width - oldx1 - 1
                    cls_dets = np.append(cls_dets_temp, cls_dets_flip, axis=0)
                keep = nms(cls_dets, cfg.TEST.NMS)
                cls_dets = cls_dets[keep, :]
                if vis:
                    vis_detections(im, imdb.classes[j], cls_dets)
                all_boxes[j][i] = cls_dets
            
        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in xrange(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]
        _t['misc'].toc()

        if len(cfg.TEST.SCALES) > 1:
            print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s {:.3f}s' \
                  .format(i + 1, num_images, _t['im_detect'].average_time, 
                    _t['im_detect_1'].average_time, _t['misc'].average_time)
        else:
            print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
                  .format(i + 1, num_images, _t['im_detect'].average_time,
                      _t['misc'].average_time)

    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    print 'Evaluating detections'
    imdb.evaluate_detections(all_boxes, output_dir)
