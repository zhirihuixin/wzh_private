# -*- coding:utf-8 -*-
import sys

sys.path.append('/home/yanglu/workspace/py-faster-rcnn-0302/caffe-fast-rcnn/python')
sys.path.append('/home/yanglu/workspace/py-faster-rcnn-0302/tools/fast_rcnn/')
import evaluation_det_ms as eval
import datetime
import numpy as np
import matplotlib.pyplot as plt
import caffe
import cv2
from config import cfg

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'resnet101_ohem', 'tvmonitor')


color_map = {0: [0, 0, 0], 1: [0, 0, 128], 2: [0, 128, 0], 3: [0, 128, 128], 4: [128, 0, 0], 5: [128, 0, 128],
             6: [128, 128, 0], 7: [128, 128, 128], 8: [0, 0, 64], 9: [0, 0, 192], 10: [0, 128, 64],
             11: [0, 128, 192], 12: [128, 0, 64], 13: [128, 0, 192], 14: [128, 128, 64], 15: [128, 128, 192],
             16: [0, 64, 0], 17: [0, 64, 128], 18: [0, 192, 0], 19: [0, 192, 128], 20: [128, 64, 0],
             255: [255, 255, 255]}

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

if gpu_mode:
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
else:
    caffe.set_mode_cpu()

t1 = datetime.datetime.now()
net = caffe.Net(model_deploy, model_weights, caffe.TEST)
t2 = datetime.datetime.now()
print 'load model:', t2 - t1


def boxes_filter(class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    _objs = []
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return _objs

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        _objs.append(dict(bbox=bbox, score=score, class_name=class_name))

    return _objs

def detector(_net, im_path):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im = cv2.imread(im_path)

    # Detect all object classes and regress object bounds
    scores, boxes = eval.caffe_detect_process(net, im)

    CONF_THRESH = 0.8
    objs = []
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4:8]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = eval.nms(dets, 0.3)
        dets = dets[keep, :]
        _obj = boxes_filter(cls, dets, thresh=CONF_THRESH)
        objs.extend(_obj)

    im = im[:, :, (2, 1, 0)]
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    currentAxis = plt.gca()
    for i in objs:
        print i
        color = colors[CLASSES.index(i['class_name'])]
        display_txt = '%s: %.2f' % (i['class_name'], i['score'])
        coords = (int(i['bbox'][0]), int(i['bbox'][1])), \
                 int(i['bbox'][2] - i['bbox'][0]) + 1, \
                 int(i['bbox'][3] - i['bbox'][1]) + 1
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=3))
        currentAxis.text(int(i['bbox'][0]), int(i['bbox'][1]), display_txt, bbox={'facecolor': color, 'alpha': 0.5})
    plt.imshow(im)
    plt.title('detection')
    plt.savefig('./save_new.jpg', dpi=80)
    plt.show()


if __name__ == '__main__':


    # Warmup on a dummy image
    # im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    # for i in xrange(2):
    #     _, _ = im_detect(net, im)
    #
    detector(net, '000123.jpg')