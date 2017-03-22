import pickle
import numpy as np

pkl_file = open('D:\wangzhihui\\123\wzh\ResNext101_32_4d_faster_rcnn_merge_bn_scale_iter_60000\\detections.pkl', 'rb')
data1 = pickle.load(pkl_file)

def py_cpu_nms(dets, thresh):
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

print type(data1)
print type(data1[1])
print len(data1[1])
aa = data1[1][26]
print aa

for i in xrange(4952):
    if len(data1[1][i]) > 0:
        aa = np.append(aa, data1[1][i], axis=0)
print len(aa)
keep = py_cpu_nms(aa, 0.3)
bb = aa[keep, :]
print len(bb)
print bb[45]

aa = np.append(aa, aa, axis=0)
aa = np.append(aa, aa, axis=0)
print len(aa)

keep = py_cpu_nms(aa, 0.3)
cc = aa[keep, :]
print len(cc)
print cc[45]

#################################################################################
import numpy as np
a = [[1, 2, 4, 5], [6, 8, 1, 0]]
b = [[1, 2, 4, 3], [6, 8, 1, 2]]
c = [[1, 2, 4, 9], [6, 8, 1, 5]]
aa = np.array(a)
print aa[:, 0::2]
aa = np.append(a, b, axis=0)
aa = np.append(aa, c, axis=0)
bb = aa[:, 0:3]
inds = np.where(aa[:, 1] > 1)[0]
print inds
cls_scores = aa[inds, 3]
print cls_scores
print aa
print bb
print bb[:, np.newaxis]
cls = np.hstack((aa, bb)).astype(np.float32, copy=False)
print cls
