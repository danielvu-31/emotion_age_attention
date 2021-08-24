
import numpy as np


def remove_prefix(state_dict, prefix):
    '''
    Old style model is stored with all names of parameters sharing common prefix 'module.'
    '''
    print('remove prefix \'{}\''.format(prefix))
    def f(x): return x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def drop_prefix(state_dict, prefix):
    '''
    Drop some layer weight from state_dict base on prefix
    '''
    return {key: value for key, value in state_dict.items() if not key.startswith(prefix)}


def _filter_preds(max_classes,
                  max_scores,
                  boxes,
                  landms,
                  top_k_before_nms,
                  nms_threshold,
                  top_k_after_nms,
                  nms_per_class):
    '''
    Inherit this function from Review Assistant project
    '''
    def get_index(item, idx):
        return item[idx]

    # Ignore background
    inds = max_classes != 0
    boxes, landms, max_scores, max_classes = \
        map(get_index, [boxes, landms, max_scores, max_classes], [inds]*4)

    # Keep top-K before NMS
    order = max_scores.argsort()[::-1][:top_k_before_nms]
    boxes, landms, max_scores, max_classes = \
        map(get_index, [boxes, landms, max_scores, max_classes], [order]*4)

    # Do NMS
    dets = np.hstack((boxes, max_scores[:, np.newaxis])).astype(np.float32, copy=False)
    if nms_per_class:
        all_classes = np.unique(max_classes)
        all_dets = np.empty((0, 5)).astype(dets.dtype)
        all_landms = np.empty((0, 10)).astype(landms.dtype)
        all_max_classes = np.empty((0,)).astype(max_classes.dtype)
        for class_id in all_classes:
            class_inds = max_classes == class_id
            class_dets = dets[class_inds]
            class_landms = landms[class_inds]
            class_max_classes = max_classes[class_inds]
            if len(class_dets) > 1:
                keep = nms(class_dets, nms_threshold)
                class_dets = class_dets[keep]
                class_landms = class_landms[keep]
                class_max_classes = class_max_classes[keep]
            all_dets = np.concatenate((all_dets, class_dets), axis=0)
            all_landms = np.concatenate((all_landms, class_landms), axis=0)
            all_max_classes = np.concatenate((all_max_classes, class_max_classes), axis=0)
        dets = all_dets
        landms = all_landms
        max_classes = all_max_classes
        order = dets[:, -1].argsort()[::-1]
        dets = dets[order]
        landms = landms[order]
        max_classes = max_classes[order]
    else:
        if len(dets) > 1:
            keep = nms(dets, nms_threshold)
            dets = dets[keep]
            landms = landms[keep]
            max_classes = max_classes[keep]

    # Keep top-K after NMS
    dets = dets[:top_k_after_nms]
    landms = landms[:top_k_after_nms]
    max_classes = max_classes[:top_k_after_nms]

    dets = np.concatenate((dets, landms, max_classes[:, np.newaxis]), axis=1)

    return dets


def filter_preds(cls_pred,
                 loc_pred,
                 lm_pred,
                 box_scale,
                 lm_scale,
                 priors,
                 variance,
                 top_k_before_nms,
                 nms_threshold,
                 top_k_after_nms,
                 nms_per_class):
    '''
    Inherit this function from Review Assistant project
    '''
    boxes = decode_np(loc_pred, priors, variance)
    boxes *= box_scale

    landms = decode_landm_np(lm_pred, priors, variance)
    landms *= lm_scale

    # Get max score and corresponding class for each prediction (bbox)
    max_classes = np.argmax(cls_pred, axis=-1)
    max_scores = np.max(cls_pred, axis=-1) #TODO must optimize this operator

    dets = _filter_preds(max_classes=max_classes,
                         max_scores=max_scores,
                         boxes=boxes,
                         landms=landms,
                         top_k_before_nms=top_k_before_nms,
                         nms_threshold=nms_threshold,
                         top_k_after_nms=top_k_after_nms,
                         nms_per_class=nms_per_class)

    return dets

def decode_landm_np(pre, priors, variances):
    '''
    Decode landm from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        pre (ndarray): landm predictions for loc layers,
            Shape: [num_priors,10]
        priors (ndarray): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded landm predictions
    '''
    '''
    Inherit this function from Review Assistant project
    '''
    landms = np.concatenate((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                             priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                             priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                             priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                             priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                             ), axis=1)
    return landms


def decode_np(loc, priors, variances):
    '''Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (ndarray): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (ndarray): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    '''
    '''
    Inherit this function from Review Assistant project
    '''
    boxes = np.concatenate((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), axis=1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def nms(dets, thresh):
    '''Pure Python NMS baseline.'''
    '''
    Inherit this function from Review Assistant project
    '''
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
