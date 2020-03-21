import torch
import math
import numpy as np

CAN_USE_GPU = torch.cuda.is_available()


def base_anchor(w, ratios=(0.5, 1, 2)):
    return [((w-1)/2, (w-1)/2, w/math.sqrt(ratio), w*math.sqrt(ratio)) for ratio in ratios]

# 把一批x, y, w, h的box转成xmin, ymin, xmax, ymax的box
def to_box(boxes):
    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return torch.round(torch.stack([x-(w-1)/2, y-(h-1)/2, x+(w-1)/2, y+(h-1)/2], dim=0).t())


def build_anchors(f_w, f_h, f_s, ratios=(0.5, 1, 2), scalars=(8, 16, 24)):
    base = torch.Tensor(base_anchor(f_s, ratios=ratios))
    base_expand = to_box(torch.cat([torch.cat([base[:, 0:2], base[:, 2:4]*scalar], dim=1) for scalar in scalars], 0))
    shift_x, shift_y = np.meshgrid(np.array(range(f_w))*f_s, np.array(range(f_h))*f_s)
    shifts = torch.Tensor(np.vstack([shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel()]).transpose())
    return torch.cat([base_expand+shift.view(1, 4) for shift in shifts], dim=0)


def batch_iou(boxes1, boxes2):
    x1, y1, x2, y2 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    x1_, y1_, x2_, y2_ = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]
    lx, ly = torch.stack([x1, x1_], 0).max(dim=0)[0], torch.stack([y1, y1_], 0).max(dim=0)[0]
    rx, ry = torch.stack([x2, x2_], 0).min(dim=0)[0], torch.stack([y2, y2_], 0).min(dim=0)[0]
    mask = ((lx < rx) & (ly < ry)).int()
    jiao_ji = (ry-ly+1)*(rx-lx+1)
    bing_ji = (x2-x1+1)*(y2-y1+1)+(x2_-x1_+1)*(y2_-y1_+1)-jiao_ji
    bing_ji[bing_ji == 0] = 1e-5
    return (jiao_ji/bing_ji)*mask


# 返回与一个batch的box与gts的iou, 假设一批gts有k个gt，anchors_num个box，那返回一个batch x anchor_num x k的iou矩阵
def batch_boxes_gts_iou(boxes, gts):
    batch_size, box_num, gt_num = gts.shape[0], boxes.shape[1], gts.shape[1]
    return batch_iou(
        boxes.contiguous().view(batch_size, box_num, 1, 4)
            .expand_as(torch.empty(size=(batch_size, box_num, gt_num, 4)))
            .contiguous().view(-1, 4),
        gts.view(batch_size, 1, gt_num, 4)
            .expand_as(torch.empty(size=(batch_size, box_num, gt_num, 4)))
            .contiguous().view(-1, 4)
    ).view(batch_size, box_num, gt_num)


def box2point(boxes):
    x, y, w, h = (boxes[:, 0]+boxes[:, 2])/2, (boxes[:, 1]+boxes[:, 3])/2, boxes[:, 2]-boxes[:, 0]+1, boxes[:, 3]-boxes[:, 1]+1
    return x, y, w, h


def two_box_iou(box1, box2):
    x1, y1, x2, y2 = box1[0].item(), box1[1].item(), box1[2].item(), box1[3].item()
    x1_, y1_, x2_, y2_ = box2[0].item(), box2[1].item(), box2[2].item(), box2[3].item()
    lx, ly = max(x1, x1_), max(y1, y1_)
    rx, ry = min(x2, x2_), min(y2, y2_)
    if lx < rx and ly < ry:
        jiao_ji = (ry - ly + 1) * (rx - lx + 1)
        bing_ji = (x2 - x1 + 1) * (y2 - y1 + 1) + (x2_ - x1_ + 1) * (y2_ - y1_ + 1) - jiao_ji
        return jiao_ji / bing_ji
    else:
        return 0


def nms(boxes, sorce, thresh=0.5):
    label = torch.ones(size=(len(boxes),))
    iou = [[two_box_iou(anchor, base) for anchor in boxes] for base in boxes]
    while (label == 1).sum() > 0:
        v, index = (sorce * label).topk(1, dim=0)
        label[index.item()] = 0
        for i in range(boxes.shape[0]):
            if i != index.item() and iou[index.item()][i] > thresh:
                label[i] = -1
    # label == 0就是选出来的box
    return label


if __name__ == '__main__':
    boxes = [
        [[0, 0, 15, 15],
         [0, 0, 16, 16],
         [1, 1, 16, 16]],
        [[0, 0, 16, 16],
         [1, 1, 16, 16],
         [0, 0, 15, 15]],
    ]
    gts = [
        [[0, 0, 15, 15],
         [0, 0, 15, 15]],
        [[0, 0, 15, 15],
         [1, 1, 16, 16]]
    ]
    boxes, gts = torch.Tensor(boxes), torch.Tensor(gts)
    print(batch_boxes_gts_iou(boxes, gts))