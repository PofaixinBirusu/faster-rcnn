import torch
import numpy as np
from torch import nn
from faster_rcnn.box import build_anchors
from faster_rcnn.box import batch_boxes_gts_iou
from faster_rcnn.box import box2point
from faster_rcnn.box import to_box
from faster_rcnn.box import two_box_iou
from faster_rcnn.box import nms
from faster_rcnn.model import VGG

CAN_USE_GPU = torch.cuda.is_available()


class RCNN(nn.Module):
    def __init__(self, n_class=3, f_w=50, f_h=37, f_s=16, scales=(8, 16, 24)):
        super(RCNN, self).__init__()
        self.net = VGG()
        self.n_class = n_class
        # rpn网络
        self.more_deep = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )
        self.cls_pred_net = nn.Conv2d(512, 18, kernel_size=1, stride=1)
        self.offset_pred_net = nn.Conv2d(512, 36, kernel_size=1, stride=1)
        # roi网络
        self.roi_cls_pred_net = nn.Linear(25088, n_class+1)
        self.roi_offset_pred_net = nn.Linear(25088, 4)

        # 冻结rpn的参数, 这个看情况，rpn训练得很好了就冻结
        for param in self.more_deep.parameters():
            param.requires_grad = False
        for param in self.cls_pred_net.parameters():
            param.requires_grad = False
        for param in self.offset_pred_net.parameters():
            param.requires_grad = False

        self.anchors = build_anchors(f_w, f_h, f_s, scalars=scales)
        self.valid_index = torch.where(
            (self.anchors[:, 0] >= 0) &
            (self.anchors[:, 1] >= 0) &
            (self.anchors[:, 2] < f_w*f_s) &
            (self.anchors[:, 3] < f_h*f_s))[0]
        print(len(self.valid_index))
        self.valid_anchor = self.anchors[self.valid_index]
        if CAN_USE_GPU:
            self.valid_anchor = self.valid_anchor.cuda()

    def predict(self, imgs):
        batch_size = imgs.shape[0]
        feature = self.net(imgs)
        out = self.more_deep(feature)
        cls_pred = self.cls_pred_net(out).permute([0, 2, 3, 1]).contiguous().view(batch_size, -1, 18).view(batch_size, -1, 2)
        offset_pred = self.offset_pred_net(out).permute(([0, 2, 3, 1])).contiguous().view(batch_size, -1, 36).view(batch_size, -1, 4)
        # size: batch_size x wh9 x 4,  batch_size x wh9 x 2
        return offset_pred, cls_pred, feature

    def loss(self, imgs, gts, cls, sample_num=256, roi_num=60, k1=400, k2=200):
        rpn_loss, rois, feature_map = self.rpn_loss(imgs, gts, sample_num, k1, k2)
        rois = torch.stack(rois, dim=0).float().cuda()
        rcnn_loss = self.roi_loss(rois, gts, cls, feature_map, roi_num=roi_num)
        return rpn_loss+rcnn_loss

    def rpn_loss(self, imgs, gts, sample_num=256, k1=400, k2=200):
        # batch_size x valid_num x gt_num
        batch_size, valid_anchor_num = gts.shape[0], len(self.valid_index)
        iou = batch_boxes_gts_iou(
            self.valid_anchor.view(1, valid_anchor_num, 4)
                .expand_as(torch.empty(batch_size, valid_anchor_num, 4)), gts)
        max_overlap, argmax_overlap = iou.max(dim=2)
        # -1代表无用的anchor, 1代表正样本，0代表负样本
        label = torch.zeros(batch_size, valid_anchor_num).fill_(-1)
        label[max_overlap < 0.3] = 0
        # gt_mask表达的是一个batch中哪些gt有用，哪些没用, 比如每张图有两个gt，一个batch3张图
        # [ [0, 1]
        #   [1, 0],
        #   [1, 1]] 这样来表示每张图上的gt哪个有用, 没用的gt就是[0, 0, 0, 0]这种填充上去的
        invalid_gt = torch.Tensor([0, 0, 0, 0])
        if CAN_USE_GPU:
            invalid_gt = invalid_gt.cuda()
        gt_mask = torch.stack(
            [torch.Tensor([0 if every_gt.eq(invalid_gt).sum().item() == 4 else 1
                           for every_gt in gt]) for gt in gts], dim=0)
        if CAN_USE_GPU:
            gt_mask = gt_mask.cuda()
            label = label.cuda()

        gt_argmax = iou.permute([0, 2, 1]).max(2)[1]
        # 与gt有最大iou的标签是正样本，其中[0, 0, 0, 0]这种无用gt不算
        for i in range(batch_size):
            label[i, gt_argmax[i][gt_mask[i] == 1]] = 1
        label[max_overlap >= 0.7] = 1
        # 每张图挑选sample_num/2个正样本和sample_num/2个负样本
        for i in range(batch_size):
            pos_num, current_pos_num = sample_num // 2, (label[i] == 1).sum().item()
            # 正样本太多了，去掉一些
            if current_pos_num > pos_num:
                discard = (label[i] == 1).nonzero().view(-1)[
                    torch.Tensor(np.random.permutation(current_pos_num)[:current_pos_num - pos_num]).long()
                ]
                label[i, discard] = -1
            neg_num, current_neg_num = sample_num - (label[i] == 1).sum().item(), (label[i] == 0).sum().item()
            # 负样本太多
            if current_neg_num > neg_num:
                discard = (label[i] == 0).nonzero().view(-1)[
                    torch.Tensor(np.random.permutation(current_neg_num)[:current_neg_num - neg_num]).long()
                ]
                label[i, discard] = -1
        offset_pred, cls_pred, feature_map = self.predict(imgs)
        loss = 0
        offset_loss_fn, cls_loss_fn = nn.SmoothL1Loss(reduction="sum"), nn.CrossEntropyLoss()
        rois = []
        for i in range(batch_size):
            print("pos_num: %d  neg_num: %d" % ((label[i] == 1).sum().item(), (label[i] == 0).sum().item()))
            # 先把正锚框选出来
            pos_anchor = self.valid_anchor[label[i] == 1]
            # 正锚框对应的偏移值 pos_num x 4
            offset_predict = offset_pred[i, self.valid_index][label[i] == 1]
            # 正锚框对应的标签   pos_num x 4
            pos_gt = gts[i, argmax_overlap[i]][label[i] == 1]
            x_a, y_a, w_a, h_a = (pos_anchor[:, 0] + pos_anchor[:, 2]) / 2, \
                                 (pos_anchor[:, 1] + pos_anchor[:, 3]) / 2, \
                                 pos_anchor[:, 2] - pos_anchor[:, 0] + 1, \
                                 pos_anchor[:, 3] - pos_anchor[:, 1] + 1
            x_gt, y_gt, w_gt, h_gt = (pos_gt[:, 0] + pos_gt[:, 2]) / 2, \
                                     (pos_gt[:, 1] + pos_gt[:, 3]) / 2, \
                                     pos_gt[:, 2] - pos_gt[:, 0] + 1, \
                                     pos_gt[:, 3] - pos_gt[:, 1] + 1
            offset_target = torch.stack([(x_gt - x_a) / w_a, (y_gt - y_a) / h_a,
                                         torch.log(w_gt / w_a), torch.log(h_gt / h_a)], dim=0).t()
            if CAN_USE_GPU:
                offset_target = offset_target.cuda()
            offset_loss = offset_loss_fn(offset_predict, offset_target) / sample_num
            # 正锚框对应的分类值 pos_num x 2
            cls_pos_predict = cls_pred[i, self.valid_index][label[i] == 1]
            # 负样本对应的分类值 neg_num x 2
            cls_neg_predict = cls_pred[i, self.valid_index][label[i] == 0]
            cls_predict = torch.cat([cls_pos_predict, cls_neg_predict], dim=0)
            # 分类真值
            cls_target = torch.cat([torch.ones(cls_pos_predict.shape[0]),
                                    torch.zeros(cls_neg_predict.shape[0])], dim=0).long()
            if CAN_USE_GPU:
                cls_target = cls_target.cuda()
            cls_loss = cls_loss_fn(cls_predict, cls_target)
            print("rpn cls loss: %.3f  offset loss: %.3f" % (cls_loss.item(), offset_loss.item()))
            loss += offset_loss + cls_loss
            # 算完了loss之后，要推荐一些区域输给下一层网络
            roi = self.rp(offset_pred[i], cls_pred[i], k1, k2)
            rois.append(roi)
        # rpn的loss计算完了，接下来算rcnn的loss
        # 你们就别占着显存了，你们已经没用了
        _, _ = iou.cpu(), label.cpu()
        # rois和feature都是下一层要用的
        return loss, rois, feature_map

    def roi_loss(self, rois, gts, cls, feature_map, roi_num):
        batch_size, k2 = rois.shape[0], rois.shape[1]
        iou = batch_boxes_gts_iou(rois, gts)
        max_overlap, argmax_overlap = iou.max(dim=2)
        # -1代表无用的anchor, 1代表正样本，0代表负样本
        label = torch.zeros(batch_size, k2).fill_(-1)
        label[max_overlap < 0.5] = 0
        label[max_overlap >= 0.5] = 1
        offset_loss_fn, cls_loss_fn = nn.SmoothL1Loss(reduction="sum"), nn.CrossEntropyLoss()
        loss = 0
        for i in range(batch_size):
            pos_num, current_pos_num = roi_num//4, (label[i] == 1).sum().item()
            # 正样本太多了，去掉一些
            if current_pos_num > pos_num:
                discard = (label[i] == 1).nonzero().view(-1)[
                    torch.Tensor(np.random.permutation(current_pos_num)[:current_pos_num-pos_num]).long()
                ]
                label[i, discard] = -1
            neg_num, current_neg_num = roi_num-(label[i] == 1).sum().item(), (label[i] == 0).sum().item()
            # 负样本太多
            if current_neg_num > neg_num:
                discard = (label[i] == 0).nonzero().view(-1)[
                    torch.Tensor(np.random.permutation(current_neg_num)[:current_neg_num-neg_num]).long()
                ]
                label[i, discard] = -1
            print("roi  pos: %d  neg: %d" % ((label[i] == 1).sum().item(), (label[i] == 0).sum().item()))
            pos_gt = gts[i, argmax_overlap[i]][label[i] == 1]
            pos_cls = cls[i, argmax_overlap[i]][label[i] == 1]
            neg_cls = (torch.zeros(size=((label[i] == 0).sum().item(), ))+self.n_class).long().cuda()
            cls_target = torch.cat([pos_cls, neg_cls], dim=0)
            pos_roi = rois[i, label[i] == 1]
            neg_roi = rois[i, label[i] == 0]
            x_a, y_a, w_a, h_a = box2point(pos_roi)
            x_gt, y_gt, w_gt, h_gt = box2point(pos_gt)
            # size: pos_roi_num x 4
            offset_target = torch.stack([(x_gt - x_a) / w_a, (y_gt - y_a) / h_a,
                                         torch.log(w_gt / w_a), torch.log(h_gt / h_a)], dim=0).t()
            f_roi = self.feature_roi(feature_map[i], torch.cat([pos_roi, neg_roi], dim=0))
            f_roi = f_roi.view(roi_num, -1)
            # print(f_roi.shape)
            cls_pred = self.roi_cls_pred_net(f_roi)
            cls_loss = cls_loss_fn(cls_pred, cls_target)
            offset_pred = self.roi_offset_pred_net(f_roi)[:(label[i] == 1).sum().item()]
            offset_loss = offset_loss_fn(offset_pred, offset_target)*3 / roi_num
            loss += offset_loss+cls_loss
            print("roi cls loss: %.3f  offset loss: %.3f" % (cls_loss.item(), offset_loss.item()))
        return loss

    def feature_roi(self, feature_map, rois):
        roi_pooling = nn.AdaptiveMaxPool2d(output_size=(7, 7))
        f_roi_s = []
        h, w = feature_map.shape[1], feature_map.shape[2]
        for roi in rois:
            roi = (roi//16).int()
            xmin, ymin, xmax, ymax = roi[0].item(), roi[1].item(), roi[2].item(), roi[3].item()
            if xmin < 0:
                xmin = 0
            if ymin < 0:
                ymin = 0
            if xmax >= w:
                xmax = w-1
            if ymax >= h:
                ymax = h-1
            f_roi = feature_map[:, ymin:(ymax+1), xmin:(xmax+1)]
            f_roi_s.append(roi_pooling(f_roi))
        return torch.stack(f_roi_s, dim=0)

    # 先选k1个分数最高的区域，nms后再选k2个最高的区域作为推荐区域
    def rp(self, offset_pred, cls_pred, k1=400, k2=200):
        offset_pred, cls_pred = offset_pred[self.valid_index], cls_pred[self.valid_index]
        cls_sorce = torch.softmax(cls_pred, dim=1)[:, 1]
        top_k_index = torch.topk(cls_sorce, k1, dim=0)[1]
        select_anchor = self.valid_anchor[top_k_index]
        select_offset = offset_pred[top_k_index]
        tx, ty, tw, th = select_offset[:, 0], select_offset[:, 1], select_offset[:, 2], select_offset[:, 3]
        x_a, y_a, w_a, h_a = box2point(select_anchor)
        x, y, w, h = tx * w_a + x_a, ty * h_a + y_a, w_a * torch.exp(tw), h_a * torch.exp(th)
        boxes_pred = torch.stack([x, y, w, h], dim=0).t()
        boxes_pred = to_box(boxes_pred).int().cpu()
        # nms 这一步在cpu里做，别搞得炸显存了
        cls_sorce = cls_sorce[top_k_index].cpu()
        label = nms(boxes_pred, cls_sorce, thresh=0.6)
        # 这些box已经按照分数排好了序的
        boxes_pred = boxes_pred[label == 0]
        # 先选k1个sorce最大的框，nms完了之后选最大的k2个
        print("after nms: "+str(len(boxes_pred)))
        return boxes_pred[:k2]

    def propose(self, img, k=10):
        offset_pred, cls_pred, feature_map = self.predict(img)
        offset_pred, cls_pred = offset_pred[0, self.valid_index], cls_pred[0, self.valid_index]
        cls_sorce = torch.softmax(cls_pred, dim=1)[:, 1]
        top_k_index = torch.topk(cls_sorce, k, dim=0)[1]
        select_anchor = self.valid_anchor[top_k_index]
        select_offset = offset_pred[top_k_index]
        tx, ty, tw, th = select_offset[:, 0], select_offset[:, 1], select_offset[:, 2], select_offset[:, 3]
        x_a, y_a, w_a, h_a = box2point(select_anchor)
        x, y, w, h = tx*w_a+x_a, ty*h_a+y_a, w_a*torch.exp(tw), h_a*torch.exp(th)
        boxes_pred = torch.stack([x, y, w, h], dim=0).t()
        boxes_pred = to_box(boxes_pred).int().cpu()
        # nms
        cls_sorce = cls_sorce[top_k_index].cpu()
        label = nms(boxes_pred, cls_sorce, thresh=0.5)
        # 把nms过的roi拿出来，塞给下一层判断细致的分类和roi的坐标回归
        rois = boxes_pred[label == 0]
        roi_num = rois.shape[0]
        f_roi = self.feature_roi(feature_map[0], rois).view(roi_num, -1)
        cls_pred = self.roi_cls_pred_net(f_roi)
        offset_pred = self.roi_offset_pred_net(f_roi)
        cls_pred, offset_pred = cls_pred.cpu(), offset_pred.cpu()
        tx, ty, tw, th = offset_pred[:, 0], offset_pred[:, 1], offset_pred[:, 2], offset_pred[:, 3]
        x_a, y_a, w_a, h_a = box2point(rois)
        x, y, w, h = tx*w_a+x_a, ty*h_a+y_a, w_a*torch.exp(tw), h_a*torch.exp(th)
        boxes_pred = to_box(torch.stack([x, y, w, h], dim=0).t())
        cls_sorce = cls_sorce[label == 0]
        cls = cls_pred.argmax(dim=1)
        boxes_pred, cls, cls_sorce = boxes_pred[cls != self.n_class], cls[cls != self.n_class], cls_sorce[cls != self.n_class]
        # 再来一遍nms
        label = nms(boxes_pred, cls_sorce, thresh=0.3)
        return boxes_pred[label == 0], cls[label == 0], cls_sorce[label == 0]
        # return rois


if __name__ == '__main__':
    x = [
        [0, 1, 1, 0, 5],
        [0, 2, 1, 0, 3],
        [0, 1, 8, 0, 7],
    ]
    x = torch.Tensor(x)
    print(x[torch.topk(x[:, 4], 3, dim=0)[1]])

    iou = [
        [
            [0.4, 0],
            [0.4, 0],
            [0.6, 0]
        ],
        [
            [0, 0.6],
            [0, 0.9],
            [0, 0.7]
        ]
    ]
    gt_mask = [
        [1, 0],
        [0, 1]
    ]
    label = [
        [0, 0, 0],
        [0, 0, 0]
    ]
    iou, gt_mask, label = torch.Tensor(iou), torch.Tensor(gt_mask), torch.Tensor(label)
    ind = iou.permute([0, 2, 1]).max(2)[1]
    batch_size = 2
    for i in range(batch_size):
        label[i, ind[i][gt_mask[i] == 1]] = 1
        print(label)

    label = [
        [1, 0, 1, 1, 1, 0, 1],
        [0, 1, 1, 0, 1, 1, 0]
    ]
    label = torch.Tensor(label)
    for i in range(batch_size):
        current_pos = (label[i] == 1).sum().item()
        print(current_pos)
        print()
        ind = (label[i] == 1).nonzero().view(-1)[torch.Tensor(np.random.permutation(current_pos)[:current_pos-3]).long()]
        label[i, ind] = 0
    print(label)
    anchors = [
        [[1, 1, 1, 1],
         [2, 2, 2, 2],
         [3, 3, 3, 3]],
        [[4, 4, 4, 4],
         [5, 5, 5, 5],
         [6, 6, 6, 6]]
    ]
    label = [
        [0, 1, -1],
        [-1, 1, 0]
    ]
    anchors, label = torch.Tensor(anchors), torch.Tensor(label)
    for i in range(batch_size):
        x = anchors[i, label[i] == 1]
        print(x)
    x = [[1, 2, 3],
         [1, 2, 3],
         [1, 2, 3]]
    y = [[1.9, 2.9, 3.9],
         [1.9, 2.9, 3.9],
         [1.9, 2.9, 3.9]]
    x, y = torch.Tensor(x), torch.Tensor(y)
    loss = nn.SmoothL1Loss(reduction="sum")
    print(loss(x, y))
    print(0.5*0.81*3)
    x, y = [1, 1, 0], [0, 0]
    x, y = torch.Tensor(x), torch.Tensor(y)
    print(torch.cat([x, y], dim=0))
    x = [
        [[[1, 2],
         [1, 2]],
        [[2, 3],
         [2, 3]]],
        [[[4, 5],
          [5, 6]],
         [[7, 8],
          [8, 9]]],
    ]
    x = torch.Tensor(x)
    roi_pooling = nn.AdaptiveMaxPool2d(output_size=(7, 7))
    print(roi_pooling(x))
