from torch.utils import data
import pandas as pd
from faster_rcnn.util import imread
from faster_rcnn.util import cv2pil
from torchvision import transforms
import torch
import cv2


class AutoDriveDataset(data.Dataset):
    def __init__(self, root_path, label_path, tw=800, th=600):
        super(AutoDriveDataset, self).__init__()
        self.img_list = []
        self.tw, self.th = tw, th
        df = pd.read_csv(root_path+label_path, usecols=['xmin', 'ymin', "xmax", "ymax", "Label", "Frame"])
        class_name = ["Car", "Truck", "Pedestrian"]
        class2index = {class_name: i for i, class_name in enumerate(class_name)}

        class ImgInfo:
            def __init__(self, path, gt, cls):
                self.path, self.gt, self.cls = path, gt, cls

        img_info = {}
        print(df.head())
        print("加载数据集...")
        # for i in range(len(df)):
        for i in range(512):
            xmin, ymin, xmax, ymax, img_path, label = \
                df["xmin"][i], df["xmax"][i], df["ymin"][i], df["ymax"][i], df["Frame"][i], df["Label"][i]
            if img_info.get(img_path) is None:
                img_info[img_path] = [[], []]
            img_info[img_path][0].append([int(xmin), int(ymin), int(xmax), int(ymax)])
            img_info[img_path][1].append(class2index[label])
        for img_name in img_info.keys():
            self.img_list.append(ImgInfo(path=root_path+img_name,
                                         gt=img_info[img_name][0], cls=img_info[img_name][1]))
        print(len(self.img_list))
        self.img_list = self.img_list[:32]
        self.max_gt_num = 0
        for img_info in self.img_list:
            if self.max_gt_num < len(img_info.gt):
                self.max_gt_num = len(img_info.gt)
        print(self.max_gt_num)

        print("加载完毕")

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_info = self.img_list[index]
        old_w, old_h = 1920, 1200
        img, new_w, new_h = imread(img_info.path, tw=self.tw, th=self.th)
        offset_x, offset_y = (self.tw-new_w)//2, (self.th-new_h)//2
        gts = [[int(gt[0]/old_w*new_w+offset_x), int(gt[1]/old_h*new_h+offset_y),
                int(gt[2]/old_w*new_w+offset_x), int(gt[3]/old_h*new_h+offset_y)] for gt in img_info.gt]
        cls = img_info.cls
        while len(gts) < self.max_gt_num:
            gts.append([0, 0, 0, 0])
            cls.append(-1)
        # for gt in gts:
        #     img = cv2.rectangle(img, (gt[0], gt[1]), (gt[2], gt[3]), color=(0, 255, 0), thickness=2)
        # cv2.imshow("test", img)
        # cv2.waitKey(0)
        img2tensor = transforms.ToTensor()
        img = img2tensor(cv2pil(img))
        return img, torch.Tensor(gts), torch.Tensor(cls).long()


if __name__ == '__main__':
    root_path = "C:/Users/XR/Desktop/object-detection-crowdai/"
    label_path = "labels.csv"
    dataset = AutoDriveDataset(root_path, label_path, tw=600, th=380)
    print(len(dataset))
    for i in range(len(dataset)):
        x = dataset[i]
    # from faster_rcnn.rpn import RPN
    # rpn = RPN(None, scales=(4, 8, 16))
    # for i in range(len(dataset)):
    #     img, gts = dataset[i]
    #     gts = gts.cuda()
    #     boxes = rpn.loss(gts.unsqueeze(0), 32).int()
    #     for box in boxes:
    #         img = cv2.rectangle(img, (box[0].item(), box[1].item()), (box[2].item(), box[3].item()), color=(0, 0, 255), thickness=2)
    #     cv2.imshow("test", img)
    #     cv2.waitKey(0)