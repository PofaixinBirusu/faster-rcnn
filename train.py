import torch
from data import AutoDriveDataset
from faster_rcnn.rcnn import RCNN
from faster_rcnn.model import VGG
from torch.utils import data
from torchvision import transforms
import cv2
import numpy

CAN_USE_GPU = torch.cuda.is_available()
batch_size = 1
learning_rate = 0.0001
epoch = 100
param_path = "./param/backbone.pth"
rcnn = RCNN(scales=(1, 4, 8), f_w=37, f_h=23)
if CAN_USE_GPU:
    rcnn = rcnn.cuda()
optimizer = torch.optim.Adam(rcnn.parameters(), lr=learning_rate)
rcnn.load_state_dict(torch.load(param_path))


if __name__ == '__main__':
    root_path = "C:/Users/XR/Desktop/object-detection-crowdai/"
    label_path = "labels.csv"
    dataset = AutoDriveDataset(root_path, label_path, tw=600, th=380)
    dataloader = data.DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size)

    def single_test(img):
        rcnn.eval()
        tensor2pil = transforms.ToPILImage()
        class_name = ["Car", "Truck", "Pedestrian"]
        with torch.no_grad():
            boxes, cls, sorce = rcnn.propose(img.unsqueeze(0).cuda(), k=120)
            # boxes = rpn.propose(img.unsqueeze(0).cuda(), k=60)
            img = tensor2pil(img)
            img = cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)
            for i, box in enumerate(boxes):
                box = box.int()
                img = cv2.rectangle(img, (box[0].item(), box[1].item()),
                                         (box[2].item(), box[3].item()), color=(0, 255, 0), thickness=2)
                img = cv2.putText(img, "%s %.1f" % (class_name[cls[i]], sorce[i]),
                                  (box[0].item(), box[1].item()), cv2.FONT_HERSHEY_SIMPLEX,
                                  0.4, (0, 255, 0), 1)
            cv2.imshow("test", img)
            cv2.waitKey(0)

    for epoch_count in range(1, 1+epoch):
        # loss_val = 0
        # rcnn.train()
        # torch.cuda.empty_cache()
        # for imgs, gts, cls in dataloader:
        #     torch.cuda.empty_cache()
        #     if CAN_USE_GPU:
        #         imgs, gts, cls = imgs.cuda(), gts.cuda(), cls.cuda()
        #     loss = rcnn.loss(imgs, gts, cls, sample_num=32)
        #     print(loss)
        #     optimizer.zero_grad()
        #     torch.cuda.empty_cache()
        #     loss.backward()
        #     optimizer.step()
        #     loss_val += loss.item()
        # print("loss: %.3f" % loss_val)
        # torch.save(rcnn.state_dict(), param_path)
        for i in range(0, 32):
            single_test(dataset[i][0])