# faster-rcnn复现
### 效果图
[1]:https://github.com/PofaixinBirusu/faster-rcnn/blob/master/show-images/1.png
[2]:https://github.com/PofaixinBirusu/faster-rcnn/blob/master/show-images/2.png
[3]:https://github.com/PofaixinBirusu/faster-rcnn/blob/master/show-images/3.png
[4]:https://github.com/PofaixinBirusu/faster-rcnn/blob/master/show-images/4.png

### 训练
训练是把rpn的loss和roi的loss加起来一次性反向传播，但是不知为什么好像roi的坐标回归训练得不好~