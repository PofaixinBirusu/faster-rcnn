# faster-rcnn复现
### 效果图
在show-images那个文件夹里~ 我不知道怎么传图
### 训练
训练是把rpn的loss和roi的loss加起来一次性反向传播，但是不知为什么好像roi的坐标回归训练得不好~