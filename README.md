# 基于图像去噪实现图像文本消除

此repository使用基于U-Net的网络的降噪模型，需要安装pytorch框架。

## 数据准备

训练集与测试集的默认目录为`./data/train`与`./data/valid`，train.py会先使用训练集进行训练，随后使用测试集计算测试集图像的PSNR，因此可以将相关数据集放入两个文件夹。train.py默认batch size为4，每次迭代需要训练500个batch（可以在训练前通过命令行参数设置），因此默认train size为至少2000张图片。

test.py默认测试图片路径为`./data/test`，因此你可以将需要测试的图片放入test中。

## 模型训练

可以通过输入以下命令查看需要设置的参数：
```
python train.py -h
```
如可以输入以下命令使用cuda迭代100次加速训练：
```
python train.py -e 100 --cuda
```
每迭代一次，会将相关数据保存在指定路径下，默认路径为`./ckpts`

## 模型测试
可以通过输入以下命令查看需要设置的参数：
```
python test.py -h
```
如需要添加文字水印，可以添加参数`--add-text`；通过`--noise-param`确定添加文本覆盖范围。程序执行完毕后，会显示图像并将其保存在指定目录下（默认`./test_save`）。如果需要对图片预处理，需要指定图片放缩大小。目前只支持单张图片处理，且模型测试暂不支持cuda加速。
示例：

```
python test.py --add-text --noise-param 0.25 --imge-name noise.png --pre-set 2
```