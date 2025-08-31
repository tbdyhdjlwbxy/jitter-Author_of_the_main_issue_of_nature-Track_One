# jitter-Author_of_the_main_issue_of_nature-Track_One
来自第五届计图人工智能挑战赛赛道一nature正刊作者队伍的开源代码
# 一、数据集
由第五届计图人工智能挑战赛赛道一提供的乳腺癌超声影像：

TrainSet/images/train: 训练集图像

TrainSet/label/train.txt: 训练集label

TrainSet/label/val.txt: 验证集label

TrainSet/label/trainval.txt: 训练集+验证集label

TestSetA: A榜测试集图像

TestSetB: B榜测试集图像
# 二、思路
本项目旨在对乳腺癌超声影像根据BI-RADS标准进行六分类细粒度分类任务

采用ResNet50作为基线模型，发现测试集性能欠佳，分析表明存在严重的样本不均衡问题；

通过复制少数类样本平衡数据分布，分析显示0、1、5类别效果较好，2、3、4类别仍需提升；

引入Swin Transformer模型，通过超参数调优和将交叉熵损失函数替换为Focal Loss，测试集准确率提升至78%～80%；

采用Albumentations工具进行超声图像专用增强（高斯模糊、亮度调整、随机旋转），并结合难样本挖掘技术，准确率达到81%；

在Swin Transformer基础上融合ResNet50模型，结合局部特征与全局特征学习能力，最终测试集准确率提升至83.05%，B榜测试集准确率80.6%
# 三、运行
操作系统版本：Ubuntu 20.04.6 LTS

CUDA版本：CUDA 12.2

Python版本：python 3.7.16

运行以下命令创建环境:
```python
conda env create -f environment.yml -n jittor_env
conda activate jittor_env
```
环境创建完毕后，如果想通过训练得到checkpoint可以直接在终端分别输入以下三行命令:
```python
python 绝对路径/代码封存_81.25.py  # 替换为你的实际绝对路径
python 绝对路径/代码封存_82.35.py  # 替换为你的实际绝对路径
python 绝对路径/代码封存_83.05.py  # 替换为你的实际绝对路径
```
checkpoint分别会保存到baseline/的model_save_1、model_save_2、model_save_3目录下，若想得到checkpoint的推理结果则直接在上述的终端命令后分别直接添加--testonly，例如:
```python
python 绝对路径/代码封存_81.25.py --testonly   # 替换为你的实际绝对路径
python 绝对路径/代码封存_82.35.py --testonly   # 替换为你的实际绝对路径
python 绝对路径/代码封存_83.05.py --testonly   # 替换为你的实际绝对路径
```
测试集预测结果将分别对应保存在baseline/目录下。


