# Distillation
[Github仓库](https://github.com/0Jmins0/Distillation)

# 问题设定
多视图的特征学习（AlexNet->蒸馏到AlexNet/CLIP里）+ 图像检索

# 前期准备
## 数据集
  * 180多，随机/选取 30个
  * 压缩17G，解压后1.57T，想办法传上去
  * 切分数据集为已知类别 (``DS``) 和未知类别 (``DU``)各20类，再分别切分成训练集(``DtSr/DtUr``)和检索集(``DrSe/DrUe``),使用``DtSr``训练，分别在``DrSe/DrUe``上测试
  * 标签
    * ```
      图像A（类别1，实例1，视图1）与图像B（类别1，实例1，视图2）：标签为1。
      图像A（类别1，实例1，视图1）与图像C（类别1，实例2，视图1）：标签为0。
      ```
```
数据集/
    类别1/
        实例1/
            视图1.jpg
            视图2.jpg
            ...
        实例2/
            视图1.jpg
            视图2.jpg
            ...
    类别2/
        实例1/
            视图1.jpg
            视图2.jpg
            ...
        实例2/
            视图1.jpg
            视图2.jpg
            ...
```


## 工作流程
1. 搭建baseline
   * 多视图特征学习
     * 现在是如何提取和融合多视图的特征的，怎么操作？用的什么网络？
       * 多视图分类，其中输入数据包含多个视图或来源，旨在基于这些多个视图对样本的标签进行分类或预测。
       * 使用 ``MVCNN`` 的结构，将其中的 ``CNN1`` 替换成 ``CLIP``,后接视图池化层，再接一个 ``CNN2`` 输出最终的特征
   * 图像检索
     * 输入：单张图片？一组多视图？
       * 单张图片
     * 分类的目标是输出label判别对错，检索的目标是什么？如何求loss？
       * 分为：实例级别、种类级别
       * 还是分类任务，多使用 **三元组损失** 
     * 如何同时考虑实例和分类的
       * 考虑 LOSS = label loss + instance loss（并赋予 label loss 稍大的占比）
2. 替换蒸馏模型


## TODO LIST
* 【已完成】数据集重构代码
* 数据集预处理和构建代码


## 使用介绍
1. 调用 `remain_30` 函数，将 $180$ 张视图随机保留 $30$ 张
2. 运行 `rebuild_dataset.py`，将原本数据集重构成如下结构
    ```
    dataset_final/
        DS/
            train/
                class1/
                    instance1/
                    instance2/
                class2/
                    instance1/
                    instance2/
                ...
            retrieval/
                class1/
                    instance3/
                    instance4/
                class2/
                    instance3/
                    instance4/
                ...
        DU/
            train/
                class21/
                    instance1/
                    instance2/
                class22/
                    instance1/
                    instance2/
                ...
            retrieval/
                class21/
                    instance3/
                    instance4/
                class22/
                    instance3/
                    instance4/
                ...
    ```
3. 



## STEP 1 特征提取

### 数据集 ModelNet40_180
**描述：** 有 $a$ 个类别，每个类别有 $b_i$ 个实例，每个实例有 $180$ 张视图

**特征提取：**
1. 每张视图提出一个一定规格的矩阵 $A_i$
2. 将每个矩阵 $A_i$ 压缩成一个 $(x,)$ 的一维向量，得到 $180$ 个特征向量
3. 将 $180$ 张视图的特征向量取均值，得到一个 $(x,)$ 的特征向量
4. 最终合并每个实例、每个类别，得到 $(x , a , b)$ 规格的特征矩阵


||CLIP|SAM|DINOv2|
|----|----|----|----|
|modele|ViT-B/32|ViT-L|dinov2_vits14|
|Input_Size|(224,224)|(1024,1024)|(1024 // 14 * 14, 1024 // 14 * 14)|
|特征|encode_image|image_encoder|x_prenorm(归一化之前的特征)|
|Output_Size_Before|(180, 512)|(180, 256, 64, 64)|(180, 5330, 384)|
|Output_Size_Mean_Flatten|(512,)|(1048576,)|(2046720,)|
|Output_Size_After(每个类别目前只传了3个实例)|(3, 512)|(3,1048576)|(3, 2046720)|
|range(肉眼看)|(-1e-1,1e-1)|(-1e-2,1e-2)|(-1e-1,1e-1)|

**Q:**
* 每个模型都有好多，S/B/L，该如何选择 **（选小的）**
* 对于每一个模型，具体应该提取哪一部分特征，是需要边做边实验哪个好嘛 **（输出层）**
* 多视图的目的
* 关系的表示

## STEP 2 搭建 baseline
任务目标：图片分类，超过教师模型

通过多视图的特征提取，抓取不同角度的关系，提高分类精度
  