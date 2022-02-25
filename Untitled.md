# 基于注意力机制和深度学习预测蛋白质可溶性的分类模型研究



注意力机制的双向长短记忆神经网络模型



## 摘要

蛋白质在生物生命活动中发挥着重要作用，其含量约占细胞的7~10%。天然蛋白质通常由20种不同的氨基酸组成，氨基酸序列决定了大多数蛋白质的结构、性质和功能。蛋白质的可溶性就是蛋白质的重要性质之一，其表征了蛋白质溶解于溶剂的能力，研究蛋白质的可溶性对于利用蛋白质有着重要意义。传统方法通过实验测定蛋白质的溶解度，但人力物力消耗大。近年来随着机器学习的发展，使应用计算方法预测蛋白质溶解度成为可能——基于特征的序列编码策略和基于深度学习的预测模型构建方法。基于特征的序列编码策略通过对蛋白质序列进行特征设计、特征选择和算法选择，利用人类先验知识构建分类模型，虽然仍存在一些问题，但已经取得了良好表现。而基于深度学习的预测模型构建方法在通过蛋白质序列预测蛋白质结构和功能上已经展现出了独特的优势。

而基于深度学习的预测模型构建方法在通过蛋白质序列预测蛋白质结构和功能上有着广阔的前景，并且已经在一些项目上取得了成功。本研究使用双向长短记忆神经网络模型，并加入了注意力机制。



### 关键词：

蛋白质可溶性；机器学习；深度学习；注意力机制；BiLSTM；



## 绪论

### 研究背景

蛋白质是生命体的重要组成部分，其含量约占细胞的7~10%，是含量最多的有机物。蛋白质主要可以分为结构蛋白和功能蛋白，功能包括维持细胞稳定，控制细胞物质运输，参与细胞信息交流，催化生化反应，在生命活动中发挥着重要作用。天然蛋白质通常由20种不同氨基酸组成，在核糖体合成并在高尔基体和内质网中折叠修饰。其氨基酸序列决定了蛋白质的结构、性质和功能。也就是说，蛋白质的信息都储存在其序列中。解密蛋白质序列信息，就能预测蛋白质的结构、性质和功能。

蛋白质的溶解度是蛋白质的重要性质之一，其表征了蛋白质溶解于溶剂的能力。在蛋白质工程的科学研究中，获得大量纯化、可溶性的、有生物活性的蛋白质是研究蛋白质理化性质与功能研究的基础。因此，获得蛋白质溶解度信息是一项重要的工作。研究蛋白质溶解性主要有两种方法：传统方法通过蛋白质可溶性实验来研究，其精准度高但费时费力；而基于计算的方法通过蛋白质序列特征构建机器学习模型，在预测蛋白质的可溶性有着良好表现。

### 研究现状

基于计算的方法研究蛋白质可溶性主要有两种技术路线——基于特征的序列编码策略和基于深度学习的预测模型构建方法。

目前，基于特征的序列编码策略在蛋白质可溶性研究上已经取得了成果。首先通过人类先验知识对蛋白质序列进行人工特征设计，包括二肽频率[1] 、三肽频率、氨基酸组成（Asn，Thr，Tyr）[2] ；之后应用复杂的特征选择技术去除无关或冗余的特征；最后，根据所选择的特征选择合适的机器学习算法。现在已经出现了许多基于特征的序列编码策略构建的蛋白质溶解性预测工具，包括PaRSnIP[3] ，PROSO II[4] ，CCSOL[5] 等等。

但是，特征设计通常是通过反复实验发现特征，需要大量的人工工作获取先验知识；而人工设计的特征可能与预测无关或者冗余，给使用机器学习算法训练的预测模型带来负面影响，需要进行大量的特征筛选工作。综上所述，基于特征的序列编码策略相当复杂。

而基于深度学习的预测模型构建方法可以避免这些问题。和基于特征的学习方法不同，深度学习算法可以从原始输入数据中自动学习合适的表示，无需设计特征和选择特征。[6] 深度学习已经在自然语言处理（NLP）、计算机视觉领域取得了巨大成功[7] ，使用基于注意力机制的深度学习模型的AlphaFold2更是在通过蛋白质序列预测蛋白质结构中取得了突破性的进展。[8] 在蛋白质可溶性预测上，Khurana等提出了使用卷积神经网络（CNNs）的DeepSol模型，这是首次将深度学习模型运用在蛋白质可溶性研究上。[9]



### 研究意义

近年来深度学习领域的快速发展，促进了使用基于计算的方法研究生物大分子性质、结构和功能。如应用了基于注意力机制的AlphaFold2，使预测蛋白质结构的难题迎刃而解。将领域内先进的深度学习模型和方法应用到蛋白质可溶性预测上，有望攻克预测蛋白质可溶性的难关。降低蛋白质可溶性研究成本，有助于蛋白质药物设计，利用大肠杆菌等微生物高效生产可溶性重组蛋白质，甚至能够给部分如阿尔海默茨症等由难溶性蛋白质导致的人类疾病研究带来有利影响。



## 研究内容及方法

### 蛋白质可溶性数据集

我们的训练集来源于美国东北结构基因组中心（NESG）在2001年至2008年对蛋白质表达和纯化获取的结果，包括82%细菌，12%古细菌，6%人类和0.3%其他真核生物。跟据在低速离心去除难溶性物质后蛋白质在上清液中的产量，对每个蛋白质样本给予溶解度评分S（0~5），S值为0表示在上清液中没有回收率，5表示回收率约为100%，中间值与这两个极端值之间大致成线性比例。[10]



### 双向长短记忆神经网络模型

长短记忆神经网络模型（LSTM）如下图所示[11]：

![../_images/lstm-3.svg](https://zh-v2.d2l.ai/_images/lstm-3.svg)

长短记忆神经网络引入了记忆元，是隐状态的一种特殊的类型，用于记录附加的信息，并且加入了三种门来控制记忆元，分别为遗忘门、输入门和输出门。通过隐状态 H~t-1~ 和输入 X~t~ 以及各自权重和偏置项计算各门的值，通过遗忘门决定保留多少过往记忆 C~t-1~  的内容，通过输入门决定采用多少来自于候选记忆元 C~t~ 的新内容，通过输出门决定传递多少记忆 C~t~ 的内容给新的隐状态 H~t~ 。计算公式如下：

$\mathbf{I}_t = \sigma(\mathbf{X}_t \mathbf{W}_{xi} + \mathbf{H}_{t-1} \mathbf{W}_{hi} + \mathbf{b}_i)$

$\mathbf{F}_t = \sigma(\mathbf{X}_t \mathbf{W}_{xf} + \mathbf{H}_{t-1} \mathbf{W}_{hf} + \mathbf{b}_f)$

$\mathbf{O}_t = \sigma(\mathbf{X}_t \mathbf{W}_{xo} + \mathbf{H}_{t-1} \mathbf{W}_{ho} + \mathbf{b}_o)$

$\tilde{\mathbf{C}}_t = \text{tanh}(\mathbf{X}_t \mathbf{W}_{xc} + \mathbf{H}_{t-1} \mathbf{W}_{hc} + \mathbf{b}_c)$

$\mathbf{C}_t = \mathbf{F}_t \odot \mathbf{C}_{t-1} + \mathbf{I}_t \odot \tilde{\mathbf{C}}_t$

$\mathbf{H}_t = \mathbf{O}_t \odot \tanh(\mathbf{C}_t)$

其中，W~mn~ 是权重参数，b~l~ 为偏置参数。

对于一般的有时序性的长短记忆神经网络模型，每个单元只能考虑过往记忆时间 t-1 的内容。例如在一个人的说话过程中，只能通过他在当前时刻前所说的话预测他所说的含义或者预测他在下一个时间将要说什么话，而不能通过他在下一个时间说的话来反推。但是如果对于完形填空问题，需要通过上下和下文内容推测缺失的词语，就显得无能为力了。而双向循环神经网络解决了这样的问题。它能够既考虑过往时间 t-1 的内容，又考虑到时间 t 及 t 以后的内容。其实现方式是将输入时序 t 逆转。如下图所示：

![img](https://zh-v2.d2l.ai/_images/birnn.svg)



### 注意力机制

注意力机制受启发于灵长类动物。它们的视觉系统接受到了大量而又复杂的信息输入，但是它们的中枢神经系统对这些信息输入的反应是不同的。它们会专注于其中一部分对它们而言重要的信息，而减少对其他不同要信息的关注甚至忽视这些信息[11]。这样的机制在生物漫长的进化过程中得以保留，充分说明了它的合理性。由此受启发，注意力机制开始被用于研究认知神经科学领域，并逐渐获得机器学习界的认可和广泛应用。

杨紫超等人提出了一种用于文档分类的层次注意力机制，包括两个层次的注意力机制，分别用于单词和句子上[12]。其中，单词注意力机制原理与本研究相似——在一个句子（氨基酸序列）中每个词语对句子意义（蛋白质性质）的贡献并非全然一样。通过提取出重要的词语，并且汇总它们的表示来形成句子向量。公式如下：

$u_{it}=tanh(W_wh_{it}+b_w)$

$\alpha_{it}=\frac{exp(\mathbf{u}_{it}^\top u_w)}{\sum_t exp(\mathbf{u}_{it}^\top u_w)}$

$s_i=\sum\limits_{t} a_{it}h_{it} $



### 洪泛法

“过拟合”是机器学习中常遇到的问题，是影响实验模型表现的因素之一。出现“过拟合”现象意味着模型训练集误差下降趋于0，测试集误差上升，需要防止此问题的出现。防止模型“过拟合”常用的方法有权重衰减、dropout、标签平滑等。Takashi Ishida等提出了一种名为“洪泛法”的方法[13]，通过设定的洪泛水平b，将训练集损失保持在一个小的恒定值，以避免训练损失降为0。公式如下：

$\widetilde{J}(\theta)=|J(\theta)-b|+b$

其中原始学习目标为 J ，洪泛水平为 b 。当 J(θ) ≥ b 时，对学习目标的更新没有影响，使模型梯度下降；而当 J < b 时，学习目标更新为 2b - J(θ) > b ，导致模型梯度上升，训练集损失增加。此时，原本测试集误差应该上升，因为模型梯度上升，测试集误差反而下降。如下图[B]阶段所示：

![image-20220224181837658](C:\Users\hp\AppData\Roaming\Typora\typora-user-images\image-20220224181837658.png)![image-20220224181844887](C:\Users\hp\AppData\Roaming\Typora\typora-user-images\image-20220224181844887.png)



## 研究结果及讨论

### 评价指标

鉴于本研究（预测蛋白质是否可溶）是一个二分类问题，于是采用以下评价指标：

准确率ACC：

$ACC=\frac{TP+TN}{TP+TN+FP+FN}$

特异性Spe：

$Spe=\frac{TN}{TN+FP}$

敏感性Sen：

$Sen=\frac{TP}{TP+FN}$

马修相关系数MCC：

$MCC=\frac{TP \times TN -FP \times FN}{\sqrt{(TP+FN)\times(TP+FP)\times(TN+FP)\times(TN+FN)}}$



其中：

TP：true positive 真阳性，正确地将阳性样本预测为阳性

TN：true negative 真阴性，正确地将阴性样本预测为阳性

FP：false positive 假阳性，错误地将阴性样本预测为阳性

FN：false negative 假阴性，错误地将阳性样本预测为阴性



### 研究过程及结果

#### 实验环境

本文实验环境的处理器CPU为 Intel(R) Core(TM) i5-9300H CPU @ 2.40GHz，图形加速卡GPU为 NVIDIA GeForce GTX 1050 4GB，操作系统为 Windows 10 20H2 家庭版（64 bit）。本文实验虚拟环境为 python 3.6，采用框架 torch 1.10.2+cu113 [14]构建网络模型。本文实验虚拟环境所有安装包在附录中列出。

#### 实验数据

本文实验数据来源于NESG，共9703条蛋白质序列和溶解度信息。蛋白质氨基酸序列长度最大为979，平均长度为243.8，上四分位数据为355。氨基酸种类为21（包含未知氨基酸以X表示）。将每条序列转化为长度为360的二十维序列特征向量。原始溶解度范围为0至5的整数，0表示不溶，大于0表示可溶，为二分类样本。其中3806条为不溶蛋白质，5897条为可溶蛋白质。使用one-hot编码将蛋白质溶解信息转化为一个二维向量。

将9703条蛋白质信息按照 3 : 1 : 1 的比例通过分层抽样（sklearn.model_selection.StratifiedShuffleSplit[15]）划分为训练集，验证集和测试集，数量分别为5821条，1941条，1941条。

#### 实验结果

1. 使用pytorch双向长短记忆神经网络架构（BiLSTM）进行训练，分别设置batch size=32，learning rate=1e-3，训练次数=25，使用GPU。最终结果为

   ```
   train loss:0.6240262520182264
   train acc:0.6351965343231684
   valid loss:0.7005206292769948
   valid acc:0.5832991805232939
   test acc:0.5797131149495234
   ```

2. 使用pytorch双向长短记忆神经网络架构（BiLSTM）并加入分层注意力机制（Hierarchical Attention Network），同样设置batch size=32，learning rate=1e-3，训练次数=25，使用GPU，并且使用洪泛法控制训练loss的下降，设置系数b=0.25。最终结果为

   train loss:0.6558668416935008
   train acc:0.619156457565643
   valid loss:0.6763222852060872
   valid acc:0.6010464670196656

   test acc:0.6093029956663808

3. 本文实验所有数据和源代码均可以在 https://github.com/wsxokm-XL/YModel 中获取。





#### 实验结果比较

![image-20220224162527255](C:\Users\hp\AppData\Roaming\Typora\typora-user-images\image-20220224162527255.png)

   

## 结论

本论文的研究以蛋白质原始序列作为输入，单独使用双向长短记忆神经网络模型和注意力机制构建机器学习模型预测蛋白质可溶性，使用较少的算力在较短的时间内训练模型，获得了0.580的准确率，超过了部分基于特征的序列编码策略构建的工具。从实验结果可以看出，使用双向长短记忆神经网络的模型在蛋白质可溶性预测有较大的潜力。在双向长短记忆神经网络中加入简单的注意力机制，便使得准确率从0.580增加到0.609，可以看出注意力机制对蛋白质可溶性预测准确度有所提升。但是本文模型构建过于简单，训练轮次较少，导致模型准确率相比于最先进的蛋白质可溶性预测工具仍有较大差距。未来如果能够在本文模型基础上增加训练轮次，增大数据量，将双向长短记忆神经网络模型更换为参数更少的门控循环单元（GRU），并结合更复杂的注意力机制，预期能够取得更好的模型表现。



## 参与项目心得

作为生物信息学专业的一名本科生，在参与“基于注意力机制和深度学习预测蛋白质可溶性的分类模型研究”项目前，我一直困惑于生物信息学是做什么的，能有什么意义。一个是自然界中生物打交道的学科，另一个是研究计算机的科学，如何能将这两者融汇为一个专业，令我费解。但在我参与项目的过程中，我渐渐明白了生物信息学的作用——将最先进的计算方法运用到生物学相关信息的处理中，便利生物学的研究。就像我所参加的项目，使用深度学习模型预测蛋白质可溶性。想要知道蛋白质可溶性，传统的方法都是通过实验测定，不但耗费研究时间，而且需要花费许多人力物力。但是生物信息学研究者们设计了许多高效的能够预测蛋白质可溶性的工具，使得这个工作变得十分简单。无论是设计蛋白质药物，还是可溶性蛋白质生产，都比以前要更加容易，这就是生物信息学所发挥出的巨大作用。

在我参与项目的过程中遇到了许多困难。从一开始，我对python，对机器学习几乎是一无所知。我听了麦吉尔大学的丁俊教授讲解的课程，让我对机器学习有了初步的了解，掌握了许多模型的数学原理，并且产生了浓厚的兴趣。我的导师熊毅老师也经常鼓励我，在项目上指导我，知无不答；同时他也在学习上、生活上给我提供了许多宝贵的建议，令我受益匪浅。我的同学也经常帮助我，给我提供学习资料，解答我的问题。还有许许多多在Github和个人博客上无私慷慨地分享的陌生人，他们让我学习到了许多知识，收获了很多经验。

通过参加这个项目，我了解了生物信息学的作用，掌握了许多生物学、机器学习和数据科学的知识，增强了阅读文献和动手编程的能力，也磨炼了我的意志。我觉得我从中收获很多，感受很多。

平时生活中，我常常听到生物工程专业的室友抱怨他们所使用的生物学软件不好用，希望能有更好的软件出现，或许这就是我作为一名生物信息学专业的学生未来的使命吧。



## 参考文献

[1] Idicula-Thomas, S., Kulkarni, A.J., Kulkarni, B.D., Jayaraman, V.K. and Balaji, P.V. (2006) A support vector machine-based method for predicting the propensity of a protein to be soluble or to form inclusion body on overexpression in Escherichia coli. Bioinformatics, 22, 278-284.

[2] Idicula-Thomas, S. and Balaji, P.V. (2005) Understanding the relationship between the primary structure of proteins and its propensity to be soluble on overexpression in Escherichia coli. Protein Sci., 14, 582-592.

[3] Rawi, R., Mall, R., Kunji, K., Shen, C.H., Kwong, P.D. and Chuang, G.Y. (2018) PaRSnIP: sequence-based protein solubility prediction using gradient boosting machine. Bioinformatics, 34, 1092-1098.

[4] Smialowski,P. et al. (2012) PROSO II - a new method for protein solubility prediction. FEBS J., 279, 2192–2200.

[5] Agostini,F. et al. (2012) Sequence-based prediction of protein solubility. J. Mol. Biol., 421, 237–241.

[6] Li F, Chen J, Leier A, et al. DeepCleave: a deep learning predictor for caspase and matrix metalloprotease substrates and cleavage sites. *Bioinformatics*. 2020;36(4):1057-1065. 

[7] 牛富生,郭延哺,李维华,刘文洋.基于序列特征融合的蛋白质可溶性预测[J].计算机科学,2022,49(01):285-291.

[8] Jumper, J*et al*. Highly accurate protein structure prediction with AlphaFold.*Nature*(2021).

[9] KHURANA S,RAWI R,KUNJI K,et al.DeepSol:a deep lear-ning framework for sequence-based protein solubility prediction[J].Bioinformatics,2018,34(15):2605-2613.

[10] Price,W.N. et al. (2011) Large-scale experimental studies show unexpected amino acid effects on protein expression and solubility in vivo in E. coli. Microb. Inf. Exp., 1, 6.

[11] 阿斯顿·张,李沐,扎卡里等.动手学深度学习[M].北京:人民邮电出版社,2019.

[12] Yang Z, Yang D, Dyer C, et al. Hierarchical attention networks for document classification[C]//Proceedings of the 2016 conference of the North American chapter of the association for computational linguistics: human language technologies. 2016: 1480-1489.

[13] Ishida T ,  Yamane I ,  Sakai T , et al. Do We Need Zero Training Loss After Achieving Zero Training Error?[J].  2020. （洪泛法）

[14] Paszke A, Gross S, Chintala S, et al. Automatic differentiation in pytorch[J]. 2017. （torch）

[15] Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.（sklearn）









Nadaraya, E.A., 1964. On estimating regression. Theory of Probability & Its Applications, 9(1), pp.141-142.

## 谢辞

首先我要感谢我的导师熊毅老师，他从项目开始就一直给予我支持和鼓励，解答我的疑惑，指导我一步步完成课题项目。我还要感谢麦吉尔大学丁俊教授，他教授了我许多知识，将我领进了机器学习的大门。我还要感谢我的同学余谷风给了我许多帮助。最后，我还要感谢许许多多在互联网上分享知识，传播知识的人，我从他们发布的文章和视频中学习到了很多相关知识，对我有所帮助。



## 附录

本文实验虚拟环境所有安装包目录，虚拟环境版本为 python3.6

```
# packages in environment at D:\anaconda\envs\myproject:
#
# Name                    Version                   Build  Channel
argon2-cffi               21.3.0                   pypi_0    pypi
argon2-cffi-bindings      21.2.0                   pypi_0    pypi
async-generator           1.10                     pypi_0    pypi
attrs                     21.4.0                   pypi_0    pypi
backcall                  0.2.0                    pypi_0    pypi
blas                      1.0                         mkl
bleach                    4.1.0                    pypi_0    pypi
certifi                   2021.5.30        py36haa95532_0
cffi                      1.15.0                   pypi_0    pypi
charset-normalizer        2.0.12                   pypi_0    pypi
colorama                  0.4.4                    pypi_0    pypi
cudatools                 0.0.1                    pypi_0    pypi
cycler                    0.11.0                   pypi_0    pypi
d2l                       0.17.0                   pypi_0    pypi
dataclasses               0.8                      pypi_0    pypi
decorator                 5.1.1                    pypi_0    pypi
defusedxml                0.7.1                    pypi_0    pypi
entrypoints               0.4                      pypi_0    pypi
icc_rt                    2019.0.0             h0cc432a_1
idna                      3.3                      pypi_0    pypi
importlib-metadata        4.8.3                    pypi_0    pypi
intel-openmp              2022.0.0          haa95532_3663
ipykernel                 5.5.6                    pypi_0    pypi
ipython                   7.16.3                   pypi_0    pypi
ipython-genutils          0.2.0                    pypi_0    pypi
ipywidgets                7.6.5                    pypi_0    pypi
jedi                      0.17.2                   pypi_0    pypi
jinja2                    3.0.3                    pypi_0    pypi
joblib                    1.0.1              pyhd3eb1b0_0
jsonschema                3.2.0                    pypi_0    pypi
jupyter                   1.0.0                    pypi_0    pypi
jupyter-client            7.1.2                    pypi_0    pypi
jupyter-console           6.4.0                    pypi_0    pypi
jupyter-core              4.9.2                    pypi_0    pypi
jupyterlab-pygments       0.1.2                    pypi_0    pypi
jupyterlab-widgets        1.0.2                    pypi_0    pypi
kiwisolver                1.3.1                    pypi_0    pypi
markupsafe                2.0.1                    pypi_0    pypi
matplotlib                3.3.4                    pypi_0    pypi
mistune                   0.8.4                    pypi_0    pypi
mkl                       2020.2                      256
mkl-service               2.3.0            py36h196d8e1_0
mkl_fft                   1.3.0            py36h46781fe_0
mkl_random                1.1.1            py36h47e9c7a_0
nbclient                  0.5.9                    pypi_0    pypi
nbconvert                 6.0.7                    pypi_0    pypi
nbformat                  5.1.3                    pypi_0    pypi
nest-asyncio              1.5.4                    pypi_0    pypi
notebook                  6.4.8                    pypi_0    pypi
numpy                     1.19.5                   pypi_0    pypi
numpy-base                1.19.2           py36ha3acd2a_0
packaging                 21.3                     pypi_0    pypi
pandas                    1.1.5                    pypi_0    pypi
pandocfilters             1.5.0                    pypi_0    pypi
parso                     0.7.1                    pypi_0    pypi
pickleshare               0.7.5                    pypi_0    pypi
pillow                    8.4.0                    pypi_0    pypi
pip                       21.2.2           py36haa95532_0
prometheus-client         0.13.1                   pypi_0    pypi
prompt-toolkit            3.0.28                   pypi_0    pypi
pycparser                 2.21                     pypi_0    pypi
pygments                  2.11.2                   pypi_0    pypi
pyparsing                 3.0.7                    pypi_0    pypi
pyrsistent                0.18.0                   pypi_0    pypi
python                    3.6.13               h3758d61_0
python-dateutil           2.8.2                    pypi_0    pypi
pytz                      2021.3                   pypi_0    pypi
pywin32                   303                      pypi_0    pypi
pywinpty                  1.1.6                    pypi_0    pypi
pyzmq                     22.3.0                   pypi_0    pypi
qtconsole                 5.2.2                    pypi_0    pypi
qtpy                      2.0.1                    pypi_0    pypi
requests                  2.27.1                   pypi_0    pypi
scikit-learn              0.24.2           py36hf11a4ad_1
scipy                     1.5.2            py36h9439919_0
send2trash                1.8.0                    pypi_0    pypi
setuptools                58.0.4           py36haa95532_0
six                       1.16.0             pyhd3eb1b0_1
sqlite                    3.37.2               h2bbff1b_0
terminado                 0.12.1                   pypi_0    pypi
testpath                  0.5.0                    pypi_0    pypi
threadpoolctl             2.2.0              pyh0d69192_0
torch                     1.10.2+cu113             pypi_0    pypi
torchaudio                0.10.2+cu113             pypi_0    pypi
torchvision               0.11.3+cu113             pypi_0    pypi
tornado                   6.1                      pypi_0    pypi
traitlets                 4.3.3                    pypi_0    pypi
typing-extensions         4.1.1                    pypi_0    pypi
urllib3                   1.26.8                   pypi_0    pypi
vc                        14.2                 h21ff451_1
vs2015_runtime            14.27.29016          h5e58377_2
wcwidth                   0.2.5                    pypi_0    pypi
webencodings              0.5.1                    pypi_0    pypi
wheel                     0.37.1             pyhd3eb1b0_0
widgetsnbextension        3.5.2                    pypi_0    pypi
wincertstore              0.2              py36h7fe50ca_0
zipp                      3.6.0                    pypi_0    pypi
```

