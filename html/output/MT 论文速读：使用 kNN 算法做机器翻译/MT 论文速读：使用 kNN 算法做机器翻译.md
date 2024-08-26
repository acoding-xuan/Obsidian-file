---
url: https://zhuanlan.zhihu.com/p/479866520
title: MT 论文速读：使用 kNN 算法做机器翻译
date: 2024-04-20 23:15:21
tag: 
summary: 
---
今天介绍的 ICLR'21 论文_Nearest Neighbor Machine Translation_使用了 [kNN 算法](https://zhuanlan.zhihu.com/p/25994179)来做机器翻译。目前机器翻译的主流做法都是大数据 + 大参数模型做文本生成任务，而这篇论文则是利用 kNN 算法设计了新的机器翻译系统——即使用 kNN 算法在大规模的预处理数据上做检索，搜寻最相似的 k 条数据来辅助翻译模型的生成。

作者认为使用 kNN 算法来做机器翻译有三大优点：1. 表达能力强：kNN 算法可以在翻译生成时检索任意量的数据；2. 适应力强：可以通过替换 kNN 检索的数据来改变模型的领域；3. 可解释性：kNN 算法可以很容易地获知我们使用了哪些样本来帮助生成翻译结果。原论文的方法描述和实验设计都非常详实，是一篇优质论文。

各位看官觉得有帮助还请点个赞~

## Nearest Neighbor Machine Translation

### 发表：ICLR'21

### 作者：Urvashi Khandelwal, Angela Fan,Dan Jurafsky, Luke Zettlemoyer, Mike Lewis

### 机构：斯坦福大学、Facebook AI

### 链接：[https://openreview.net/forum?id=7wCBOfJ8hJM](https://link.zhihu.com/?target=https%3A//openreview.net/forum%3Fid%3D7wCBOfJ8hJM)

### 研究目的：

把 kNN 算法引入机器翻译任务。新算法在提升 BLEU 性能的同时也有很好的解释性，在机器翻译领域适应上也有不错的效果。

### 方法简介：

作者的算法分为两个步骤，同时还需要一个预训练好的 MT 神经网络模型（下面简称_预训练模型，_记作 f）来帮助表征文本，具体做法如下

1.  数据存储：作者把原始数据处理成特殊的 key-value 对的形式。其中 key 为被预训练模型表征的源语句 + 目标语句的前缀，一般使用预训练模型 decoder 最后一层隐藏层向量作为 <源语句，目标语句前缀> 表征；value 则为目标语句前缀的下一个单词。

举个例子，对于 "I love you -> 我爱你" 的英中语对，我们可以按照中文侧汉字的顺序构建多组 key-value 对，每组分别是：

a. key=f(<I love you, BOS> )， value = 我

# 其中 BOS 代表 Begin of Sentence，

# f(<x,y>) 代表预训练模型输入 x 和 y 作为源语句和目标语句时，预训练模型 decoder 最后一层的隐藏层向量

b. key=f(<I love you, BOS 我> )， value = 爱

c. key=f(<I love you, BOS 我爱> )， value = 你

d. key=f(<I love you, BOS 我爱你> )， value=EOS

# 其中 EOS 代表 End of Sentence

2. 文本生成：做翻译时，我们也用类似的方法可以得到 f(<源语句，目标语句前缀>) 的表征，我们这里记作 query，此时，这可以通过 kNN 算法去查找上一步构建 key-value 对的中最邻近 k 个的表征向量 key，并将其 value 作为候选单词。作者提出根据 key 和 query 的距离作为权重，来规整 value 的概率：

![](<assets/1713626121496.png>)

最终的生成新词概率分布为 kNN 概率与预训练模型新词的概率的线性插值：

![](<assets/1713626121588.png>)

作者的配图也巧妙的展示了 kNN 概率的计算过程：

![](<assets/1713626121673.png>)

### 实验探索：

作者的主实验分两组设定：1. 在 WMT’19 英德语对上做双语翻译；2. 在 CCMatrix 数据上做 17 个语种的多语言翻译。使用的测试集都来自于 WMT‘17 newstest。

在不增加额外训练的情况下，相比原始的 MT 模型，kNN 在英德双语上提升了 1.5 个 BLEU、在多语种翻译上平均提升了 1.4 个 BLEU。

此外，作者还探索了机器翻译领域适应 (domain adaptation)。作者在不加入额外训练的情况下，仅把 key-value 数据存储步骤的语料替换为其他领域的数据。作者发现，在通用 WMT 数据集上与训练的模型 + 特定领域的 key-value 数据可以大幅提升原模型的 BLEU 分数（平均 + 9.2 BLEU）。同时，作者发现 kNN-MT 在领域适应上具有健壮性：即引入了领域外数据进入 key-value 数据后，也很少影响原 domain 的性能。

作者也简要地提及了 kNN-MT 目前的缺点：计算开销大、效率太慢。在作者的实验中，从十亿级别的数据中检索 64 个最接近样本，会导致 kNN-MT 的生成速度比传统系统慢两个数量级。作者呼吁今后的研究者们可以开发更快的 kNN 搜索工具。

### 笔者简评

一篇有开创性的机器翻译论文，首次把 [kNN 类语言模型](https://zhuanlan.zhihu.com/p/90890672)运用到了机器翻译领域，也验证了其效果。

笔者认为 kNN 算分能够提升 MT 模型的主要原因在于能够检索大量的数据来帮助模型生成新词，即作者所言的”表现力强 “和” 适应力“强。笔者认为，kNN 检索大幅度缓解了常规神经网络模型的[灾难遗忘](https://link.zhihu.com/?target=https%3A//www.cnblogs.com/JarvanWang/p/9152636.html)，同时还可以利用训练集以外的数据来增强表现力，此两点是 kNN-MT 在 in-domain 和 domain adaptation 上表现出色的原因。

作为一个新方法，目前 kNN-MT 的缺点也很明显：1. 仍然需要先训练一个常规的 MT 模型，并不能节约参数；2. 构建 key-value 数据库比较费时；3. 生成时需要检索整个 key-value 数据库，也非常比较费时。其中第三点是 kNN 算法的普遍缺点（测试时高开销），而前两点则没有继承 kNN 算法的普遍优点 (无参数、0 训练成本)——笔者认为这是因为 kNN 在作者的模型中其实更多是一个” 辅助 “的角色，即作者的方法本身还是要依赖一个常规模型作为主体，用 kNN 的检索来帮忙——这种” 辅助“角色从 key-value 数据库的构造过程 & 测试时仍然需要计算预训练模型的概率分布上也可见一斑。kNN-MT 的高额计算开销也使得在 domain adaptation 的实际运用上，相比 finetune 没有那么大的优势。

总之，kNN-MT 还是给近年来大模型 + 大数据的训练范式以外提供了一条新的性能提升之路，给予适当的优化，笔者很看好 kNN-MT 在机器翻译领域的应用前景。

#看到这里了就点个赞支持一下笔者吧