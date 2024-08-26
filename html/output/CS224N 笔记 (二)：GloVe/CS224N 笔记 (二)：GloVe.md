---
url: https://zhuanlan.zhihu.com/p/60208480
title: CS224N 笔记 (二)：GloVe
date: 2023-10-16 20:31:57a
tags: 
summary:
---
上一讲 [CS224N 笔记 (一)：Word Vector](https://zhuanlan.zhihu.com/p/59016893) 主要介绍了 Word2Vec 模型，它是一种基于 local context window 的 direct prediction 预测模型，对于学习 word vector，还有另一类模型是 count based global matrix factorization，这一讲主要介绍了后一类模型以及 Manning 教授组结合两者优点提出的 GloVe 模型。

## **SVD 模型**

count based 模型的经典代表是 SVD(Single Value Decomposition）模型。通常我们先扫过一遍所有数据，然后得到单词同时出现的矩阵 co-occurrence matrix，假设矩阵用 $X$X 表示，然后我们对其进行 SVD 得到 $X$X 的分解形式 $USV^T$USV^T 。

如何产生矩阵 $X$X 通常有两种选择。一是 word-document co-occurrence matrix, 其基本假设是在同一篇文章中出现的单词更有可能相互关联。假设单词 $i$i 出现在文章 $j$j 中，则矩阵元素 $X_{ij}$X_{ij} 加一，当我们处理完数据库中的所有文章后，就得到了矩阵 $X$X ，其大小为 $|V|\times M$|V|\times M ，其中 $|V|$|V| 为词汇量，而 $M$M 为文章数。这一构建单词文章 co-occurrence matrix 的方法也是经典的 Latent Semantic Analysis 所采用的。

另一种选择是利用某个定长窗口中单词与单词同时出现的次数来产生 window-based word-word co-occurrence matrix。下面以窗口长度为 1 来举例，假设我们的数据包含以下几个句子：

1.I like deep learning.

2.I like NLP.

3.I enjoy flying。

则我们可以得到如下的 word-word co-occurrence matrix:

![](<assets/1697459518010.png>)

可以看出，随着词汇量的增大，矩阵 $X$X 的尺度会越来大，为了有效的存储，我们可以对其进行 SVD 处理，即将其转化为酉矩阵 X 对角矩阵 X 酉矩阵的形式：

![](<assets/1697459519819.png>)

为了减少尺度同时尽量保存有效信息，可保留对角矩阵的最大的 k 个值，其余置零，并将酉矩阵的相应的行列保留，其余置零：

![](<assets/1697459521755.png>)

这就是经典的 SVD 算法。

## **GloVe 算法**

比较 SVD 这种 count based 模型与 Word2Vec 这种 direct prediction 模型，它们各有优缺点：
**Count based 模型优点是训练快速，并且有效的利用了统计信息，缺点是对于高频词汇较为偏向，并且仅能概括词组的相关性，而且有的时候产生的 word vector 对于解释词的含义如 word analogy 等任务效果不好**；

**Direct Prediction 优点是可以概括比相关性更为复杂的信息，进行 word analogy 等任务时效果较好，缺点是对统计信息利用的不够充分。**

所以 Manning 教授他们想采取一种方法可以结合两者的优势，并将这种算法命名为 GloVe（Global Vectors 的缩写），表示他们可以有效的利用全局的统计信息。

那么如何有效的利用 `word-word， co-occurrence count` 并能学习到词语背后的含义呢？首先为表述问题简洁需要，先定义一些符号：对于矩阵 $X$ ,$X_{ij}$ 代表了单词 $j$ 出现在单词 $i$ 上下文中的次数，则 $X_i=\sum_k X_{ik}$ 即代表所有出现在单词 $i$i 的上下文中的单词次数。我们用 $P_{ij}=P(j|i)=X_{ij}/X_i$来代表单词 $j$ 出现在单词 $i$上下文中的概率。


我们用一个小例子来解释如何利用 co-occurrence probability 来表示词汇含义：

![](<assets/1697459522008.png>)

例如我们想区分热力学上两种不同状态 `ice` 冰与蒸汽 `steam`，它们之间的关系可通过与不同的单词 $x$ 的 co-occurrence probability 的比值来描述，例如对于 solid 固态，虽然 $P(solid|ice)$ 与 $P(solid|steam)$本身很小，不能透露有效的信息，但是它们的比值 $\frac{P(solid|ice)}{P(solid|steam)}$ 却较大，因为 solid 更常用来描述 ice 的状态而不是 steam 的状态，所以在 ice 的上下文中出现几率较大，对于 gas 则恰恰相反，而对于 water 这种描述 ice 与 steam 均可或者 fashion 这种与两者都没什么联系的单词，则比值接近于 1。所以相较于单纯的 co-occurrence probability，实际上 `co-occurrence probability 的相对比值更有意义`。

基于这些观察，视频里直接给出了 GloVe 的损失函数形式：

![](<assets/1697459522217.png>)

个人觉得这里跳跃性略大，并没有解释其背后的思路，所以我们需要参考 GloVe 论文 [https://nlp.stanford.edu/pubs/glove.pdf](https://link.zhihu.com/?target=https%3A//nlp.stanford.edu/pubs/glove.pdf) 来了解一下作者的思路以便理解。有意思的是该文第一作者 Jeffrey Pennington 在成为 Manning 组的 PostDoc 之前是理论物理的博士，他用了物理学家简化假设做 back-of-envelope 计算合理推断的习惯，而并没有从第一性原理出发做严谨的推导，在了解其思路时需要记住这点。

基于对于以上概率比值的观察，我们假设模型的函数有如下形式：

$$F(w_i,w_j,\tilde w_k) = \frac{P_{ik}}{P_{jk}}$$

其中， $\tilde{w}$代表了 `context vector`，如上例中的 `solid，gas，water，fashion` 等。 $w_i,w_j$ 则是我们要比较的两个词汇，如上例中的 `ice，steam`。

$F$ 的可选的形式过多，我们希望有所限定。首先我们希望的是 $F$ 能有效的在单词向量空间内表示概率比值，由于向量空间是线性空间，一个自然的假设是 $F$ 是关于向量 $w_i,w_j$ 的差的形式：

$$F(w_i-w_j,\tilde w_k) = \frac{P_{ik}}{P_{jk}}$$

等式右边为标量形式，左边如何操作能将矢量转化为标量形式呢？一个自然的选择是矢量的点乘形式：

$$F((w_i-w_j)^T\tilde w_k) = \frac{P_{ik}}{P_{jk}}$$

在此，作者又对其进行了对称性分析，即对于 `word-word co-occurrence`，将向量划分为 center word 还是 context word 的选择是不重要的，即我们在交换 $w\leftrightarrow \tilde w$与 $X\leftrightarrow X^T$的时候该式仍然成立。如何保证这种对称性呢？

我们分两步来进行，首先要求满足 $F((w_i-w_j)^T\tilde w_k) = \frac{F(w_i^T\tilde w_k)}{F(w_j^T\tilde w_k)}$，该方程的解为 $F=exp$ 同时与 $F((w_i-w_j)^T\tilde w_k) = \frac{P_{ik}}{P_{jk}}$F((w_i-w_j)^T\tilde w_k) = \frac{P_{ik}}{P_{jk}} 相比较有 $F(w_i^T\tilde w_k)=P_{ik}=\frac{X_{ik}}{X_i}$F(w_i^T\tilde w_k)=P_{ik}=\frac{X_{ik}}{X_i}

所以， $w_i^T\tilde w_k = log(P_{ik}) = log(X_{ik})-log(X_i)$w_i^T\tilde w_k = log(P_{ik}) = log(X_{ik})-log(X_i)

注意其中 $log(X_i)$log(X_i) 破坏了交换 $w\leftrightarrow \tilde w$w\leftrightarrow \tilde w 与 $X\leftrightarrow X^T$X\leftrightarrow X^T 时的对称性，但是这一项并不依赖于 $k$k ，所以我们可以将其融合进关于 $w_i$w_i 的 bias 项 $b_i$b_i ，第二部就是为了平衡对称性，我们再加入关于 $\tilde w_k$\tilde w_k 的 bias 项 $\tilde b_k$\tilde b_k ，我们就可以得到 $w_i^T\tilde w_k + b_i + \tilde b_k = log(X_{ik})$w_i^T\tilde w_k + b_i + \tilde b_k = log(X_{ik}) 的形式。

另一方面作者注意到模型的一个缺点是对于所有的 co-occurence 的权重是一样的，即使是那些较少发生的 co-occurrence。作者认为这些可能是噪声，所以他加入了前面的 $f(X_{ij})$f(X_{ij}) 项来做 weighted least squares regression 模型，即为

$J=\sum_{i,j=1}^Vf(X_{ij})(w_i^T\tilde w_j + b_i + \tilde b_j -logX_{ij})^2$J=\sum_{i,j=1}^Vf(X_{ij})(w_i^T\tilde w_j + b_i + \tilde b_j -logX_{ij})^2 的形式。

其中权重项 $f$f 需满足一下条件：

1.  $f(0)=0$f(0)=0 ，因为要求 $lim_{x\rightarrow 0}f(x)log^2x$lim_{x\rightarrow 0}f(x)log^2x 是有限的。
2.  较少发生的 co-occurrence 所占比重较小。
3.  对于较多发生的 co-occurrence， $f(x)$f(x) 也不能过大。

作者试验效果较好的权重函数形式是

![](<assets/1697459522536.png>)

![](<assets/1697459522868.png>)

可以看出，GloVe 的模型公式的导出有很多假设与 trial and error 的函数设置，想寻求更严谨些的理论的小伙伴可参考这篇文章 [A Latent Variable Model Approach to PMI-based Word Embeddings](https://link.zhihu.com/?target=http%3A//aclweb.org/anthology/Q16-1028)。

## **GloVe 与 Word2Vec 性能比较**

虽然 GloVe 的作者在原论文中说 GloVe 结合了 SVD 与 Word2Vec 的优势，训练速度快并且在各项任务中性能优于 Word2Vec，但是我们应该持有怀疑的态度看待这一结果，可能作者在比较结果时对于 GloVe 模型参数选择较为精细而 Word2Vec 参数较为粗糙导致 GloVe 性能较好，或者换另一个数据集，改换样本数量，两者的性能又会有不同。实际上，在另一篇论文 [Evaluation methods for unsupervised word embeddings](https://link.zhihu.com/?target=http%3A//www.aclweb.org/anthology/D15-1036) 中基于各种 intrinsic 和 extrinsic 任务的性能比较中，Word2Vec 结果要优于或不亚于 GloVe。实际应用中也是 Word2Vec 被采用的更多，对于新的任务，不妨对各种 embedding 方法都做尝试，选择合适自己问题的方法。
