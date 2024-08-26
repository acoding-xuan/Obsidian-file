---
url: https://zhuanlan.zhihu.com/p/64887738
title: CS224N 笔记 (八)：机器翻译、Seq2Seq 与 Attention
date: 2023-10-17 17:00:48
tag: 
summary: 
---
这一讲研究 NLP 中一个重要的领域——机器翻译 Machine Translation。机器翻译顾名思义，就是将源语言中的文字转换成对应的目标语言中的文字的任务。早期的机器翻译很多是人为 rule-based 的，随后逐渐发展出依赖于统计信息的 Statistical Machine Translation（简称 SMT），之后又发展出利用神经网络使得准确度大幅改善的 Neural Machine Translation（NMT），它依赖于 Sequence-to-Sequence 的模型架构，以及进一步将其改善的 Attention 机制。

## **SMT**

SMT 的核心思想是从数据中学习出概率模型。假设我们想要从法语翻译成英语，我们希望给定输入的法语句子 $x$x ，找到对应最好的英语句子 $y$y , 即找到 $\underset{y}{\mathrm{argmax}}P(y|x)$\underset{y}{\mathrm{argmax}}P(y|x)

利用贝叶斯法则，这相当于求 $\underset{y}{\mathrm{argmax}}P(x|y)P(y)$\underset{y}{\mathrm{argmax}}P(x|y)P(y) , 其中 $P(x|y)$P(x|y) 被称作 translation model 模拟单词或词组该如何翻译，主要是处理局部的词组的翻译， $P(y)$P(y) 被称作 language model 即如何挑选这些单词或词组组成合理的目标语言中的句子。之前[语言模型](https://zhuanlan.zhihu.com/p/63397627)已总结过如何学习语言模型，而对于 translation model, 我们需要 parallel data，即大量的成对的法语、英语句子。

SMT 系统实际上要比这个复杂很多，而且需要很多 feature engineering，很难维护。而 NMT 的出现极大的改善了这些问题。

## **Neural Machine Translation 与 Seq2Seq**

NMT 依赖于 Sequence-to-Sequence 的模型架构，即通过一个 RNN 作为 encoder 将输入的源语言转化为某表征空间中的向量，再通过另一个 RNN 作为 decoder 将其转化为目标语言中的句子。我们可以将 decoder 看做预测目标句子 $y$y 的下一个单词的语言模型，同时其概率依赖于源句子的 encoding，一个将法语翻译成英语的 Seq2Seq 模型如下图所示。

![](<assets/1697533248209.png>)

训练过程中，损失函数与语言模型中类似，即各步中目标单词的 log probability 的相反数的平均值：

![](<assets/1697533248448.png>)

上例中的损失函数如下图所示，并且我们可以看出损失函数的梯度可以一直反向传播到 encoder，模型可以整体优化，所以 Seq2Seq 也被看做是 end2end 模型：

![](<assets/1697533248648.png>)

在做 inference 的时候，我们可以选择 greedy decoding 即每一步均选取概率最大的单词并将其作为下一步的 decoder input。

![](<assets/1697533248831.png>)

但是 greedy decoding 的问题是可能当前的最大概率的单词对于翻译整个句子来讲不一定是最优的选择，但由于每次我们都做 greedy 的选择我们没机会选择另一条整体最优的路径。

为解决这一问题，一个常用的方法是 beam search decoding，其基本思想是在 decoder 的每一步，我们不仅仅是取概率最大的单词，而是保存 k 个当前最有可能的翻译假设，其中 k 称作 beam size，通常在 5 到 10 之间。

对于 $y_1,...,y_t$y_1,...,y_t 的翻译，它的分数是

![](<assets/1697533249056.png>)

分数越高越好，但求和中的每一项都是负数，这会导致长的翻译分数更低，所以最后在选取整句概率最大的翻译时，要对分数做归一化

![](<assets/1697533249300.png>)

与 SMT 相较，NMT 的优势是我们可以整体的优化模型，而不是需要分开若干个模型各自优化，并且需要 feature engineering 较少，且模型更灵活，准确度也更高。其缺点是更难理解也更纠错，且很难设定一些人为的规则来进行控制。

## **BLEU**

对于机器翻译模型，我们如何来衡量它的好坏呢？一个常用的指标是 BLEU（全称是 **B**i**l**ingual **E**valuation **U**nderstudy)。BLEU 基本思想是看你 machine translation 中 n-gram 在 reference translation（人工翻译作为 reference）中相应出现的几率。

我们用

![](<assets/1697533249581.png>)

来表示 n-gram 的 precision score， $w_n=1/2^n$w_n=1/2^n 作为权重，另外引入对过短的翻译的 penalty $\beta = e^{min(0,1-\frac{len_{ref}}{len_{MT}})}$\beta = e^{min(0,1-\frac{len_{ref}}{len_{MT}})}

BLEU score 可以表示为（其中 k 通常选为 4）

![](<assets/1697533249936.png>)

BLEU 可以较好的反应翻译的准确度，当然它也不是完美的，也有许多关于如何改进机器翻译 evaluation metric 方面的研究。

## **Attention**

观察 Seq2Seq 模型，我们会发现它有一个信息的瓶颈即我们需要将输入的所有信息都 encode 到 encoder 的最后一个 hidden state 上，这通常是不现实的，因此引入 Attention 机制来消除这一瓶颈：在 decoder 的每一步，通过与 encoder 的直接关联来决定当前翻译应关注的源语句的重点部分。详细步骤可参考之前的总结文章 [Attention 机制详解（一）——Seq2Seq 中的 Attention](https://zhuanlan.zhihu.com/p/47063917)。

## **参考资料**

第八讲讲义 [http://web.stanford.edu/class/cs224n/slides/cs224n-2019-lecture08-nmt.pdf](https://link.zhihu.com/?target=http%3A//web.stanford.edu/class/cs224n/slides/cs224n-2019-lecture08-nmt.pdf)

补充材料 [http://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes06-NMT_seq2seq_attention.pdf](https://link.zhihu.com/?target=http%3A//web.stanford.edu/class/cs224n/readings/cs224n-2019-notes06-NMT_seq2seq_attention.pdf)

第八讲视频 [https://youtu.be/XXtpJxZBa2c](https://link.zhihu.com/?target=https%3A//youtu.be/XXtpJxZBa2c)