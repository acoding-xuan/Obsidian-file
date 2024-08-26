---
url: https://zhuanlan.zhihu.com/p/63397627
title: CS224N 笔记 (六)：语言模型与 RNN
date: 2023-10-17 16:34:15
tag: 
summary: 
---
CS224N 第六讲主要研究语言模型及循环神经网络在语言模型中的应用。

## **语言模型**

语言模型 (Language Modelling) 研究的是根据已知序列推测下一个单词的问题，即假设已知 $x^{(1)},x^{(2)},...,x^{(t)}$x^{(1)},x^{(2)},...,x^{(t)} ，预测下一个词 $x^{(t+1)}$x^{(t+1)} 的概率 $P(x^{(t+1)}|x^{(t)},...,x^{(1)})$P(x^{(t+1)}|x^{(t)},...,x^{(1)}) 的问题。根据条件概率的链式法则，我们也可以将其看做一系列词出现的概率问题 $P(x^{(t)},...,x^{(1)})=\prod_{t=1}^{T}P(x^{(t)}|x^{(t-1)},...,x^{(1)})$P(x^{(t)},...,x^{(1)})=\prod_{t=1}^{T}P(x^{(t)}|x^{(t-1)},...,x^{(1)})

语言模型应用广泛，比如手机上打字可以智能预测下一个你要打的词是什么，或者谷歌搜索时自动填充问题等都属于语言模型的应用。语言模型是很多涉及到产生文字或预测文字概率的 NLP 问题的组成部分，如语音识别、手写文字识别、自动纠错、机器翻译等等。

## **经典 n-gram 模型**

较为经典的语言模型是 n-gram 模型。n-gram 的定义就是连续的 n 个单词。例如对于 the students opened their __这句话，unigram 就是 "the", "students", "opened", "their", bigram 就是 "the students", "students opened", "opened their", 3-gram 就是 "the students opened", "students opened their"， 以此类推。该模型的核心思想是 n-gram 的概率应正比于其出现的频率，并且假设 $P(x^{(t+1)})$P(x^{(t+1)}) 仅仅依赖于它之前的 n-1 个单词，即

$P(x^{(t+1)}|x^{(t)},...,x^{(1)}) = P(x^{(t+1)}|x^{(t)},...,x^{(t-n+2)}) = \\ \frac{P(x^{(t+1)},x^{(t)},...,x^{(t-n+2)})}{P(x^{(t)},...,x^{(t-n+2)})} \approx \frac{count(x^{(t+1)},x^{(t)},...,x^{(t-n+2)})}{count(x^{(t)},...,x^{(t-n+2)})}$P(x^{(t+1)}|x^{(t)},...,x^{(1)}) = P(x^{(t+1)}|x^{(t)},...,x^{(t-n+2)}) = \\ \frac{P(x^{(t+1)},x^{(t)},...,x^{(t-n+2)})}{P(x^{(t)},...,x^{(t-n+2)})} \approx \frac{count(x^{(t+1)},x^{(t)},...,x^{(t-n+2)})}{count(x^{(t)},...,x^{(t-n+2)})}

其中 count 是通过处理大量文本对相应的 n-gram 出现次数计数得到的。

n-gram 模型主要有两大问题：

1.  稀疏问题 Sparsity Problem。在我们之前的大量文本中，可能分子或分母的组合没有出现过，则其计数为零。并且随着 n 的增大，稀疏性更严重。
2.  我们必须存储所有的 n-gram 对应的计数，随着 n 的增大，模型存储量也会增大。

这些限制了 n 的大小，但如果 n 过小，则我们无法体现稍微远一些的词语对当前词语的影响，这会极大的限制处理语言问题中很多需要依赖相对长程的上文来推测当前单词的任务的能力。

## **Window-based DNN**

一个自然的将神经网络应用到语言模型中的思路是 window-based DNN，即将定长窗口中的 word embedding 连在一起，将其经过神经网络做对下一个单词的分类预测，其类的个数为语裤中的词汇量，如下图所示

![](<assets/1697531656078.png>)

与 n-gram 模型相比较，它解决了稀疏问题与存储问题，但它仍然存在一些问题：窗口大小固定，扩大窗口会使矩阵 W 变大，且 $x^{(1)},x^{(2)}$x^{(1)},x^{(2)} 与 W 的不同列相乘，没有任何可共享的参数。

## **RNN**

RNN(Recurrent Neural Network) 结构通过不断的应用同一个矩阵 W 可实现参数的有效共享，并且没有固定窗口的限制。其基本结构如下图所示：

![](<assets/1697531656829.png>)

即前一时刻的隐藏状态以及当前的输入通过矩阵 W 加权求和然后在对其非线性变换并利用 softmax 得到对应的输出：

![](<assets/1697531656944.png>)

对于语言模型，RNN 的应用如下所示，即对 word embedding vector 进行如上操作：

![](<assets/1697531657282.png>)

RNN 的训练过程同样依赖于大量的文本，在每个时刻 t 计算模型预测的输出 $y^{(t)}$y^{(t)} 与真实值 $\hat y^{(t)}$\hat y^{(t)} 即 $x^{(t+1)}$x^{(t+1)} 的 cross-entropy loss，即

![](<assets/1697531657427.png>)

对于文本量为 T 的总的损失即为所有交叉熵损失的平均值：

![](<assets/1697531657841.png>)

语言模型的评估指标是 perplexity，即我们文本库的概率的倒数，也可以表示为损失函数的指数形式，所以 perplexity 越小表示我们的语言模型越好。

![](<assets/1697531658160.png>)

![](<assets/1697531658430.png>)

与之前的模型相比，RNN 模型的优势是：

1. 可以处理任意长度的输入。

2. 理论上 t 时刻可以利用之前很早的历史信息。

3. 对于长的输入序列，模型大小并不增加。

4. 每一时刻都采用相同的权重矩阵，有效的进行了参数共享。

其劣势是：

1. 由于需要顺序计算而不是并行计算，RNN 计算较慢。

2. 实际上由于梯度消失等问题距离较远的早期历史信息难以捕捉。

当然 RNN 不仅限于语言模型，它还可以应用于诸如 Named Entity Recognition, Sentiment Analysis 等问题。

上面我们讨论的是单向 (unidirectional) 的 RNN, 对于某些我们具有整个句子的上下文问题（不适用于语言模型，因为语言模型只有上文），我们可以用双向 (bidirectional) 的 RNN，例如 sentiment analysis 问题, 对于 the movie was terribly exciting 这句话，如果我们只看到 terribly 的上文，我们会认为这是一个负面的词，但如果结合下文 exciting，就会发现 terribly 是形容 exciting 的激烈程度，在这里是一个很正面的词，所以我们既要结合上文又要结合下文来处理这一问题。我们可以用两个分别从前向后以及从后向前读的 RNN，并将它们的隐藏向量联结起来，再进行 sentiment 的 classification，其结构如下：

![](<assets/1697531658892.png>)

总结一下，RNN 可以有效的进行参数共享，并且可以处理任意长度的输入，但是它存在着梯度消失的问题，下一讲会详细讲解该问题以及为解决它产生的 RNN 模型的变种如 LSTM，GRU 等模型。

**参考资料**

CS224N 第六讲讲义 [http://web.stanford.edu/class/cs224n/slides/cs224n-2019-lecture06-rnnlm.pdf](https://link.zhihu.com/?target=http%3A//web.stanford.edu/class/cs224n/slides/cs224n-2019-lecture06-rnnlm.pdf)

补充资料 [http://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes05-LM_RNN.pdf](https://link.zhihu.com/?target=http%3A//web.stanford.edu/class/cs224n/readings/cs224n-2019-notes05-LM_RNN.pdf)

第六讲视频 [https://youtu.be/iWea12EAu6U](https://link.zhihu.com/?target=https%3A//youtu.be/iWea12EAu6U)