---
url: https://zhuanlan.zhihu.com/p/66268929
title: CS224N 笔记 (五):Dependency Parsing
date: 2023-10-16 21:17:10
tag: 
summary: 
---
对于**句法结构 (syntactic structure)** 分析，主要有两种方式：Constituency Parsing 与 Dependency Parsing。

## **Constituency Parsing 基本概念**

Constituency Parsing 主要用 phrase structure grammer 即短语语法来不断的将词语整理成嵌套的组成成分，又被称为 context-free grammers，简写做 CFG。

其主要步骤是先对每个词做词性分析 part of speech, 简称 POS，然后再将其组成短语，再将短语不断递归构成更大的短语。

例如，对于 the cuddly cat by the door, 先做 POS 分析，the 是限定词，用 Det(Determiner) 表示，cuddly 是形容词，用 Adj(Adjective) 代表，cat 和 door 是名词，用 N(Noun) 表示, by 是介词，用 P(Preposition) 表示。

然后 the cuddly cat 构成名词短语 NP(Noun Phrase)，这里由 Det(the)+Adj(cuddly)+N(cat) 构成，by the door 构成介词短语 PP(Preposition Phrase), 这里由 P(by)+NP(the door) 构成。

最后，整个短语 the cuddly cat by the door 是 NP，由 NP（the cuddly cat）+ PP(by the door) 构成。

关于 Constituency Parsing, 第 18 讲还会涉及，这一讲集中讨论 Dependency Parsing。

## **Dependency Parsing 基本概念**

Dependency Structure 展示了词语之前的依赖关系, 通常用箭头表示其依存关系，有时也会在箭头上标出其具体的语法关系，如是主语还是宾语关系等。

Dependency Structure 有两种表现形式，一种是直接在句子上标出依存关系箭头及语法关系，如：

![](<assets/1697462230516.png>)

另一种是将其做成树状机构（Dependency Tree Graph）

![](<assets/1697462231377.png>)

Dependency Parsing 可以看做是给定输入句子 $S=w_0w_1...w_n$S=w_0w_1...w_n （其中 $w_0$w_0 常常是 fake ROOT，使得句子中每一个词都依赖于另一个节点）构建对应的 Dependency Tree Graph 的任务。而这个树如何构建呢？一个有效的方法是 Transition-based Dependency Parsing。

## **Transition-based Dependency Parsing**

Transition-based Dependency Parsing 可以看做是 state machine，对于 $S=w_0w_1...w_n$S=w_0w_1...w_n ，state 由三部分构成 $(\sigma, \beta, A)$(\sigma, \beta, A) 。

$\sigma$\sigma 是 $S$S 中若干 $w_i$w_i 构成的 stack。

$\beta$\beta 是 $S$S 中若干 $w_i$w_i 构成的 buffer。

$A$A 是 dependency arc 构成的集合，每一条边的形式是 $(w_i,r,w_j)$(w_i,r,w_j) ，其中 r 描述了节点的依存关系如动宾关系等。

初始状态时， $\sigma$\sigma 仅包含 ROOT $w_0$w_0 ， $\beta$\beta 包含了所有的单词 $w_1...w_n$w_1...w_n ，而 $A$A 是空集 $\phi$\phi 。最终的目标是$\sigma$\sigma 包含 ROOT $w_0$w_0， $\beta$\beta 清空，而 $A$A 包含了所有的 dependency arc， $A$A 就是我们想要的描述 Dependency 的结果。

state 之间的 transition 有三类：

*   SHIFT：将 buffer 中的第一个词移出并放到 stack 上。
*   LEFT-ARC：将 $(w_j,r,w_i)$(w_j,r,w_i) 加入边的集合 $A$A ，其中 $w_i$w_i 是 stack 上的次顶层的词， $w_j$w_j 是 stack 上的最顶层的词。
*   RIGHT-ARC: 将 $(w_i,r,w_j)$(w_i,r,w_j) 加入边的集合 $A$A ，其中 $w_i$w_i 是 stack 上的次顶层的词， $w_j$w_j 是 stack 上的最顶层的词。

我们不断的进行上述三类操作，直到从初始态达到最终态。在每个状态下如何选择哪种操作呢？当我们考虑到 LEFT-ARC 与 RIGHT-ARC 各有 | R|（|R | 为 r 的类的个数）种 class，我们可以将其看做是 class 数为 2|R|+1 的分类问题，可以用 SVM 等传统机器学习方法解决。

## **Evaluation**

当我们有了 Dependency Parsing 的模型后，我们如何对其准确性进行评估呢？

我们有两个 metric，一个是 LAS（labeled attachment score）即只有 arc 的箭头方向以及语法关系均正确时才算正确，以及 UAS（unlabeled attachment score）即只要 arc 的箭头方向正确即可。

一个具体的例子如下图所示：

![](<assets/1697462232629.png>)

## **Neural Dependency Parsing**

传统的 Transition-based Dependency Parsing 对 feature engineering 要求较高，我们可以用神经网络来减少 human labor。

对于 Neural Dependency Parser，其输入特征通常包含三种：

*   stack 和 buffer 中的单词及其 dependent word。
*   单词的 Part-of-Speech tag。
*   描述语法关系的 arc label。

![](<assets/1697462233851.png>)

我们将其转换为 embedding vector 并将它们联结起来作为输入层，再经过若干非线性的隐藏层，最后加入 softmax layer 得到每个 class 的概率。

![](<assets/1697462234040.png>)

利用这样简单的前置神经网络，我们就可以减少 feature engineering 并提高准确度，当然，在之后几讲中的 RNN 模型也可以应用到 Dependency Parsing 任务中。

**参考资料**

第五讲讲义 [http://web.stanford.edu/class/cs224n/slides/cs224n-2019-lecture05-dep-parsing.pdf](https://link.zhihu.com/?target=http%3A//web.stanford.edu/class/cs224n/slides/cs224n-2019-lecture05-dep-parsing.pdf)

补充材料 [http://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes04-dependencyparsing.pdf](https://link.zhihu.com/?target=http%3A//web.stanford.edu/class/cs224n/readings/cs224n-2019-notes04-dependencyparsing.pdf)

第五讲视频 [https://youtu.be/nC9_RfjYwqA](https://link.zhihu.com/?target=https%3A//youtu.be/nC9_RfjYwqA)