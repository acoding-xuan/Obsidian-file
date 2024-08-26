# 其它模型
## Word2Vec
Word2Vec 的训练模型本质上是只具有一个隐含层的神经元网络（如下图）。

![](<assets/1698915504063.png>)

它的输入是采用 One-Hot 编码的词汇表向量，它的输出也是 One-Hot 编码的词汇表向量。使用所有的样本，训练这个神经元网络，等到收敛之后，从输入层到隐含层的那些权重，便是每一个词的采用 Distributed Representation 的词向量。比如，上图中单词的 Word embedding 后的向量便是矩阵 $W_{V×N}$ 的第 i 行的转置。这样我们就把原本维数为 V 的词向量变成了维数为 N 的词向量（N 远小于 V），并且词向量间保留了一定的相关关系。

### cbow 与 skip-gram
与 CBOW 根据语境预测目标单词不同，Skip-gram 根据当前单词预测语境，如下图右部分所示。假如我们有一个句子 “There is an apple on the table” 作为训练数据，CBOW 的输入为（is,an,on,the），输出为 apple。而 Skip-gram 的输入为 apple，输出为（is,an,on,the）。
![](<assets/1698915504277.png>)
https://zhuanlan.zhihu.com/p/61635013

## FastText

![](<assets/1698918124215.png>)
### （1）字符级别的 n-gram
word2vec 把语料库中的每个单词当成原子的，它会为每个单词生成一个向量。这忽略了单词内部的形态特征，比如：“apple” 和 “apples”，“达观数据” 和“达观”，这两个例子中，两个单词都有较多公共字符，即它们的内部形态类似，但是在传统的 word2vec 中，这种单词内部形态信息因为它们被转换成不同的 id 丢失了。
为了克服这个问题，fastText 使用了`字符级别的 n-grams` 来表示一个单词。对于单词 “apple”，假设 n 的取值为 3，则它的 trigram 有 “<ap”, “app”, “ppl”, “ple”, “le>”

其中，<表示前缀，>表示后缀。于是，我们可以用这些 trigram 来表示 “apple” 这个单词，进一步，我们可以用这 5 个 trigram 的向量叠加来表示 “apple” 的词向量。

这带来两点**好处**：
1. 对于低频词生成的词向量效果会更好。因为它们的 n-gram 可以和其它词共享。
2. 对于训练词库之外的单词，仍然可以构建它们的词向量。我们可以叠加它们的字符级 n-gram 向量。
### （2）模型架构
之前提到过，fastText 模型架构和 word2vec 的 CBOW 模型架构非常相似。下面是 fastText 模型架构图：
![](<assets/1698918124215.png>)

注意：此架构图没有展示词向量的训练过程。可以看到，和 CBOW 一样，fastText 模型也只有三层：输入层、隐含层、输出层（Hierarchical Softmax），输入都是多个经向量表示的单词，输出都是一个特定的 target，隐含层都是对多个词向量的叠加平均。不同的是，CBOW 的输入是目标单词的上下文，fastText 的输入是多个单词及其 n-gram 特征，这些特征用来表示单个文档；CBOW 的输入单词被 onehot 编码过，fastText 的输入特征是被 embedding 过；CBOW 的输出是目标词汇，fastText 的输出是文档对应的类标。

值得注意的是，fastText 在输入时，将单词的字符级别的 n-gram 向量作为额外的特征；在输出时，fastText 采用了分层 Softmax，大大降低了模型训练时间。这两个知识点在前文中已经讲过，这里不再赘述。
fastText 相关公式的推导和 CBOW 非常类似，这里也不展开了。

## （3）核心思想

现在抛开那些不是很讨人喜欢的公式推导，来想一想 fastText 文本分类的核心思想是什么？

仔细观察模型的后半部分，即从隐含层输出到输出层输出，会发现它就是一个 softmax 线性多类别分类器，分类器的输入是一个用来表征当前文档的向量；模型的前半部分，即从输入层输入到隐含层输出部分，主要在做一件事情：生成用来表征文档的向量。那么它是如何做的呢？叠加构成这篇文档的所有词及 n-gram 的词向量，然后取平均。叠加词向量背后的思想就是传统的词袋法，即将文档看成一个由词构成的集合。

于是 fastText 的核心思想就是：将整篇文档的词及 n-gram 向量叠加平均得到文档向量，然后使用文档向量做 softmax 多分类。这中间涉及到两个技巧：字符级 n-gram 特征的引入以及分层 Softmax 分类。

# CNN model
### textcnn
![[Pasted image 20231102145520.png]]

思路：定义一个二维卷积核长度自己指定， 宽度为嵌入后向量的大小。将卷积的结果进行池化后拼成一个向量即可。

## DCNN(Dynamic Convolution Neural Network)
https://zhuanlan.zhihu.com/p/266842306
感觉没什么东西 就提出了几个操作。
![[Pasted image 20231102211027.png]]
## DPCNN
![[Pasted image 20231102221608.png]]
上图中的ShallowCNN指TextCNN。DPCNN的核心改进如下：

1. 在Region embedding时不采用CNN那样加权卷积的做法，而是**对n个词进行pooling后再加个1x1的卷积**，因为实验下来效果差不多，且作者认为前者的表示能力更强，容易过拟合
2. 使用1/2池化层，用size=3 stride=2的卷积核，直接**让模型可编码的sequence长度翻倍**（自己在纸上画一下就get啦）
3. 残差链接，参考ResNet，减缓梯度弥散问题

凭借以上一些精妙的改进，DPCNN相比TextCNN有1-2个百分点的提升。



# RNN model

## text rnn

```python
# %%
# code by Tae Hwan Jung @graykode
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def make_batch():
    input_batch = []
    target_batch = []

    for sen in sentences:
        word = sen.split()  # space tokenizer
        input = [word_dict[n] for n in word[:-1]]  # create (1~n-1) as input
        target = word_dict[word[-1]]  # create (n) as target, We usually call this 'casual language model'

        input_batch.append(np.eye(n_class)[input])
        target_batch.append(target)

    return input_batch, target_batch

class TextRNN(nn.Module):
    def __init__(self):
        super(TextRNN, self).__init__()
        self.rnn = nn.RNN(input_size=n_class, hidden_size=n_hidden)
        self.W = nn.Linear(n_hidden, n_class, bias=False)
        self.b = nn.Parameter(torch.ones([n_class]))

    def forward(self, hidden, X):
        X = X.transpose(0, 1) # X : [n_step, batch_size, n_class]
        outputs, hidden = self.rnn(X, hidden)
        # outputs : [n_step, batch_size, num_directions(=1) * n_hidden]
        # hidden : [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        outputs = outputs[-1] # [batch_size, num_directions(=1) * n_hidden]
        model = self.W(outputs) + self.b # model : [batch_size, n_class]
        return model

if __name__ == '__main__':
    n_step = 2 # number of cells(= number of Step)
    n_hidden = 5 # number of hidden units in one cell

    sentences = ["i like dog", "i love coffee", "i hate milk"]

    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    number_dict = {i: w for i, w in enumerate(word_list)}
    n_class = len(word_dict)
    batch_size = len(sentences)

    model = TextRNN()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    input_batch, target_batch = make_batch()
    input_batch = torch.FloatTensor(input_batch)
    target_batch = torch.LongTensor(target_batch)

    # Training
    for epoch in range(5000):
        optimizer.zero_grad()

        # hidden : [num_layers * num_directions, batch, hidden_size]
        hidden = torch.zeros(1, batch_size, n_hidden)
        # input_batch : [batch_size, n_step, n_class]
        output = model(hidden, input_batch)

        # output : [batch_size, n_class], target_batch : [batch_size] (LongTensor, not one-hot)
        loss = criterion(output, target_batch)
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    input = [sen.split()[:2] for sen in sentences]

    # Predict
    hidden = torch.zeros(1, batch_size, n_hidden)
    predict = model(hidden, input_batch).data.max(1, keepdim=True)[1]
    print([sen.split()[:2] for sen in sentences], '->', [number_dict[n.item()] for n in predict.squeeze()])
```