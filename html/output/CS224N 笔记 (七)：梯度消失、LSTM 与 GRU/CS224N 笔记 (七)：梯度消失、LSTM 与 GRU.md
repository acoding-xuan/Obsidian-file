---
url: https://zhuanlan.zhihu.com/p/63557635
title: CS224N 笔记 (七)：梯度消失、LSTM 与 GRU
date: 2023-10-17 16:44:54
tag: 
summary: 
---
这一讲主要研究 RNN 中梯度消失以及梯度爆炸问题，以及为解决梯度消失而设计的 RNN 模型的变种如 LSTM，GRU 等模型。

## **梯度消失**

RNN 理论上是可以捕捉较早的历史信息，但是由于 Vanishing Gradient 的问题会导致远程信息无法有效的被捕捉到。

RNN 的输入、输出及 hidden state 的关系有如下的公式表示：

$h_t=f(Wh_{t-1}+W_xx_t) \\ \hat y_t=softmax(Uh_t)$h_t=f(Wh_{t-1}+W_xx_t) \\ \hat y_t=softmax(Uh_t)

并且其损失函数为

$J=\frac{1}{T}\sum_{t=1}^TJ_t \\ J_t=\sum_{j=1}^{|V|}y_{t,j}log\hat y_{t,j}$J=\frac{1}{T}\sum_{t=1}^TJ_t \\ J_t=\sum_{j=1}^{|V|}y_{t,j}log\hat y_{t,j}

所以损失函数相对于 W 的梯度为

$\frac{\partial J}{\partial W} = \sum_{t=1}^T\frac{\partial J_t}{\partial W} \\ \frac{\partial J_t}{\partial W} =\sum_{k=1}^t\frac{\partial J_t}{\partial y_t}\frac{\partial y_t}{\partial h_t}\frac{\partial h_t}{\partial h_k}\frac{\partial h_k}{\partial W}$\frac{\partial J}{\partial W} = \sum_{t=1}^T\frac{\partial J_t}{\partial W} \\ \frac{\partial J_t}{\partial W} =\sum_{k=1}^t\frac{\partial J_t}{\partial y_t}\frac{\partial y_t}{\partial h_t}\frac{\partial h_t}{\partial h_k}\frac{\partial h_k}{\partial W}

其中 $\frac{\partial h_t}{\partial h_k}=\prod_{j=k+1}^{t}\frac{\partial h_j}{\partial h_{j-1}}=\prod_{j=k+1}^{t}W^{t-k}\times diag[f'(h_{j-1})]$\frac{\partial h_t}{\partial h_k}=\prod_{j=k+1}^{t}\frac{\partial h_j}{\partial h_{j-1}}=\prod_{j=k+1}^{t}W^{t-k}\times diag[f'(h_{j-1})]

假设矩阵 W 的最大的本征值也小于 1，则 t-k 越大即其相距越远，其梯度会呈指数级衰减，这一问题被称作 vanishing gradient 梯度消失，它导致我们无法分辨 t 时刻与 k 时刻究竟是数据本身毫无关联还是由于梯度消失而导致我们无法捕捉到这一关联。这就导致了我们只能学习到近程的关系而不能学习到远程的关系，会影响很多语言处理问题的准确度。

## **梯度爆炸**

与梯度消失类似，由于存在矩阵 W 的指数项，如果其本征值大于 1，则随着时间越远该梯度指数级增大，这又造成了另一问题 exploding gradient 梯度爆炸，这就会造成当我们步进更新时，步进过大而越过了极值点，一个简单的解决方案是 gradient clipping，即如果梯度大于某一阈值，则在 SGD 新一步更新时按比例缩小梯度值，即我们仍然向梯度下降的方向行进但是其步长缩小，其伪代码如下：

![](<assets/1697532294492.png>)

不进行 clipping 与进行 clipping 的学习过程对比如下

![](<assets/1697532294655.png>)

可见左图中由于没有进行 clipping，步进长度过大，导致损失函数更新到一个不理想的区域，而右图中进行 clipping 后每次步进减小，能更有效的达到极值点区域。

梯度爆炸问题我们可以通过简单的 gradient clipping 来解决，那对于梯度消失问题呢？其基本思路是我们设置一些存储单元来更有效的进行长程信息的存储，LSTM 与 GRU 都是基于此基本思想设计的。

## **LSTM**

LSTM, 全称 Long Short Term Memory。其基本思路是除了 hidden state _t$h_t$h_t 之外，引入 cell state $c_t$c_t 来存储长程信息，LSTM 可以通过控制 gate 来擦除，存储或写入 cell state。

LSTM 的公式如下：

![](<assets/1697532294790.png>)

New Memory cell $\tilde c_t$\tilde c_t 通过将新的输入 $x_t$x_t 与代表之前的 context 的 hidden state $h_{t-1}$h_{t-1} 结合生成新的 memory。

Input gate $i_t$i_t 决定了新的信息是否有保留的价值。

Forget gate $f_t$f_t 决定了之前的 cell state 有多少保留的价值。

Final memory cell $c_t$c_t 通过将 forget gate 与前 memory cell 作元素积得到了前 memory cell 需要传递下去的信息，并与 input gate 和新的 memory cell 的元素积求和。极端情况是 forget gate 值为 1，则之前的 memory cell 的全部信息都会传递下去，使得梯度消失问题不复存在。

output gate $o_t$o_t 决定了 memory cell 中有多少信息需要保存在 hidden state 中。

## **GRU**

GRU(gated recurrent unit) 可以看做是将 LSTM 中的 forget gate 和 input gate 合并成了一个 update gate，即下式中的 $z_{t}$z_{t}，同时将 cell state 也合并到 hidden state 中。

![](<assets/1697532295123.png>)

Update Gate 决定了前一时刻的 hidden state 有多少可以保留到下一时刻的 hidden state。reset gate 决定了在产生新的 memory $\tilde h_{t}$\tilde h_{t} 时前一时刻 hidden state 有多少贡献。

虽然还存在 RNN 的很多其他变种，但是 LSTM 与 RNN 是最广泛应用的。最大的区别就是 GRU 有更少的参数，更便于计算，对于模型效果方面，两者类似。通常我们可以从 LSTM 开始，如果需要提升效率的话再准换成 GRU。

## **延伸**

梯度消失仅仅局限于 RNN 结构吗？实际上对于深度神经网络，尤其层数较多时，由于链式法则以及非线性函数的选择，梯度消失问题也会出现，导致底层的 layer 学习速度很慢，只不过由于 RNN 中由于不断的乘以相同的权重矩阵导致其梯度消失问题更显著。

对于非 RNN 的深度神经网络如前馈神经网络与卷积神经网络，为了解决梯度消失问题，通常可以引入更直接的联结使得梯度更容易传递。

例如在 ResNet 中，通过引入与前层输入一样的联结使得信息更有效的流动，这使得我们可以能够训练更深层的模型。这一联结被称作 Residual Connection 或 Skip Connection。

![](<assets/1697532295566.png>)

更进一步的，在 DenseNet 中，我们可以将几层之间全部建立直接联结，形成 dense connection。

![](<assets/1697532295920.png>)

下一讲中，我们还会看到更有效的捕捉长程信息的机制——Attention 模型, to be continued。

## **参考资料**

第七讲讲义 [http://web.stanford.edu/class/cs224n/slides/cs224n-2019-lecture07-fancy-rnn.pdf](https://link.zhihu.com/?target=http%3A//web.stanford.edu/class/cs224n/slides/cs224n-2019-lecture07-fancy-rnn.pdf)

第七讲视频 [https://youtu.be/QEw0qEa0E50](https://link.zhihu.com/?target=https%3A//youtu.be/QEw0qEa0E50)