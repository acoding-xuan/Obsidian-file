> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [blog.csdn.net](https://blog.csdn.net/m0_67505927/article/details/123209347)

[Transformer](https://so.csdn.net/so/search?q=Transformer&spm=1001.2101.3001.7020) 是什么呢？
----------------------------------------------------------------------------------------

\qquad Transformer 最早起源于论文 [Attention is all your need](https://arxiv.org/abs/1706.03762)，是谷歌云 TPU 推荐的参考模型。  
\qquad 目前，在 NLP 领域当中，主要存在三种特征处理器——CNN、RNN 以及 Transformer，当前 Transformer 的流行程度已经大过 CNN 和 RNN，它抛弃了传统 CNN 和 RNN 神经网络，整个网络结构完全由 Attention 机制以及前馈神经网络组成。首先给出一个来自原论文的 Transformer 整体架构图方便之后回顾。  
![](https://img-blog.csdnimg.cn/a921f339611841c5a0900fa360c12e58.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5ZOq5pyJ54Gs5bmz5Yeh77yf,size_20,color_FFFFFF,t_70,g_se,x_16) \qquad 上图中的 Transformer 可以说是一个使用 “self attention” 的 Seq2seq 模型。  
那么要想了解 Transformer，就必须先了解 "self attention"。  
\qquad 如果给出一个 Sequence 要处理，最常想到的可能就是 RNN 了，如下图 1 所示。RNN 被经常使用在输入是有序列信息的模型中，但它也存在一个问题——它不容易被 “平行化”。那么“平行化” 是什么呢？  
\qquad 比如说在 RNN 中 a1,a2,a3,a4 就是输入，b1,b2,b3,b4 就是输出。对于单向 RNN，如果你要输出 b3 那么你需要把 a1,a2,a3 都输入并运算了才能得到；对于双向 RNN，如果你要输出任何一个 bi, 那么你要把所有的 ai 都输入并运算过才能得到。它们无法同时进行运算得出 b1,b2,b3,b4。

![](https://img-blog.csdnimg.cn/7ec0f568d3f147ecb12d1e032b15f54f.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5ZOq5pyJ54Gs5bmz5Yeh77yf,size_8,color_FFFFFF,t_70,g_se,x_16#pic_center)  
\qquad 而针对 RNN 无法 “平行化” 这个问题，有人提出了使用 CNN 来取代 RNN，如下图所示。输入输出依然为 ai、bi。它利用一个个 Filter（如下图黄色三角形）（我的理解是类似于计网的滑动窗口协议）去得出相应的输出，比如 b1 是通过 a1,a2 一起得出；b2 是通过 a1,a2,a3 得出。可能会存在一个疑问——这样不就只考虑临近输入的信息，而对长距离信息没有考虑了？  
\qquad 当然不是这样，它可以考虑长距离信息的输入，只需要在输出 bi 上再叠加一层 Filters 就能涵盖更多的信息，如下图黄色三角形，所有输入 ai 运算得出 b1,b2,b3 作为该层的输入。所以说只要你叠加的层数够多，它可以包含你所有的输入信息。  
\qquad 回到咱们对 “平行化” 问题的解答：使用 CNN 是可以做到 “平行化” 的，下图中每一个蓝色的三角形，并不用等前面的三角形执行完才能执行，它们可以同时进行运算。  
![](https://img-blog.csdnimg.cn/273791fd9e5f41e58d0944b11527f9d1.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5ZOq5pyJ54Gs5bmz5Yeh77yf,size_11,color_FFFFFF,t_70,g_se,x_16#pic_center)

### self attention

\qquad self attention 模型输入的 xi 先做 embedding 得到 ai，每一个 xi 都分别乘上三个不同的 w 得到 q、k、v。  
![](https://img-blog.csdnimg.cn/0388af537b884f1680914fab1d6536c8.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5ZOq5pyJ54Gs5bmz5Yeh77yf,size_19,color_FFFFFF,t_70,g_se,x_16)  
其中： \qquad \qquad \qquad \qquad   a i = W x i \ a^i=Wx^i  ai=Wxi  
\qquad \qquad \qquad \qquad \qquad   q i = W q a i \ q^i=W^qa^i  qi=Wqai  
\qquad \qquad \qquad \qquad \qquad   k i = W k a i \ k^i=W^ka^i  ki=Wkai  
\qquad \qquad \qquad \qquad \qquad   v i = W v a i \ v^i=W^va^i  vi=Wvai  
拿每个 qi 去对每个 ki 做点积得到   a 1 , i \ a_{1,i}  a1,i​，其中 d 是 q 和 k 的维度。  
\qquad \qquad \qquad \qquad \qquad   a 1 , i = q 1 ⋅ k i / d \ a_{1,i}=q^1·k^i/{\sqrt d}  a1,i​=q1⋅ki/d ​  
![](https://img-blog.csdnimg.cn/483b2a07da354b09b82152f7fab60b3c.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5ZOq5pyJ54Gs5bmz5Yeh77yf,size_20,color_FFFFFF,t_70,g_se,x_16)  
再把   a 1 , i \ a_{1,i}  a1,i​经过一个 Soft-max 之后得到 a ^ 1 , i \hat a_{1,i} a^1,i​  
a ^ 1 , i = e x p ( a 1 , i ) / ∑ j e x p ( a 1 , j ) \hat a_{1,i} =exp(a_{1,i})/\sum_{j} exp(a_{1,j}) a^1,i​=exp(a1,i​)/j∑​exp(a1,j​)  
![](https://img-blog.csdnimg.cn/91ce83c58c0044f5a14e89b10fc0a662.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5ZOq5pyJ54Gs5bmz5Yeh77yf,size_20,color_FFFFFF,t_70,g_se,x_16)

\qquad 接下来把 a ^ 1 , j \hat a_{1,j} a^1,j​与对应的 v j v^j vj 分别做乘积最后求和得出第一个输出 b 1 b_1 b1​，同理可得到所有 b i b_i bi​。  
b 1 = ∑ i n a ^ 1 , i v i b^1 =\sum_{i}^n \hat a_{1,i}v^i b1=i∑n​a^1,i​vi  
![](https://img-blog.csdnimg.cn/d07e64c8b9a9419f934dec3e39da4a1c.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5ZOq5pyJ54Gs5bmz5Yeh77yf,size_19,color_FFFFFF,t_70,g_se,x_16)

\qquad 那么到这里就可以看出输出 b1 是综合了所有的输入 xi 信息，同时这样做的优势在于——当 b1 只需要考虑局部信息的时候（比如重点关注 x1,x2 就行了），那么它可以让 a ^ 1 , 3 \hat a_{1,3} a^1,3​和 a ^ 1 , 4 \hat a_{1,4} a^1,4​输出的值为 0 就行了。

### 那么 self attention 是这么做平行化的呢？

咱们复习一下前面说到的 q、k、v 的计算：  
\qquad \qquad \qquad \qquad \qquad   q i = W q a i \ q^i=W^qa^i  qi=Wqai  
\qquad \qquad \qquad \qquad \qquad   k i = W k a i \ k^i=W^ka^i  ki=Wkai  
\qquad \qquad \qquad \qquad \qquad   v i = W v a i \ v^i=W^va^i  vi=Wvai  
\qquad 因为   q 1 = w q a 1 \ q^1=w^qa^1  q1=wqa1，那么根据矩阵运算原理，我们将   a 1 、 a 2 、 a 3 、 a 4 \ a^1、a^2、a^3、a^4  a1、a2、a3、a4 串起来作为一个矩阵 I 与   w q \ w^q  wq 相乘可以得到   q 1 、 q 2 、 q 3 、 q 4 \ q^1、q^2、q^3、q^4  q1、q2、q3、q4 构成的矩阵 Q。同理可得   k i 、 v i \ k^i、v^i  ki、vi 的矩阵 K、V。  
![](https://img-blog.csdnimg.cn/b535a5612000464ba06839c60f88257e.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5ZOq5pyJ54Gs5bmz5Yeh77yf,size_20,color_FFFFFF,t_70,g_se,x_16)

然后我们再回忆观察一下   a 1 , i \ a_{1,i}  a1,i​的计算过程 (为方便理解，此处省略 d \sqrt d d ​)：  
\qquad \qquad \qquad   a 1 , 1 = k 1 ⋅ q 1 \ a_{1,1}=k^1·q^1  a1,1​=k1⋅q1 \qquad   a 1 , 2 = k 2 ⋅ q 1 \ a_{1,2}=k^2·q^1  a1,2​=k2⋅q1  
\qquad \qquad \qquad   a 1 , 3 = k 3 ⋅ q 1 \ a_{1,3}=k^3·q^1  a1,3​=k3⋅q1 \qquad   a 1 , 4 = k 4 ⋅ q 1 \ a_{1,4}=k^4·q^1  a1,4​=k4⋅q1  
\qquad 我们可以发现计算都是用   q 1 \ q^1  q1 去乘以每个   k i \ k^i  ki 得出   a 1 , i \ a_{1,i}  a1,i​，那么我们将   k i \ k^i  ki 叠加起来与   q 1 \ q^1  q1 相乘得到一列向量   a 1 , i \ a_{1,i}  a1,i​(i=1,2,3,4)。然后你再加上所有的   q i \ q^i  qi 就可以得到整个   a i , j \ a_{i,j}  ai,j​矩阵。最后对   a i , j \ a_{i,j}  ai,j​的每一列做一个 soft-max 就得到 a ^ i , j \hat a_{i,j} a^i,j​矩阵。  
![](https://img-blog.csdnimg.cn/c7ba286250484069ad40c03c6614cef4.png)

最后再把 a ^ i , j \hat a_{i,j} a^i,j​与所有   v i \ v^i  vi 构成的矩阵 V 相乘即可得到输出。  
![](https://img-blog.csdnimg.cn/1b2f43d5da954f9690d081a05d91274c.png)

\qquad 在这里我们对输入 I 到输出 O 之间做的事情做一个总结：我们先用 I 分别乘上对应的   W i \ W^i  Wi 得到矩阵 Q,K,V，再把 Q 与   K T \ K^T  KT 相乘得到矩阵 A，再对 A 做 soft-max 处理得到矩阵 KaTeX parse error: Expected group after '^' at position 7: \hat A^̲，最后再将 KaTeX parse error: Expected group after '^' at position 7: \hat A^̲与 V 相乘得到输出结果 O。整个过程都是进行矩阵乘法，都可以使用 GPU 加速。  
![](https://img-blog.csdnimg.cn/3aaef73ad9e240958982f3d798b3aea5.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5ZOq5pyJ54Gs5bmz5Yeh77yf,size_20,color_FFFFFF,t_70,g_se,x_16)

### self-attention 的变形——Multi-head Self-attention

\qquad Multi-head Self-attention 跟 self-attention 一样都会生成 q、k、v，但是 Multi-head Self-attention 会再将 q、k、v 分裂出多个   q 1 , 2 \ q^{1,2}  q1,2（这里举例分裂成两个），然后它也将 q 跟 k 去进行相乘计算，但是只跟其对应的 k、v 进行计算，比如   q 1 , 1 \ q^{1,1}  q1,1 只会与   k 1 , 1 \ k^{1,1}  k1,1、   k 2 , 1 \ k^{2,1}  k2,1 进行运算，然后一样的乘以对应的 v 得到输出   b 1 , 1 \ b^{1,1}  b1,1。  
\qquad \qquad \qquad   q 1 , 1 = W q , 1 q 1 \ q^{1,1}=W^{q,1}q^1  q1,1=Wq,1q1 \qquad \qquad   q 1 , 2 = W q , 2 q 1 \ q^{1,2}=W^{q,2}q^1  q1,2=Wq,2q1  
![](https://img-blog.csdnimg.cn/165e45cebc51478e8782bb4173313590.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5ZOq5pyJ54Gs5bmz5Yeh77yf,size_20,color_FFFFFF,t_70,g_se,x_16)  
\qquad 对于   b i , 1 \ b^{i,1}  bi,1 再进行一步处理就得到我们在 self-attention 所做的一步骤的输出   b i \ b^i  bi。  
![](https://img-blog.csdnimg.cn/a04eb88c77624f7c9cf559f573456a24.png)

**那么这个 Multi-head Self-attention 设置多个 q,k,v 有什么好处呢？**  
\qquad 举例来说，有可能不同的 head 关注的点不一样，有一些 head 可能只关注局部的信息，有一些 head 可能想要关注全局的信息，有了多头注意里机制后，每个 head 可以各司其职去做自己想做的事情。

**Positional Encoding**  
\qquad 根据前面 self-attention 介绍中，我们可以知道其中的运算是没有去考虑位置信息，而我们希望是把输入序列每个元素的位置信息考虑进去，那么就要在   a i \ a^i  ai 这一步还有加上一个位置信息向量   e i \ e^i  ei，每个   e i \ e^i  ei 都是其对应位置的独特向量。——   e i \ e^i  ei 是通过人工手设（不是学习出来的）。  
![](https://img-blog.csdnimg.cn/20ac77bdee38494facf2fc29d7832b12.png#pic_center)  
最后挂上一张来自原论文的效果图，体验一下 transformer 的强大：  
![](https://img-blog.csdnimg.cn/81277f5dd4954b4d956826a4098ca825.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5ZOq5pyJ54Gs5bmz5Yeh77yf,size_20,color_FFFFFF,t_70,g_se,x_16)