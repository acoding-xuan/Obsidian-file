---
url: https://blog.csdn.net/huang1024rui/article/details/112170547
title: 机器学习 - 白板推导系列 (二)- 数学基础笔记_rpml 边缘概率 配方法_Paul-Huang 的博客 - CSDN 博客
date: 2023-10-09 00:41:04
tag: 
summary: 
---
## 1. 高斯分布

*   假设有 N个样本，每个样本都是 p 维向量的数据：  
    $X N × p = ( x 1 , x 2 , ⋯   , x N ) T$ , $x i = ( x i 1 , x i 2 , ⋯   , x i p ) T$  $X_{N\times p}=(x_{1},x_{2},\cdots,x_{N})^{T},x_{i}=(x_{i1},x_{i2},\cdots,x_{ip})^{T} XN×p​=(x1​,x2​,⋯,xN​)T,xi​=(xi1​,xi2​,⋯,xip​)T$  
    $且 x i   i i d N ( μ , Σ ) x_i\mathop{~}\limits _{iid} N(\mu,\Sigma) xi​iid ​N(μ,Σ), 且 θ = ( μ , Σ ) \theta = (\mu,\Sigma) θ=(μ,Σ)。$
*   一般地，高斯分布的概率密度函数写为：  
* $$
    p ( x ∣ μ , Σ ) = 1 ( 2 π ) p / 2 ∣ Σ ∣ 1 / 2 e − 1 2 ( x − μ ) T Σ − 1 ( x − μ ) p(x|\mu,\Sigma)=\frac{1}{(2\pi)^{p/2}|\Sigma|^{1/2}}e^{-\frac{1}{2}(x-\mu)^{T}\Sigma^{-1}(x-\mu)} p(x∣μ,Σ)=(2π)p/2∣Σ∣1/21​e−21​(x−μ)TΣ−1(x−μ)
$$
### 1.1 一维高斯分布下的估计

θ = ( μ , Σ ) = ( μ , σ 2 ) , \theta=(\mu,\Sigma)=(\mu,\sigma^{2}), θ=(μ,Σ)=(μ,σ2), 一维高斯分布下的 MLE：  
θ M L E = a r g m a x θ ( log ⁡ p ( X ∣ θ ) ) = i i d a r g m a x θ ( ∑ i = 1 N log ⁡ p ( x i ∣ θ ) ) = a r g m a x θ ( ∑ i = 1 N log ⁡ 1 2 π σ exp ⁡ ( − ( x i − μ ) 2 / 2 σ 2 ) )

$$\begin{array}{r} \theta_{MLE}&=\mathop{argmax}\limits _{\theta}(\log p(X|\theta))\mathop{=}\limits _{iid}\mathop{argmax}\limits _{\theta}(\sum\limits _{i=1}^{N}\log p(x_{i}|\theta))\\ &=\mathop{argmax}\limits _{\theta}(\sum\limits _{i=1}^{N}\log\frac{1}{\sqrt{2\pi}\sigma}\exp(-(x_{i}-\mu)^{2}/2\sigma^{2}))\end{array}$$

θMLE​​=θargmax​(logp(X∣θ))iid=​θargmax​(i=1∑N​logp(xi​∣θ))=θargmax​(i=1∑N​log2π ​σ1​exp(−(xi​−μ)2/2σ2))​

#### 1.1.1 求一维高斯分布下的极大似然估计

极大似然估计是一种用来在给定观察数据下估计所需参数的技术。

比如，如果已知人口分布遵从正太分布，但是均值和方差未知， MLE（maximum likelihood estimation）可以利用有限的样本来估计这些参数。

*   首先对 μ \mu μ 的极值可以得到 ：  
    μ M L E = a r g m a x μ log ⁡ p ( X ∣ θ ) = a r g m a x μ ∑ i = 1 N ( x i − μ ) 2 \mu_{MLE}=\mathop{argmax}\limits _{\mu}\log p(X|\theta)=\mathop{argmax}\limits _{\mu}\sum\limits _{i=1}^{N}(x_{i}-\mu)^{2} μMLE​=μargmax​logp(X∣θ)=μargmax​i=1∑N​(xi​−μ)2  
    于是求得 μ M L E = 1 N ∑ i = 1 N x i \color{red}\mu_{MLE}=\frac{1}{N}\sum\limits _{i=1}^{N}x_{i} μMLE​=N1​i=1∑N​xi​：  
    ∂ ∂ μ ∑ i = 1 N ( x i − μ ) 2 = 0 ⟶ μ M L E = 1 N ∑ i = 1 N x i \frac{\partial}{\partial\mu}\sum\limits _{i=1}^{N}(x_{i}-\mu)^{2}=0\longrightarrow\mu_{MLE}=\frac{1}{N}\sum\limits _{i=1}^{N}x_{i} ∂μ∂​i=1∑N​(xi​−μ)2=0⟶μMLE​=N1​i=1∑N​xi​
    
*   其次对 θ \theta θ 中的另一个参数 σ \sigma σ ，有：  
    σ M L E = a r g m a x σ log ⁡ p ( X ∣ θ ) = a r g m a x σ ∑ i = 1 N [ − log ⁡ σ − 1 2 σ 2 ( x i − μ ) 2 ] = a r g m i n σ ∑ i = 1 N [ log ⁡ σ + 1 2 σ 2 ( x i − μ ) 2 ]
    
    $$\begin{array}{r}\sigma_{MLE}=\mathop{argmax}\limits _{\sigma}\log p(X|\theta)&=\mathop{argmax}\limits _{\sigma}\sum\limits _{i=1}^{N}[-\log\sigma-\frac{1}{2\sigma^{2}}(x_{i}-\mu)^{2}]\\ &=\mathop{argmin}\limits _{\sigma}\sum\limits _{i=1}^{N}[\log\sigma+\frac{1}{2\sigma^{2}}(x_{i}-\mu)^{2}]\end{array}$$
    
     σMLE​=σargmax​logp(X∣θ)​=σargmax​i=1∑N​[−logσ−2σ21​(xi​−μ)2]=σargmin​i=1∑N​[logσ+2σ21​(xi​−μ)2]​  
    于是求得 σ M L E 2 = 1 N ∑ i = 1 N ( x i − μ ) 2 \color{red}\sigma_{MLE}^{2}=\frac{1}{N}\sum\limits _{i=1}^{N}(x_{i}-\mu)^{2} σMLE2​=N1​i=1∑N​(xi​−μ)2：  
    ∂ ∂ σ ∑ i = 1 N [ log ⁡ σ + 1 2 σ 2 ( x i − μ ) 2 ] = 0 ⟶ σ M L E 2 = 1 N ∑ i = 1 N ( x i − μ ) 2 \frac{\partial}{\partial\sigma}\sum\limits _{i=1}^{N}[\log\sigma+\frac{1}{2\sigma^{2}}(x_{i}-\mu)^{2}]=0\longrightarrow\sigma_{MLE}^{2}=\frac{1}{N}\sum\limits _{i=1}^{N}(x_{i}-\mu)^{2} ∂σ∂​i=1∑N​[logσ+2σ21​(xi​−μ)2]=0⟶σMLE2​=N1​i=1∑N​(xi​−μ)2
    

### 1.1.2 一维高斯分布极值的无偏估计

*   **无偏估计的意义**  
    所谓总体参数估计量的无偏性指的是，基于不同的样本，使用该估计量可算出多个估计值，但它们的平均值等于被估参数的真值。
    
    1. 在某些场合下，**无偏性的要求是有实际意义的**。例如，假设在某厂商与某销售商之间存在长期的供货关系，则在对产品出厂质量检验方法的选择上，采用随机抽样的方法来估计次品率就很公平。这是因为从长期来看，这种估计方法是无偏的。比如这一次所估计出来的次品率实际上偏高，厂商吃亏了；但下一次的估计很可能偏低，厂商的损失就可以补回来。由于双方的交往会长期多次发生，这时采用无偏估计，总的来说可以达到互不吃亏的效果。  
    2. 不过，在某些场合中，**无偏性的要求毫无实际意义**。这里又有两种情况：
    
    *   一种情况是在某些场合中不可能发生多次抽样。例如，假设在某厂商和某销售商之间只会发生一次买卖交易，此后不可能再发生第二次商业往来。这时双方谁也吃亏不起，这里就没有什么 “平均” 可言。
    *   另一种情况则是估计误差不可能相互补偿，因此 “平均” 不得。例如，假设需要通过试验对一个批量的某种型号导弹的系统误差做出估计。这个时候，既使我们的估计的确做到了无偏，但如果这一批导弹的系统误差实际上要么偏左，要么偏右，结果只能是大部分导弹都不能命中目标，不可能存在 “偏左” 与“偏右”相互抵消，从而 “平均命中” 的概念。
    
*   μ M L E \mu_{MLE} μMLE​无偏估计：  
    E D [ μ M L E ] = E D [ 1 N ∑ i = 1 N x i ] = 1 N ∑ i = 1 N E D [ x i ] = μ \mathbb{E}_{\mathcal{D}}[\mu_{MLE}]=\mathbb{E}_{\mathcal{D}}[\frac{1}{N}\sum\limits _{i=1}^{N}x_{i}]=\frac{1}{N}\sum\limits _{i=1}^{N}\mathbb{E}_{\mathcal{D}}[x_{i}]=\mu ED​[μMLE​]=ED​[N1​i=1∑N​xi​]=N1​i=1∑N​ED​[xi​]=μ  
    因此对数据集求期望时， μ M L E = E D [ μ M L E ] \mu_{MLE}=\mathbb{E}_{\mathcal{D}}[\mu_{MLE}] μMLE​=ED​[μMLE​] ， μ M L E \mu_{MLE} μMLE​是**无偏差的**。
    
*   σ M L E \sigma_{MLE} σMLE​的无偏估计：  
    对 σ M L E \sigma_{MLE} σMLE​ 求 期望的时候由于使用了单个数据集的 μ M L E \mu_{MLE} μMLE​，  
    E D [ σ M L E 2 ] = E D [ 1 N ∑ i = 1 N ( x i − μ M L E ) 2 ] = E D [ 1 N ∑ i = 1 N ( x i 2 − 2 x i μ M L E + μ M L E 2 ) = E D [ 1 N ∑ i = 1 N x i 2 − μ M L E 2 ] = E D [ 1 N ∑ i = 1 N x i 2 − μ 2 + μ 2 − μ M L E 2 ] = E D [ 1 N ∑ i = 1 N ( x i 2 − μ 2 ) ] − E D [ μ M L E 2 − μ 2 ] = σ 2 − ( E D [ μ M L E 2 ] − μ 2 ) = σ 2 − ( E D [ μ M L E 2 ] − E D 2 [ μ M L E ] ) = σ 2 − V a r [ μ M L E ] = σ 2 − V a r [ 1 N ∑ i = 1 N x i ] = σ 2 − 1 N 2 ∑ i = 1 N V a r [ x i ] = N − 1 N σ 2
    
    $$\begin{array}{l}\mathbb{E}_{\mathcal{D}}[\sigma_{MLE}^{2}]&=\mathbb{E}_{\mathcal{D}}[\frac{1}{N}\sum\limits _{i=1}^{N}(x_{i}-\mu_{MLE})^{2}]=\mathbb{E}_{\mathcal{D}}[\frac{1}{N}\sum\limits _{i=1}^{N}(x_{i}^{2}-2x_{i}\mu_{MLE}+\mu_{MLE}^{2})\\ &=\mathbb{E}_{\mathcal{D}}[\frac{1}{N}\sum\limits _{i=1}^{N}x_{i}^{2}-\mu_{MLE}^{2}]=\mathbb{E}_{\mathcal{D}}[\frac{1}{N}\sum\limits _{i=1}^{N}x_{i}^{2}-\mu^{2}+\mu^{2}-\mu_{MLE}^{2}]\\ &= \mathbb{E}_{\mathcal{D}}[\frac{1}{N}\sum\limits _{i=1}^{N}(x_{i}^{2}-\mu^{2})]-\mathbb{E}_{\mathcal{D}}[\mu_{MLE}^{2}-\mu^{2}]=\sigma^{2}-(\mathbb{E}_{\mathcal{D}}[\mu_{MLE}^{2}]-\mu^{2})\\ &=\sigma^{2}-(\mathbb{E}_{\mathcal{D}}[\mu_{MLE}^{2}]-\mathbb{E}_{\mathcal{D}}^{2}[\mu_{MLE}])=\sigma^{2}-\color{red}{Var[\mu_{MLE}]} \\ &=\sigma^{2}-Var[\frac{1}{N}\sum\limits _{i=1}^{N}x_{i}] =\sigma^{2}-\frac{1}{N^{2}}\sum\limits _{i=1}^{N}Var[x_{i}]=\frac{N-1}{N}\sigma^{2}\end{array}$$
    
     ED​[σMLE2​]​=ED​[N1​i=1∑N​(xi​−μMLE​)2]=ED​[N1​i=1∑N​(xi2​−2xi​μMLE​+μMLE2​)=ED​[N1​i=1∑N​xi2​−μMLE2​]=ED​[N1​i=1∑N​xi2​−μ2+μ2−μMLE2​]=ED​[N1​i=1∑N​(xi2​−μ2)]−ED​[μMLE2​−μ2]=σ2−(ED​[μMLE2​]−μ2)=σ2−(ED​[μMLE2​]−ED2​[μMLE​])=σ2−Var[μMLE​]=σ2−Var[N1​i=1∑N​xi​]=σ2−N21​i=1∑N​Var[xi​]=NN−1​σ2​  
    其中 V a r Var Var 表示方差；因此对数据集求方差时， σ M L E ≠ E D [ σ M L E ] \sigma_{MLE}\neq \mathbb{E}_{\mathcal{D}}[\sigma_{MLE}] σMLE​​=ED​[σMLE​] .
    

1.  σ M L E 2 = 1 N ∑ i = 1 N ( x i − μ ) 2 \color{red}\sigma_{MLE}^{2}=\frac{1}{N}\sum\limits _{i=1}^{N}(x_{i}-\mu)^{2} σMLE2​=N1​i=1∑N​(xi​−μ)2 不是**无偏差的**。 σ M L E 2 = 1 N − 1 ∑ i = 1 N ( x i − μ ) 2 \color{red}\sigma_{MLE}^{2}=\frac{1}{N-1}\sum\limits _{i=1}^{N}(x_{i}-\mu)^{2} σMLE2​=N−11​i=1∑N​(xi​−μ)2 才是**无偏估计**。
2.  通过推到 σ M L E 2 \sigma_{MLE}^{2} σMLE2​的估计偏小，这是因为：在抽样时，样本落在中间区域的概率大，所以抽样的数据离散程度小于总体，所以抽样方差小。

### 1.2 高维高斯分布与等高线是 “椭圆”

#### 1.2.1 高维高斯分布与马氏距离

*   **高维高斯分布**  
    假设数据 x ∈ R p x\in \mathbb{R}^{p} x∈Rp，是一个随机向量：  
    x ∼ i i d N ( μ , Σ ) = 1 ( 2 π ) D / 2 ∣ Σ ∣ 1 / 2 e x p ( − 1 2 ( x − μ ) T Σ − 1 ( x − μ ) ⏟ 二 次 型 ) x = ( x 1 x 2 ⋮ x p ) μ = ( μ 1 μ 2 ⋮ μ p ) Σ = [ σ 11 σ 12 ⋯ σ 1 p σ 21 σ 22 ⋯ σ 2 p ⋮ ⋮ ⋱ ⋮ σ p 1 σ p 2 ⋯ σ p p ] p × p
    
    $$\begin{array}{r}x\overset{iid}{\sim }N(\mu ,\Sigma )=\frac{1}{(2\pi )^{D/2}|\Sigma |^{1/2}}exp(-\frac{1}{2}\underset{二次型}{\underbrace{(x-\mu)^{T}\Sigma ^{-1}(x-\mu)}})\\ x=\begin{pmatrix} x_{1}\\ x_{2}\\ \vdots \\ x_{p} \end{pmatrix}\mu =\begin{pmatrix} \mu_{1}\\ \mu_{2}\\ \vdots \\ \mu_{p} \end{pmatrix}\Sigma = \begin{bmatrix} \sigma _{11}& \sigma _{12}& \cdots & \sigma _{1p}\\ \sigma _{21}& \sigma _{22}& \cdots & \sigma _{2p}\\ \vdots & \vdots & \ddots & \vdots \\ \sigma _{p1}& \sigma _{p2}& \cdots & \sigma _{pp} \end{bmatrix}_{p\times p}\end{array}$$ x∼iidN(μ,Σ)=(2π)D/2∣Σ∣1/21​exp(−21​二次型 (x−μ)TΣ−1(x−μ)​​)x=⎝⎜⎜⎜⎛​x1​x2​⋮xp​​⎠⎟⎟⎟⎞​μ=⎝⎜⎜⎜⎛​μ1​μ2​⋮μp​​⎠⎟⎟⎟⎞​Σ=⎣⎢⎢⎢⎡​σ11​σ21​⋮σp1​​σ12​σ22​⋮σp2​​⋯⋯⋱⋯​σ1p​σ2p​⋮σpp​​⎦⎥⎥⎥⎤​p×p​​  
    令 Δ = ( x − μ ) T Σ − 1 ( x − μ ) \Delta =(x-\mu)^{T}\Sigma ^{-1}(x-\mu ) Δ=(x−μ)TΣ−1(x−μ)。其中 Σ \Sigma Σ一般是半正定的，在本次证明中假设是 正 定 的 \color{red} 正定的 正定的，即所有的特征值都是正的，没有 0。
    
    1.  正定矩阵 (PD):  
        给定一个大小为 n × n n\times n n×n 的 实 对 称 矩 阵 A \color{red} 实对称矩阵 A 实对称矩阵 A，若对于任意长度为 n n n 的非零向量 X X X，有 X T A X > 0 X^TAX>0 XTAX>0 恒成立，则矩阵 A A A 是一个正定矩阵。
    2.  半正定矩阵 (PSD)  
        给定一个大小为 n × n n\times n n×n 的 实 对 称 矩 阵 A \color{red} 实对称矩阵 A 实对称矩阵 A ，若对于任意长度为 n n n 的非零向量 X X X，有 X T A X ≥ 0 X^TAX≥0 XTAX≥0 恒成立，则矩阵 A A A 是一个半正定矩阵。
    
*   **马氏距离**  
    ( x − μ ) T Σ − 1 ( x − μ ) \sqrt{(x-\mu)^{T}\Sigma ^{-1}(x-\mu)} (x−μ)TΣ−1(x−μ) ​为马氏距离
    
    x x x 与 μ \mu μ之间，当 Σ \Sigma Σ为 I I I( 单 位 矩 阵 \color{red} 单位矩阵 单位矩阵) 时马氏距离即为 欧 氏 距 离 \color{red} 欧氏距离 欧氏距离。
    

#### 1.2.2 高斯分布等高线为椭圆

*   Σ \Sigma Σ特征值分解
    
    *   任意的 N × N N \times N N×N 实对称矩阵都有 N N N 个线性无关的特征向量。
    *   这些特征向量都可以正交单位化而得到一组正交且模为 1 的向量。
    
    故实对称矩阵 Σ \Sigma Σ可被分解成 Σ = U Λ U T \Sigma=U\Lambda U^{T} Σ=UΛUT。其中 U U T = U T U = I ， Λ = d i a g ( λ i ) i = 1 , 2 , ⋯   , p ， U = ( u 1 , u 2 , ⋯   , u p ) p × p UU^{T}=U^{T}U=I，\underset{i=1,2,\cdots ,p}{\Lambda =diag(\lambda _{i})}，U=(u _{1},u _{2},\cdots ,u _{p})_{p\times p} UUT=UTU=I，i=1,2,⋯,pΛ=diag(λi​)​，U=(u1​,u2​,⋯,up​)p×p​。因此可以写成：  
    Σ = U Λ U T = ( u 1 u 2 ⋯ u p ) [ λ 1 0 ⋯ 0 0 λ 2 ⋯ 0 ⋮ ⋮ ⋱ ⋮ 0 0 ⋯ λ p ] ( u 1 T u 2 T ⋮ u p T ) = ( u 1 λ 1 u 2 λ 2 ⋯ u p λ p ) ( u 1 T u 2 T ⋮ u p T ) = ∑ i = 1 p u i λ i u i T
    
    $$\begin{array}{r}\Sigma=U\Lambda U^{T} =\begin{pmatrix} u _{1} & u _{2} & \cdots & u _{p} \end{pmatrix}\begin{bmatrix} \lambda _{1} & 0 & \cdots & 0 \\ 0 & \lambda _{2} & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & \lambda _{p} \end{bmatrix}\begin{pmatrix} u_{1}^{T}\\ u_{2}^{T}\\ \vdots \\ u_{p}^{T} \end{pmatrix}\\ =\begin{pmatrix} u _{1}\lambda _{1} & u _{2}\lambda _{2} & \cdots & u _{p}\lambda _{p} \end{pmatrix}\begin{pmatrix} u_{1}^{T}\\ u_{2}^{T}\\ \vdots \\ u_{p}^{T} \end{pmatrix}=\sum_{i=1}^{p}u_{i}\lambda _{i}u_{i}^{T}\\ \end{array}$$
    
     Σ=UΛUT=(u1​​u2​​⋯​up​​)⎣⎢⎢⎢⎡​λ1​0⋮0​0λ2​⋮0​⋯⋯⋱⋯​00⋮λp​​⎦⎥⎥⎥⎤​⎝⎜⎜⎜⎛​u1T​u2T​⋮upT​​⎠⎟⎟⎟⎞​=(u1​λ1​​u2​λ2​​⋯​up​λp​​)⎝⎜⎜⎜⎛​u1T​u2T​⋮upT​​⎠⎟⎟⎟⎞​=∑i=1p​ui​λi​uiT​​  
    Σ − 1 \Sigma ^{-1} Σ−1 则为：  
    Σ − 1 = ( U Λ U T ) − 1 = ( U T ) − 1 Λ − 1 U − 1 = U Λ − 1 U T = ∑ i = 1 p u i 1 λ i u i T \Sigma ^{-1}=(U\Lambda U^{T})^{-1}=(U^{T})^{-1}\Lambda ^{-1}U^{-1}=U{\Lambda^{-1}}U^{T}=\sum_{i=1}^{p}u_{i}\frac{1}{\lambda _{i}}u _{i}^{T} Σ−1=(UΛUT)−1=(UT)−1Λ−1U−1=UΛ−1UT=i=1∑p​ui​λi​1​uiT​  
    其中 Λ − 1 = d i a g ( 1 λ i ) , i = 1 , 2 , ⋯   , p \Lambda^{-1}=diag(\frac{1}{\lambda _{i}}),i=1,2,\cdots,p Λ−1=diag(λi​1​),i=1,2,⋯,p。
    
*   将概率密度函数 ( p d f : p r o b a b i l i t y    d e n s i t y    f u n c t i o n \color{red}pdf: probability\;density\;function pdf:probabilitydensityfunction) 整理成椭圆方程的形式  
    Δ = ( x − μ ) T Σ − 1 ( x − μ ) = ( x − μ ) T ∑ i = 1 p u i 1 λ i u i T ( x − μ ) = ∑ i = 1 p ( x − μ ) T u i 1 λ i u i T ( x − μ ) ( 令 y i = ( x − μ ) T u i ) = ∑ i = 1 p y i 1 λ i y i T = ∑ i = 1 p y i 2 λ i
    
    $$\begin{array}{l}\Delta =(x-\mu )^{T}\Sigma ^{-1}(x-\mu )\\ =(x-\mu )^{T}\sum_{i=1}^{p}u _{i}\frac{1}{\lambda _{i}}u _{i}^{T}(x-\mu )\\ =\sum_{i=1}^{p}(x-\mu )^{T}u _{i}\frac{1}{\lambda _{i}}u _{i}^{T}(x-\mu )\\ \color{blue}(令y_{i}=(x-\mu )^{T}u _{i})\\ =\sum_{i=1}^{p}y_{i}\frac{1}{\lambda _{i}}y_{i}^{T} =\sum_{i=1}^{p}\frac{y_{i}^{2}}{\lambda _{i}} \end{array}$$
    
     Δ=(x−μ)TΣ−1(x−μ)=(x−μ)T∑i=1p​ui​λi​1​uiT​(x−μ)=∑i=1p​(x−μ)Tui​λi​1​uiT​(x−μ)(令 yi​=(x−μ)Tui​)=∑i=1p​yi​λi​1​yiT​=∑i=1p​λi​yi2​​​
    
    1.  上式中 y i = ( x − μ ) T u i y_{i}=(x-\mu )^{T}u _{i} yi​=(x−μ)Tui​可以理解为将 x x x 减去均值进行中心化以后再投影到 u i u _{i} ui​方向上，相当于做了一次坐标轴变换。
    2.  当 x x x 的维度为 2 2 2 即 p = 2 p=2 p=2 时 Δ = y 1 2 λ 1 + y 2 2 λ 2 \Delta =\frac{y_{1}^{2}}{\lambda _{1}}+\frac{y_{2}^{2}}{\lambda _{2}} Δ=λ1​y12​​+λ2​y22​​，得到类似椭圆方程的等式，所以也就可以解释为什么其等高线是椭圆形状。二维高斯分布的图像如下所示：  
        
        ![](<assets/1696783264353.png>)
        
    
    *   如果是对角矩阵的话，椭圆就是正的椭圆。
    *   每当 Δ \Delta Δ取不同值，椭圆就相当于对这一高度的等高线，也对应一个固定的概率值若 λ i = c \lambda_i=c λi​=c(常量) 时，上图便是一个圆
    

### 1.3 高斯分布的局限性

#### 1.3.1 参数过多

*   协方差矩阵 Σ p × p \Sigma _{p\times p} Σp×p​中的参数共有 1 + 2 + ⋯ + p = p ( p + 1 ) 2 1+2+\cdots +p=\frac{p(p+1)}{2} 1+2+⋯+p=2p(p+1)​个（ Σ p × p \Sigma _{p\times p} Σp×p​是对称矩阵），因此当 x x x 的维度 p p p 很大时，高斯分布的参数就会有很多，其计算复杂度为 O ( p 2 ) O(p^{2}) O(p2)。
    
    *   可以通过假设高斯分布的协方差矩阵为 对 角 矩 阵 \color{red} 对角矩阵 对角矩阵来减少参数，当高斯分布的协方差矩阵为对角矩阵时，特征向量的方向就会和原坐标轴的方向平行，因此高斯分布的等高线（同心椭圆）就不会倾斜。
    *   另外如果在高斯分布的协方差矩阵为对角矩阵为对角矩阵的基础上使得其 特 征 值 全 部 相 等 \color{red} 特征值全部相等 特征值全部相等（即 λ 1 = λ 2 = ⋯ = λ i \lambda _{1}=\lambda _{2}=\cdots=\lambda _{i} λ1​=λ2​=⋯=λi​）, 则高斯分布的等高线就会成为一个圆形，而且不会倾斜，称为**各向同性**。
    

#### 1.3.2 单个高斯分布拟合能力有限

有些数据无法用一个高斯分布表示  
因此在 GMM 中提出了混合模型：使用多个高斯分布进行混合，比如高斯混合模型。

### 1.4 边缘概率及条件概率

已知：一个多维高斯分布的联合概率 x ∼ N ( μ , Σ ) = 1 ( 2 π ) p 2 ∣ Σ ∣ 1 2 exp ⁡ ( − 1 2 ( x − μ ) T Σ − 1 ( x − μ ) ) x\sim N(\mu,\Sigma) = {1\over(2\pi)^{p\over2}\lvert\Sigma\rvert^{1\over 2}}\exp(-{1\over2}{(x-\mu)^T\Sigma^{-1}(x-\mu)}) x∼N(μ,Σ)=(2π)2p​∣Σ∣21​1​exp(−21​(x−μ)TΣ−1(x−μ))，其中 x ∈ R p x \in \mathbb R^p x∈Rp,  
x = ( x 1 x 2 ⋮ x p ) ,    μ = ( μ 1 μ 2 ⋮ μ p ) ,    Σ = ( σ 11 σ 12 ⋯ σ 1 p σ 21 σ 22 ⋯ σ 2 p ⋮ ⋮ ⋮ σ p 1 σ p 2 ⋯ σ p p ) p × p x=

$$\begin{pmatrix} x_1\\ x_2\\ \vdots\\ x_p\\ \end{pmatrix}$$

,\; \mu=

$$\begin{pmatrix} \mu_1\\ \mu_2\\ \vdots\\ \mu_p\\ \end{pmatrix}$$

,\; \Sigma=

$$\begin{pmatrix} \sigma_{11}&\sigma_{12}&\cdots&\sigma_{1p}\\ \sigma_{21}&\sigma_{22}&\cdots&\sigma_{2p}\\ \vdots&\vdots&&\vdots\\ \sigma_{p1}&\sigma_{p2}&\cdots&\sigma_{pp}\\ \end{pmatrix}$$

_{p\times p}

x=⎝⎜⎜⎜⎛​x1​x2​⋮xp​​⎠⎟⎟⎟⎞​,μ=⎝⎜⎜⎜⎛​μ1​μ2​⋮μp​​⎠⎟⎟⎟⎞​,Σ=⎝⎜⎜⎜⎛​σ11​σ21​⋮σp1​​σ12​σ22​⋮σp2​​⋯⋯⋯​σ1p​σ2p​⋮σpp​​⎠⎟⎟⎟⎞​p×p​

*   将 x x x 分为两部分，一部分为 a a a 维 x a x_a xa​ ，一部分为 b b b 维 x b x_b xb​, μ \mu μ和 Σ \Sigma Σ同理：  
    x = ( x a x b ) ,    μ = ( μ a μ b ) ,    Σ = ( σ a a σ a b σ b a σ b b ) ,    ( a + b = p ) x=
    
    $$\begin{pmatrix} x_a\\ x_b\\ \end{pmatrix}$$
    
    ,\;\mu=
    
    $$\begin{pmatrix} \mu_a\\ \mu_b\\ \end{pmatrix}$$
    
    ,\;\Sigma=
    
    $$\begin{pmatrix} \sigma_{aa}&\sigma_{ab}\\ \sigma_{ba}&\sigma_{bb}\\ \end{pmatrix}$$
    
    ,\;(a+b=p) x=(xa​xb​​),μ=(μa​μb​​),Σ=(σaa​σba​​σab​σbb​​),(a+b=p)
*   将 x x x 看为 x a x_a xa​和 x b x_b xb​的联合概率分布。
*   通用方法：配方法（RPML）；今天使用另一种方法，比配方法简便。

求其边缘概率分布及条件概率分布，即：求 P ( x a ) , P ( x b ∣ x a ) , P ( x b ) , P ( x a ∣ x b ) P(x_{a}),P(x_{b}|x_{a}),P(x_{b}),P(x_{a}|x_{b}) P(xa​),P(xb​∣xa​),P(xb​),P(xa​∣xb​)。

#### 1.4.1 定理

已 知 x ∼ N ( μ , Σ ) , x ∈ R p y = A x + B , y ∈ R q 结 论 ： y ∼ N ( A μ + B , A Σ A T ) 已知 x\sim N(\mu ,\Sigma),x\in \mathbb{R}^{p}\\ y=Ax+B,y\in \mathbb{R}^{q}\\ 结论：y\sim N(A\mu +B,A\Sigma A^{T}) 已知 x∼N(μ,Σ),x∈Rpy=Ax+B,y∈Rq 结论：y∼N(Aμ+B,AΣAT)

简单但不严谨的证明：  
E [y] = E [ A x + B ] = A E [ x ] + B = A μ + B E[y]=E[Ax+B]=AE[x]+B=A\mu +B E[y]=E[Ax+B]=AE[x]+B=Aμ+B V a r [y] = V a r [ A x + B ] = V a r [ A x ] + V a r [ B ] = A V a r [ x ] A T + 0 = A Σ A T

$$\begin{array}{l}Var[y]=Var[Ax+B]\\ =Var[Ax]+Var[B]\\ =AVar[x]A^{T}+0 =A\Sigma A^{T}\end{array}$$

 Var[y]=Var[Ax+B]=Var[Ax]+Var[B]=AVar[x]AT+0=AΣAT​

#### 1.4.2 求边缘概率

求边缘概率 P ( x a ) P(x_{a}) P(xa​)，令  
x a = ( I m 0 ) ⏟ A ( x a x b ) ⏟ x + 0 ⏟ B x_a=\underbrace{

$$\begin{pmatrix} I_m&0 \end{pmatrix}$$

}_{A} \underbrace{

$$\begin{pmatrix} x_a\\x_b \end{pmatrix}$$

}_{x}+\underbrace0_{B}

xa​=A (Im​​0​)​​x (xa​xb​​)​​+B 0​​

则：

E [ x a ] = A μ + B = ( I m 0 ) ( μ a μ b ) + 0 = μ a E[x_a]=A\mu+B=

$$\begin{pmatrix}I_m&0 \end{pmatrix}$$

$$\begin{pmatrix}\mu_a\\\mu_b \end{pmatrix}$$

+0 =\mu_a E[xa​]=Aμ+B=(Im​​0​)(μa​μb​​)+0=μa​

D [ x a ] = A Σ A T = ( I m 0 ) ( Σ a a Σ a b Σ b a Σ b b ) ( I m 0 ) = ( Σ a a Σ a b ) ( I m 0 ) = Σ a a D[x_a]=A\Sigma A^T=

$$\begin{pmatrix}I_m&0 \end{pmatrix}$$

$$\begin{pmatrix} \Sigma_{aa}&\Sigma_{ab}\\ \Sigma_{ba}&\Sigma_{bb}\\ \end{pmatrix}$$

$$\begin{pmatrix}I_m\\0 \end{pmatrix}$$

=

$$\begin{pmatrix}\Sigma_{aa}&\Sigma_{ab} \end{pmatrix}$$

$$\begin{pmatrix}I_m\\0 \end{pmatrix}$$

=\Sigma_{aa} D[xa​]=AΣAT=(Im​​0​)(Σaa​Σba​​Σab​Σbb​​)(Im​0​)=(Σaa​​Σab​​)(Im​0​)=Σaa​

所以

x a ∼ N ( μ a , Σ a a ) \color{blue}x_{a}\sim N(\mu _{a},\Sigma _{aa}) xa​∼N(μa​,Σaa​)

，同理

x b ∼ N ( μ b , Σ b b ) \color{blue}x_{b}\sim N(\mu _{b},\Sigma _{bb}) xb​∼N(μb​,Σbb​)

。

#### 1.4.3 求条件概率

求条件概率 P ( x b ∣ x a ) P(x_{b}|x_{a}) P(xb​∣xa​)

*   首先构造 { x b ⋅ a = x b − Σ b a Σ a a − 1 x a μ b ⋅ a = μ b − Σ b a Σ a a − 1 μ a Σ b b ⋅ a = Σ b b − Σ b a Σ a a − 1 Σ a b ( Σ b b ⋅ a 是 Σ a a 的 舒 尔 补 ) \color{red}\left\{
    
    $$\begin{matrix} x_{b\cdot a}=x_{b}-\Sigma _{ba}\Sigma _{aa}^{-1}x_{a}\\ \mu _{b\cdot a}=\mu_{b}-\Sigma _{ba}\Sigma _{aa}^{-1}\mu_{a}\\ \Sigma _{bb\cdot a}=\Sigma _{bb}-\Sigma _{ba}\Sigma _{aa}^{-1}\Sigma _{ab} \end{matrix}$$
    
    \right.\\ (\Sigma _{bb\cdot a} 是 \ Sigma _{aa} 的舒尔补) ⎩⎨⎧​xb⋅a​=xb​−Σba​Σaa−1​xa​μb⋅a​=μb​−Σba​Σaa−1​μa​Σbb⋅a​=Σbb​−Σba​Σaa−1​Σab​​(Σbb⋅a​是Σaa​的舒尔补)
    
    x b ⋅ a x_{b\cdot a} xb⋅a​是 x b x_b xb​与 x a x_a xa​的线性组合，故其服从高斯分布。
    

1.  先对 x b ⋅ a x_{b\cdot a} xb⋅a​进行变换，使其能够应用上述定理直接得出结果  
    x b ⋅ a = ( − Σ b a Σ a a − 1 I n ) ⏟ A ( x a x b ) ⏟ x x_{b\cdot a}=\underset{A}{\underbrace{
    
    $$\begin{pmatrix}- \Sigma _{ba}\Sigma _{aa}^{-1}& I_{n} \end{pmatrix}$$
    
    }}\underset{x}{\underbrace{
    
    $$\begin{pmatrix} x_{a}\\ x_{b} \end{pmatrix}$$
    
    }} xb⋅a​=A (−Σba​Σaa−1​​In​​)​​x (xa​xb​​)​​
    
    *   使用定理得：  
        E [ x b ⋅ a ] = ( − Σ b a Σ a a − 1 I n ) ( μ a μ b ) = μ b − Σ b a Σ a a − 1 μ a = μ b ⋅ a E[x_{b\cdot a}]=
        
        $$\begin{pmatrix} -\Sigma _{ba}\Sigma _{aa}^{-1}& I_{n} \end{pmatrix}$$
        
        $$\begin{pmatrix} \mu _{a}\\ \mu _{b} \end{pmatrix}$$
        
        =\mu_{b}-\Sigma _{ba}\Sigma _{aa}^{-1}\mu_{a}=\mu _{b\cdot a} E[xb⋅a​]=(−Σba​Σaa−1​​In​​)(μa​μb​​)=μb​−Σba​Σaa−1​μa​=μb⋅a​  
        V a r [ x b ⋅ a ] = ( − Σ b a Σ a a − 1 I ) ( Σ a a Σ a b Σ b a Σ b b ) ( − Σ b a Σ a a − 1 I ) = ( Σ b a − Σ b a Σ a a − 1 Σ a a Σ b b − Σ b a Σ a a − 1 Σ a b ) ( − Σ b a Σ a a − 1 I ) = ( 0 Σ b b − Σ b a Σ a a − 1 Σ a b ) ( − Σ b a Σ a a − 1 I ) = Σ b b − Σ b a Σ a a − 1 Σ a b = Σ b b ⋅ a
        
        $$\begin{array}{l} Var[x_{b\cdot a}] &=\begin{pmatrix}-\Sigma_{ba}\Sigma_{aa}^{-1}&I \end{pmatrix}\begin{pmatrix} \Sigma_{aa}&\Sigma_{ab}\\ \Sigma_{ba}&\Sigma_{bb}\\ \end{pmatrix} \begin{pmatrix}-\Sigma_{ba}\Sigma_{aa}^{-1}\\I \end{pmatrix}\\ &=\begin{pmatrix} \Sigma_{ba}-\Sigma_{ba}\Sigma_{aa}^{-1}\Sigma_{aa} &\Sigma_{bb}-\Sigma_{ba}\Sigma_{aa}^{-1}\Sigma_{ab} \end{pmatrix} \begin{pmatrix}-\Sigma_{ba}\Sigma_{aa}^{-1}\\I \end{pmatrix}\\ &=\begin{pmatrix} 0 &\Sigma_{bb}-\Sigma_{ba}\Sigma_{aa}^{-1}\Sigma_{ab} \end{pmatrix} \begin{pmatrix}-\Sigma_{ba}\Sigma_{aa}^{-1}\\I \end{pmatrix}\\ &=\Sigma_{bb}-\Sigma_{ba}\Sigma_{aa}^{-1}\Sigma_{ab}=\Sigma_{bb\cdot a} \end{array}$$ Var[xb⋅a​]​=(−Σba​Σaa−1​​I​)(Σaa​Σba​​Σab​Σbb​​)(−Σba​Σaa−1​I​)=(Σba​−Σba​Σaa−1​Σaa​​Σbb​−Σba​Σaa−1​Σab​​)(−Σba​Σaa−1​I​)=(0​Σbb​−Σba​Σaa−1​Σab​​)(−Σba​Σaa−1​I​)=Σbb​−Σba​Σaa−1​Σab​=Σbb⋅a​​  
        得到 x b ⋅ a ∼ N ( μ b ⋅ a , Σ b b ⋅ a ) \color{blue}x_{b\cdot a}\sim N(\mu _{b\cdot a},\Sigma _{bb\cdot a}) xb⋅a​∼N(μb⋅a​,Σbb⋅a​)。
2.  由第一个引入的量可以得到：  
    x b = x b ⋅ a + Σ b a Σ a a − 1 x a x_b=x_{b\cdot a}+\Sigma_{ba}\Sigma_{aa}^{-1}x_a xb​=xb⋅a​+Σba​Σaa−1​xa​
    
    *   此处同样利用上述定理，其中 y 为 x b ， x 为 x b . a , A 为 I , B Σ b a Σ a a − 1 x a \color{red}y 为 x_b，x 为 x_{b.a}, A 为 I , B\Sigma_{ba}\Sigma_{aa}^{-1}x_a y 为 xb​，x 为 xb.a​,A 为 I,BΣba​Σaa−1​xa​。
    *   定理： 已 知 x ∼ N ( μ , Σ ) , x ∈ R p y = A x + B , y ∈ R q 结 论 ： y ∼ N ( A μ + B , A Σ A T ) \color{red} 已知 x\sim N(\mu ,\Sigma ),x\in \mathbb{R}^{p}\\ y=Ax+B,y\in \mathbb{R}^{q}\\ 结论：y\sim N(A\mu +B,A\Sigma A^{T}) 已知 x∼N(μ,Σ),x∈Rpy=Ax+B,y∈Rq 结论：y∼N(Aμ+B,AΣAT)
    *   这里 直 接 使 用 x b 的 表 达 式 计 算 了 x b ∣ x a \color{red} 直接使用 x_b 的表达式计算了 x_b|x_a 直接使用 xb​的表达式计算了 xb​∣xa​，原因：条件概率的含义为在已知 x a x_a xa​的条件下求 x b x_b xb​的概率，因此这里假设 x a x_a xa​已知，作为常量处理。
    
    E [ x b ∣ x a ] = μ b ⋅ a + Σ b a Σ a a − 1 x a E[x_b|x_a]=\mu_{b\cdot a}+\Sigma_{ba}\Sigma_{aa}^{-1}x_a E[xb​∣xa​]=μb⋅a​+Σba​Σaa−1​xa​  
    D [ x b ∣ x a ] = Σ b b ⋅ a μ b ⋅ a + Σ b a Σ a a − 1 x a D[x_b|x_a]=\Sigma_{bb\cdot a}\mu_{b\cdot a}+\Sigma_{ba}\Sigma_{aa}^{-1}x_a D[xb​∣xa​]=Σbb⋅a​μb⋅a​+Σba​Σaa−1​xa​  
    因此可以得到 x b ∣ x a ∼ N ( μ b ⋅ a + Σ b a Σ a a − 1 x a , Σ b b ⋅ a ) \color{blue}x_{b}|x_{a}\sim N(\mu _{b\cdot a}+\Sigma _{ba}\Sigma _{aa}^{-1}x_{a},\Sigma _{bb\cdot a}) xb​∣xa​∼N(μb⋅a​+Σba​Σaa−1​xa​,Σbb⋅a​)，同理可以得到 x a ∣ x b ∼ N ( μ a ⋅ b + Σ a b Σ b b − 1 x b , Σ a a ⋅ b ) \color{blue}x_{a}|x_{b}\sim N(\mu _{a\cdot b}+\Sigma _{ab}\Sigma _{bb}^{-1}x_{b},\Sigma _{aa\cdot b}) xa​∣xb​∼N(μa⋅b​+Σab​Σbb−1​xb​,Σaa⋅b​)。
    
    * * *
    

### 1.5 联合概率分布

*   已知：  
    p (x) = N ( x ∣ μ , Λ − 1 ) p ( y ∣ x ) = N ( y ∣ A x + b , L − 1 ) p(x)=N(x|\mu ,\Lambda ^{-1})\\ p(y|x)=N(y|Ax+b ,L ^{-1}) p(x)=N(x∣μ,Λ−1)p(y∣x)=N(y∣Ax+b,L−1)
    
    *   Λ \Lambda Λ和 L L L 是精度矩阵（ p r e c i s i o n   m a t r i x precision\,matrix precisionmatrix）， p r e c i s i o n   m a t r i x = ( c o v a r i a n c e   m a t r i x ) T \color{red}precision\,matrix=(covariance\,matrix)^{T} precisionmatrix=(covariancematrix)T。
    *   有点像贝叶斯中的后验 p ( x ∣ y ) = p ( y ∣ x ) ∗ p (x) p ( y ) p(x|y)={p(y|x)*p(x)\over p(y)} p(x∣y)=p(y)p(y∣x)∗p(x)​
    *   同时有假设 y y y 与 x x x 有线性关系： y = A x + b y=Ax+b y=Ax+b
    
    求 p (y) , p ( x ∣ y ) p(y),p(x|y) p(y),p(x∣y)。
    
    *   PRML 中依然用的配方法，非常繁琐
    *   以下依旧使用构造性证明 ; 本节比上节更重要!!
    
*   已知可以确定 y y y 与 x x x 的关系为线性高斯模型，则 y y y 与 x x x 符合下述关系: y = A x + b + ϵ \color{red}y = Ax+b+\epsilon y=Ax+b+ϵ 其中 ϵ ∼ N ( 0 , L − 1 ) \epsilon\sim N(0,L^{-1}) ϵ∼N(0,L−1)
    
    *   x , y , ϵ x,y,\epsilon x,y,ϵ都是随机向量（r.v）
    *   ϵ \epsilon ϵ与 x x x 相互独立
    

#### 1.5.1 求解 p (y) p(y) p(y)

E [y] = E [ A x + b + ε ] = E [ A x + b ] + E [ ε ] = A μ + b V a r [ y ] = V a r [ A x + b + ε ] = V a r [ A x + b ] + V a r [ ε ] = A Λ − 1 A T + L − 1 E[y]=E[Ax+b+\varepsilon]=E[Ax+b]+E[\varepsilon]=A\mu+b\\ Var[y]=Var[Ax+b+\varepsilon]=Var[Ax+b]+Var[\varepsilon]=A\Lambda ^{-1}A^{T}+L ^{-1} E[y]=E[Ax+b+ε]=E[Ax+b]+E[ε]=Aμ+bVar[y]=Var[Ax+b+ε]=Var[Ax+b]+Var[ε]=AΛ−1AT+L−1  
则可以得出 y ∼ N ( A μ + b , L − 1 + A Λ − 1 A T ) \color{red}y\sim N(A\mu+b,L ^{-1}+A\Lambda ^{-1}A^{T}) y∼N(Aμ+b,L−1+AΛ−1AT)

#### 1.5.2 求解 p ( x ∣ y ) p(x|y) p(x∣y)

求解 p ( x ∣ y ) p(x|y) p(x∣y) 需要首先求解 x 与 y 的联合分布，然后根据上一部分的公式直接得到 p ( x ∣ y ) p(x|y) p(x∣y)。

联合分布的相关结论：  
x ∼ N ( μ , Σ ) = 1 ( 2 π ) p 2 ∣ Σ ∣ 1 2 exp ⁡ ( − 1 2 ( x − μ ) T Σ − 1 ( x − μ ) ) x\sim N(\mu,\Sigma) = {1\over(2\pi)^{p\over2}\lvert\Sigma\rvert^{1\over 2}}\exp(-{1\over2}{(x-\mu)^T\Sigma^{-1}(x-\mu)}) x∼N(μ,Σ)=(2π)2p​∣Σ∣21​1​exp(−21​(x−μ)TΣ−1(x−μ))，其中 x ∈ R p x \in \mathbb R^p x∈Rp x = ( x a x b ) ， 其 中 x a 是 m 维 的 ， x b 是 n 维 的 。 μ = ( μ a μ b ) Σ = ( Σ a a Σ a b Σ b a Σ b b ) x=

$$\begin{pmatrix} x_{a}\\ x_{b} \end{pmatrix}$$

，其中 x_{a} 是 m 维的，x_{b} 是 n 维的。\\ \mu =

$$\begin{pmatrix} \mu_{a}\\ \mu_{b} \end{pmatrix}$$

\Sigma =

$$\begin{pmatrix} \Sigma _{aa}&\Sigma _{ab}\\ \Sigma _{ba}&\Sigma _{bb} \end{pmatrix}$$

 x=(xa​xb​​)，其中 xa​是 m 维的，xb​是 n 维的。μ=(μa​μb​​)Σ=(Σaa​Σba​​Σab​Σbb​​)  
x a ∣ x b ∼ N ( μ a ⋅ b + Σ a b Σ b b − 1 x b , Σ a a ⋅ b ) \color{blue}x_{a}|x_{b}\sim N(\mu _{a\cdot b}+\Sigma _{ab}\Sigma _{bb}^{-1}x_{b},\Sigma _{aa\cdot b}) xa​∣xb​∼N(μa⋅b​+Σab​Σbb−1​xb​,Σaa⋅b​)

*   构 造 z = ( x y ) 构造 z=
    
    $$\begin{pmatrix} x\\ y \end{pmatrix}$$ 构造 z=(xy​), 则：  
    E [z] = ( μ A μ + b ) D [ z ] = ( c o v ( x , x ) c o v ( x , y ) c o v ( y , x ) c o v ( y , y ) ) = ( Λ − 1 c o v ( x , y ) c o v ( y , x ) L − 1 + A Λ − 1 A T ) E[z]=
    
    $$\begin{pmatrix}\mu\\A\mu+b \end{pmatrix}$$
    
    \\ D[z]=
    
    $$\begin{pmatrix} cov(x,x)&cov(x,y)\\ cov(y,x)&cov(y,y) \end{pmatrix}$$
    
     =
    
    $$\begin{pmatrix} \Lambda^{-1}&cov(x,y)\\ cov(y,x)&L^{-1}+A\Lambda^{-1}A^T \end{pmatrix}$$ E[z]=(μAμ+b​)D[z]=(cov(x,x)cov(y,x)​cov(x,y)cov(y,y)​)=(Λ−1cov(y,x)​cov(x,y)L−1+AΛ−1AT​)  
    C o v ( x , y ) = E [ ( x − E [x] ) ( y − E [ y ] ) T ] = E [ ( x − μ ) ( y − A μ − b ) T ] = E [ ( x − μ ) ( A x + b + ε − A μ − b ) T ] = E [ ( x − μ ) ( A x − A μ + ε ) T ] = E [ ( x − μ ) ( A x − A μ ) T + ( x − μ ) ε T ] = E [ ( x − μ ) ( A x − A μ ) T ] + E [ ( x − μ ) ε T ] （ 因 为 x 与 ε 独 立 ， 所 以 ( x − μ ) 与 ε 独 立 ， 所 以 E [ ( x − μ ) ε T ] = E [ ( x − μ ) ] E [ ε T ] ) = E [ ( x − μ ) ( A x − A μ ) T ] + E [ ( x − μ ) ] E [ ε T ] = E [ ( x − μ ) ( A x − A μ ) T ] + E [ ( x − μ ) ] ⋅ 0 = E [ ( x − μ ) ( A x − A μ ) T ] = E [ ( x − μ ) ( x − μ ) T A T ] = E [ ( x − μ ) ( x − μ ) T ] A T = V a r [ x ] A T = Λ − 1 A T
    
    $$\begin{array}{l} Cov(x,y)=E[(x-E[x])(y-E[y])^{T}]\\ =E[(x-\mu )(y-A\mu-b)^{T}]\\ =E[(x-\mu )(Ax+b+\varepsilon-A\mu-b)^{T}]\\ =E[(x-\mu )(Ax-A\mu+\varepsilon)^{T}]\\ =E[(x-\mu )(Ax-A\mu)^{T}+(x-\mu)\varepsilon^{T}]\\ =E[(x-\mu )(Ax-A\mu)^{T}]+E[(x-\mu)\varepsilon^{T}]\\ （\color{blue}{因为x与\varepsilon独立，所以(x-\mu)与\varepsilon独立，所以E[(x-\mu)\varepsilon^{T}]=E[(x-\mu)]E[\varepsilon^{T}]})\\ =E[(x-\mu )(Ax-A\mu)^{T}]+E[(x-\mu)]E[\varepsilon^{T}]\\ =E[(x-\mu )(Ax-A\mu)^{T}]+E[(x-\mu)]\cdot 0\\ =E[(x-\mu )(Ax-A\mu)^{T}]\\ =E[(x-\mu )(x-\mu )^{T}A^{T}]\\ =E[(x-\mu )(x-\mu )^{T}]A^{T}\\ =Var[x]A^{T}\\ =\Lambda ^{-1}A^{T}\\ \end{array}$$ Cov(x,y)=E[(x−E[x])(y−E[y])T]=E[(x−μ)(y−Aμ−b)T]=E[(x−μ)(Ax+b+ε−Aμ−b)T]=E[(x−μ)(Ax−Aμ+ε)T]=E[(x−μ)(Ax−Aμ)T+(x−μ)εT]=E[(x−μ)(Ax−Aμ)T]+E[(x−μ)εT]（因为 x 与ε独立，所以 (x−μ) 与ε独立，所以 E[(x−μ)εT]=E[(x−μ)]E[εT])=E[(x−μ)(Ax−Aμ)T]+E[(x−μ)]E[εT]=E[(x−μ)(Ax−Aμ)T]+E[(x−μ)]⋅0=E[(x−μ)(Ax−Aμ)T]=E[(x−μ)(x−μ)TAT]=E[(x−μ)(x−μ)T]AT=Var[x]AT=Λ−1AT​
*   由对称性得： c o v ( y , x ) = A Λ − 1 cov(y,x)=A\Lambda^{-1} cov(y,x)=AΛ−1
*   由此可得 z ∼ N ( ( μ A μ + b ) , ( Λ − 1 Λ − 1 A T A Λ − 1 L − 1 + A Λ − 1 A T ) ) \color{red}z\sim N(
    
    $$\begin{pmatrix}\mu\\A\mu+b \end{pmatrix}$$
    
    ,
    
    $$\begin{pmatrix} \Lambda^{-1}&\Lambda^{-1}A^T\\ A\Lambda^{-1}&L^{-1}+A\Lambda^{-1}A^T \end{pmatrix}$$
    
    ) z∼N((μAμ+b​),(Λ−1AΛ−1​Λ−1ATL−1+AΛ−1AT​))
*   套用上一部分的公式 x a ∣ x b ∼ N ( μ a ⋅ b + Σ a b Σ b b − 1 x b , Σ a a ⋅ b ) x_a|x_b\sim N(\mu_{a\cdot b}+\Sigma_{ab}\Sigma_{bb}^{-1}x_b,\Sigma_{aa\cdot b}) xa​∣xb​∼N(μa⋅b​+Σab​Σbb−1​xb​,Σaa⋅b​) 可得到：  
    E [ x ∣ y ] = μ + Λ − 1 A T ( L − 1 + A Λ − 1 A T ) − 1 ( y − A μ − b ) D [ x ∣ y ] = Λ − 1 − Λ − 1 A T ( L − 1 + A Λ − 1 A T ) − 1 A Λ − 1 E[x|y]=\mu + \Lambda^{-1}A^T(L^{-1}+A\Lambda^{-1}A^T)^{-1}(y-A\mu-b)\\ D[x|y]=\Lambda^{-1}-\Lambda^{-1}A^T(L^{-1}+A\Lambda^{-1}A^T)^{-1}A\Lambda^{-1} E[x∣y]=μ+Λ−1AT(L−1+AΛ−1AT)−1(y−Aμ−b)D[x∣y]=Λ−1−Λ−1AT(L−1+AΛ−1AT)−1AΛ−1  
    因此  
    x ∣ y ∼ N ( μ + Λ − 1 A T ( L − 1 + A Λ − 1 A T ) − 1 ( y − A μ − b ) , Λ − 1 − Λ − 1 A T ( L − 1 + A Λ − 1 A T ) − 1 A Λ − 1 ) \color{red}x|y\sim N(\mu + \Lambda^{-1}A^T(L^{-1}+A\Lambda^{-1}A^T)^{-1}(y-A\mu-b),\Lambda^{-1}-\Lambda^{-1}A^T(L^{-1}+A\Lambda^{-1}A^T)^{-1}A\Lambda^{-1}) x∣y∼N(μ+Λ−1AT(L−1+AΛ−1AT)−1(y−Aμ−b),Λ−1−Λ−1AT(L−1+AΛ−1AT)−1AΛ−1)

#### 1.6 概率 - 不等式 1 - 杰森不等式（Jensen’s Inequality）

杰森不等式在机器学习的推导中经常被用到，因此单独拿出来介绍

*   杰森不等式是什么？
    
    假设 f (x) f(x) f(x) 是 c o n v e x   f u n c t i o n \color{red}convex \ function convex function(凸函数)  
    则 E [ f (x) ] ≥ f ( E [ x ] ) \color{red}E[f(x)] \ge f(E[x]) E[f(x)]≥f(E[x])
    
*   证明方法有很多，本次采用一个构造性证明  
    
    ![](<assets/1696783264408.png>)
    
      
    如上图所示，根据 E [x] E[x] E[x] 点找到 f ( E [x] ) f(E[x]) f(E[x]) 点，然后做切线 l (x) = a x + b l(x)=ax+b l(x)=ax+b  
    因此 f ( E [x] ) = l ( E [ x ] ) = a E [ x ] + b f(E[x])=l(E[x])=aE[x]+b f(E[x])=l(E[x])=aE[x]+b  
    ∵ f (x)   i s   c o n v e x ∴ ∀ x   f ( x ) ≥ l ( x ) \because f(x) \ is\ convex\\ \therefore \forall x \ f(x)\ge l(x) ∵f(x) is convex∴∀x f(x)≥l(x)  
    对上式结论两边同时取期望  
    E [ f (x) ] ≥ E [ l ( x ) ] = E [ a x + b ] = a E [ x ] + b = f ( E [ x ] )
    
    $$\begin{array}{l}E[f(x)]&\ge E[l(x)]\\ &=E[ax+b]\\ &=aE[x]+b\\ &=f(E[x]) \end{array}$$
    
     E[f(x)]​≥E[l(x)]=E[ax+b]=aE[x]+b=f(E[x])​  
    证毕
    
*   **杰森不等式的变式**  
    实际上我们在机器学习中使用的更多的是杰森不等式的变式，如下推导  
    
    ![](<assets/1696783264434.png>)
    
      
    如上图所示，令 μ ∈ ( 0 , 1 ) \mu \in (0,1) μ∈(0,1)
    

则 c = a + μ ( b − a ) = a + μ b − μ a = ( 1 − μ ) a + μ b c=a+\mu(b-a)=a+\mu b-\mu a=(1-\mu)a+\mu b c=a+μ(b−a)=a+μb−μa=(1−μ)a+μb

令 t = 1 − μ , t ∈ ( 0 , 1 ) t=1-\mu,t\in(0,1) t=1−μ,t∈(0,1)

则 c = t a + ( 1 − t ) b \color{blue}c=ta+(1-t)b c=ta+(1−t)b

然后连接 f (a) f(a) f(a) 与 f (b) f(b) f(b) 作一条新的线为 g (x) g(x) g(x)

因为 f (x)   i s   c o n v e x f(x) \ is\ convex f(x) is convex（凸），所以 g (c) ≥ f ( c ) g(c)\ge f(c) g(c)≥f(c)

c = t a + ( 1 − t ) b ⇒ ( c − a ) : ( b − c ) = t : ( 1 − t ) c=ta+(1-t)b \Rightarrow (c-a):(b-c)=t:(1-t) c=ta+(1−t)b⇒(c−a):(b−c)=t:(1−t)

因此如上图所示，可利用相似三角形性质求得：

( g (c) − f ( a ) ) : ( f ( b ) − g ( c ) ) = t : ( 1 − t ) (g(c)-f(a)):(f(b)-g(c))=t:(1-t) (g(c)−f(a)):(f(b)−g(c))=t:(1−t)

因此 g (c) = t f ( a ) + ( 1 − t ) f ( b ) g(c)=tf(a)+(1-t)f(b) g(c)=tf(a)+(1−t)f(b)

所以 t f (a) + ( 1 − t ) f ( b ) ≥ f ( c ) = f ( t a + ( 1 − t ) b ) tf(a)+(1-t)f(b)\ge f(c)=f(ta+(1-t)b) tf(a)+(1−t)f(b)≥f(c)=f(ta+(1−t)b)  
#end

t f (a) + ( 1 − t ) f ( b ) ≥ f ( c ) = f ( t a + ( 1 − t ) b ) \color{red}tf(a)+(1-t)f(b)\ge f(c)=f(ta+(1-t)b) tf(a)+(1−t)f(b)≥f(c)=f(ta+(1−t)b)  
此式非常常用，非常重要！