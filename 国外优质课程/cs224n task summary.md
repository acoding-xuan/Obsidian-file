## task3

### Adam Optimizer (Adaptive Moment estimation)
https://zhuanlan.zhihu.com/p/268193140
#### 指数加权平均数Exponentially Weighted Average
![[Pasted image 20240406183156.png]]
![[Pasted image 20240406183256.png]]
#### Momentum(使用质数加权平均来计算梯度)
![[Pasted image 20240406183554.png]]
#### RMSprop Root Mean Square propagation
目的：将梯度除以序列数据的指数加权平均数（类似normalize操作）来更新可学习参数，以防止某一参数方向上，更新的梯度过大或者过小。

![[Pasted image 20240406183738.png]]
#### 结合成Adam
![[Pasted image 20240406183859.png]]

与编译器中的解析树类似，NLP中的解析树是用于分析句子的句法结构。主要有两种类型的使用结构-短语结构和依存结构。
### transition-based parser



# task 4
完成了一个从中文到英文的 翻译器
![[Pasted image 20240406221047.png]]
## 模型执行步骤
1. 将原语言句子对应(单词或字符)id 转化为 Word embedding，将embedding 送入卷积层并保持形状不变。
2. 将embedding 送入双向LSTM 编码器得到：
![[Pasted image 20240406221451.png]]
3. 初始化decoder的隐藏状态为：
![[Pasted image 20240406221723.png]]
4. 在第t步将目标语言句子一个单词或字符的id 对应的embedding $y_t$  并同时将其和 $o_{t - 1}$ (计算方法如下)结合生成
$\overline{y_{t}}\in \mathbb{R}^{(e+h) \times 1}$ 然后将 $\bar{y}_t$ 作为input 输入 decoder
![[Pasted image 20240406223223.png]]
5. 使用 预测状态 $h_{t}^{dec}$ 和之前encoder 的隐藏层状态一起计算乘法注意力
![[Pasted image 20240406223702.png]]
6. 使用注意力分数来计算 $o_t$
![[Pasted image 20240406222554.png]]
7. 使用softmax 函数计算预测概率 $P_t$ 
![[Pasted image 20240406224015.png]]
## 几种常见注意力

### 1. **Dot Product Attention**（点乘注意力）：
   $$
   
   \text{Attention}(q, k, v) = \text{softmax}\left(\frac{q \cdot k^T}{\sqrt{d_k}}\right) \cdot v$$

   其中 \( q \) 为查询向量，\( k \) 为键向量，\( v \) 为值向量，\( d_k \) 是键向量的维度。
   - **计算方法**：通过计算查询向量 \( q \) 和键向量 \( k \) 之间的点积来计算注意力权重，然后对值向量 \( v \) 进行加权求和。
   - **优点**：计算简单，计算效率高，尤其适用于较短的序列。
   - **缺点**：对输入维度敏感，可能存在信息丢失的问题。
### 2. **Additive Attention**（加性注意力）：
 $$ 
   \text{Attention}(q, k, v) = \text{softmax}\left(\text{score}(q, k)\right) \cdot v
   
   $$
   其中 $$\text{score}(q, k) = \text{tanh}(W_q q + W_k k + b)， W_q 、 W_k  和  b $$ 是学习参数。
   - **计算方法**：通过将查询向量和键向量经过线性变换后相加，并应用激活函数得到注意力分数，然后对值向量进行加权求和。
   - **优点**：相对于点乘注意力，更适用于处理输入维度较高的情况，具有更好的表现。
   - **缺点**：计算量较大，可能在处理大型输入序列时效率较低。
### 3. **Multiplicative Attention**（乘性注意力）：
$$ \text{Attention}(q, k, v) = \text{softmax}\left(q \cdot W \cdot k^T\right) \cdot v$$
   - **计算方法**：通过计算查询向量和键向量之间的乘积来计算注意力权重，然后对值向量进行加权求和。
   - **优点**：计算相对简单，且相对于加性注意力，乘性注意力在某些情况下可以`更有效地捕捉到序列之间的关系`。
   - **缺点**：可能存在维度灾难问题，对于高维输入序列的处理可能不够有效。
以下是各种常见注意力计算方法的公式：
### 各部分作用

#### 1D 卷积
1D 卷积层通过在一维输入序列上使用多个滤波器进行卷积操作，从而提取输入序列中的局部模式和特征。 可以用来提取出词语的关系。
### BLEU score
BLEU的全名为：bilingual evaluation understudy，即：双语互译质量评估辅助工具。它是用来评估机器翻译质量的工具。

BLUE将机器翻译的结果与其相对应的几个参考翻译作比较，算出一个综合分数。这个分数越高说明机器翻译得越好。注意BLEU算法是句子之间的比较，不是词组，也不是段落。






# task 5

## 2. Pretrainded Transformer models and knowlege access (35 points)

### (a) (0 points) Check out the demo.
In the mingpt-demo/ folder is a Jupyter notebook play char.ipynb that trains and samples from a Transformer language model. Take a look at it (locally on your computer) to get somewhat familiar with how it defines and trains models.
#### dataset
```python
import math
from torch.utils.data import Dataset
class CharDataset(Dataset):
    def __init__(self, data, block_size):
        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))
        
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data
    
    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.block_size + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        """
        arrange data and targets so that the first i elements of x
        will be asked to predict the i-th element of y. Notice that
        the eventual language model will actually make block_size
        individual predictions at the same time based on this data,
        so we are being clever and amortizing the cost of the forward
        pass of the network. So for example if block_size is 4, then
        we could e.g. sample a chunk of text "hello", the integers in
        x will correspond to "hell" and in y will be "ello". This will
        then actually "multitask" 4 separate examples at the same time
        in the language model:
        - given just "h", please predict "e" as next
        - given "he" please predict "l" next
        - given "hel" predict "l" next
        - given "hell" predict "o" next
        
        In addition, because the DataLoader will create batches of examples,
        every forward/backward pass during traning will simultaneously train
        a LOT of predictions, amortizing a lot of computation. In particular,
        for a batched input of integers X (B, T) where B is batch size and
        T is block_size and Y (B, T), the network will during training be
        simultaneously training to make B*T predictions, all at once! Of course,
        at test time we can paralellize across batch B, but unlike during training
        we cannot parallelize across the time dimension T - we have to run
        a forward pass of the network to recover the next single character of the 
        sequence along each batch dimension, and repeatedly always feed in a next
        character to get the next one.
        
        So yes there is a big asymmetry between train/test time of autoregressive
        models. During training we can go B*T at a time with every forward pass,
        but during test time we can only go B at a time, T times, with T forward 
        passes.
        """
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y
```

### 创建dataset
```python
# you can download this file at https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt
text = open('input.txt', 'r').read() # don't worry we won't run out of file handles
train_dataset = CharDataset(text, block_size) # one line of poem is roughly 50 characters
```

### 创建模型并开始训练
```python
from mingpt.model import GPT, GPTConfig
mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                  n_layer=8, n_head=8, n_embd=512)
model = GPT(mconf)


from mingpt.trainer import Trainer, TrainerConfig

# initialize a trainer instance and kick off training
tconf = TrainerConfig(max_epochs=2, batch_size=512, learning_rate=6e-4,
                      lr_decay=True, warmup_tokens=512*20, final_tokens=2*len(train_dataset)*block_size,
                      num_workers=4)
trainer = Trainer(model, train_dataset, None, tconf)
trainer.train()
```

## (b) (0 points) Read through NameDataset in src/dataset.py
 our dataset for reading name birthplace pairs.

**The task we’ll be working on with our pretrained models is attempting to access the birth place of a notable person, as written in their Wikipedia page.**

```python
Here are some examples of input-output pairs (x, y):

  x: Khatchig Mouradian. Khatchig Mouradian is a jour⁇and tran⁇nalist, writer ⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
  y: hatchig Mouradian. Khatchig Mouradian is a jour⁇and tran⁇nalist, writer ⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□

  x: Jaco⁇enry ⁇b H⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
  y: aco⁇enry ⁇b H⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□

  x: John Stephen. Born in Glasgow, Steph⁇lder's apprentice on⁇en became a we⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
  y: ohn Stephen. Born in Glasgow, Steph⁇lder's apprentice on⁇en became a we⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□

class CharCorruptionDataset(Dataset):
    def __init__(self, data, block_size):
        self.MASK_CHAR = u"\u2047" # the double question mark character, for mask ??
        self.PAD_CHAR = u"\u25A1" # the empty square character, for pad 

        chars = list(sorted(list(set(data))))
        assert self.MASK_CHAR not in chars 
        assert self.PAD_CHAR not in chars
        chars.insert(0, self.MASK_CHAR)
        chars.insert(0, self.PAD_CHAR)

        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }

        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data.split('\n')

    def __len__(self):
        # returns the length of the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # TODO [part e]: see spec above
        raise NotImplementedError
```

## (c) (0 points) Implement finetuning (without pretraining).

Take a look at run.py. It has some skeleton code specifying flags you’ll eventually need to handle as command line arguments.
#### run.py needed args
```python
argp = argparse.ArgumentParser()
argp.add_argument('function', help="Choose pretrain, finetune, or evaluate")
argp.add_argument('variant', help="Choose vanilla or perceiver") 
argp.add_argument('--bottleneck_dim', type=int, default=32)
argp.add_argument('pretrain_corpus_path', default=None)
argp.add_argument('--reading_params_path',default=None)
argp.add_argument('--writing_params_path',default=None)
argp.add_argument('--finetune_corpus_path', default=None)
argp.add_argument('--eval_corpus_path', default=None)
argp.add_argument('--outputs_path', default=None)
argp.add_argument('--pretrain_lr', default=6e-3, type=float)
argp.add_argument('--finetune_lr', default=6e-4, type=float)
argp.add_argument('--tb_expt_name', help='debug string for tb log.',
                  default='run')
args = argp.parse_args()
```

Taking inspiration from the training code in the play char.ipynb file, write code to finetune a Transformer model on the name/birthplace dataset, via examples from the NameDataset class.
You’ll have to modify two sections, marked [part c] in the code: one to initialize the model, and one to finetune it. Note that you only need to initialize the
model in the case labeled “vanilla” for now (later in section (g), we will explore a model variant). Use the hyperparameters for the Trainer specified in the run.py code.

```python
if args.variant == 'vanilla':
    # [part c] Make some model here
    model = model.GPT(mconf).to(device)


if args.reading_params_path is not None:
        model.load_state_dict(torch.load(args.reading_params_path))
        trainer_config = trainer.TrainerConfig(
            max_epochs=10,
            batch_size=256,
            learning_rate=args.finetune_lr,
            lr_decay=True,
            warmup_tokens=512*20,
            final_tokens=200*len(pretrain_dataset)*block_size,
            num_workers=2,
            writer=writer
        )
    else:
        trainer_config = trainer.TrainerConfig(
            max_epochs=75,
            batch_size=256,
            learning_rate=args.finetune_lr,
            lr_decay=True,
            warmup_tokens=512*20,
            final_tokens=200*len(pretrain_dataset)*block_size,
            num_workers=2,
            writer=writer
        )
    finetune_dataset = dataset.NameDataset(pretrain_dataset, open(args.finetune_corpus_path, encoding='utf-8').read())

    trainer = trainer.Trainer(model, finetune_dataset, None, trainer_config)
    trainer.train()

    torch.save(model.state_dict(), args.writing_params_path)

```
## (d) (5 points) Make predictions (without pretraining).

### test model
```python
python src/run.py finetune vanilla wiki.txt --writing_params_path vanilla.model.params --finetune_corpus_path birth_places_train.tsv

python src/run.py evaluate vanilla wiki.txt --reading_params_path vanilla.model.params --eval_corpus_path birth_dev.tsv --outputs_path vanilla.nopretrain.dev.predictions


python src/run.py evaluate vanilla wiki.txt --reading_params_path vanilla.model.params --eval_corpus_path birth_test_inputs.tsv --outputs_path vanilla.nopretrain.test.predictions
```
### Fill in london baseline.py
```python
import utils

eval_corpus_path = "birth_dev.tsv"
eval = open(eval_corpus_path, "r")
len_eval = len(eval.readlines())  # 先计算一共有多少行

predictions = ["London"] * len_eval
total, correct = utils.evaluate_places(eval_corpus_path, predictions)
if total > 0:
    print('Correct: {} out of {}: {}%'.format(correct, total, correct/total*100))
else:
    print('No targets provided!')
```

## (e) Define a span corruption function for pretraining.

Span corruption is explored in the T5 paper .  It randomly selects spans of text in a document and replaces them with unique tokens (noising).  Models take this noised text, and are required to output a pattern of each unique sentinel followed by the tokens that were replaced by that sentinel in the input.
###  the getitem () function
```python
"""
The __getitem__ function takes an index and returns a data point (x, y) where
x and y are Long tensors of length self.block_size. x encodes the input
sequence, and y encodes the output sequence.

0. Use the idx argument of __getitem__ to retrieve the element of self.data
at the given index. We'll call the resulting data entry a document.

1. Randomly truncate the document to a length no less than 4 characters,
and no more than int(self.block_size*7/8) characters.

- IMPORTANT: You are free to decide how to perform this random truncation, but
make sure that the length is picked _randomly_ (every possible length from 4
to int(self.block_size*7/8) has a chance of being picked) for full credit.

2. Now, break the (truncated) document into three substrings:
    
    [prefix] [masked_content] [suffix]

  In other words, choose three strings prefix, masked_content and suffix
    such that prefix + masked_content + suffix = [the original document].
  The length of [masked_content] should be random, and 1/4 the length of the
    truncated document on average.

- IMPORTANT: You are free to decide how to perform this operation, but
make sure that the length is picked _randomly_ (has a chance of being more or
less than 1/4 the length of the truncated document) for full credit.

3. Rearrange these substrings into the following form:

    [prefix] MASK_CHAR [suffix] MASK_CHAR [masked_content] [pads]
  
  This resulting string, denoted masked_string, serves as the output example.
  Here MASK_CHAR is the masking character defined in Vocabulary Specification,
    and [pads] is a string of repeated PAD_CHAR characters chosen so that the
    entire string is of length self.block_size.
  Intuitively, the [masked_content], a string, is removed from the document and
    replaced with MASK_CHAR (the masking character defined in Vocabulary
    Specification). After the suffix of the string, the MASK_CHAR is seen again,
    followed by the content that was removed, and the padding characters.

4. We now use masked_string to construct the input and output example pair. To
do so, simply take the input string to be masked_string[:-1], and the output
string to be masked_string[1:]. In other words, for each character, the goal is
to predict the next character in the masked string.

5. Making use of the vocabulary that you defined, encode the resulting input
and output strings as Long tensors and return the resulting data point.

----------------
Here are some examples of input-output pairs (x, y):

  x: Khatchig Mouradian. Khatchig Mouradian is a jour⁇and tran⁇nalist, writer ⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
  y: hatchig Mouradian. Khatchig Mouradian is a jour⁇and tran⁇nalist, writer ⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□

  x: Jaco⁇enry ⁇b H⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
  y: aco⁇enry ⁇b H⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□

  x: John Stephen. Born in Glasgow, Steph⁇lder's apprentice on⁇en became a we⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
  y: ohn Stephen. Born in Glasgow, Steph⁇lder's apprentice on⁇en became a we⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
"""
def __getitem__(self, idx):
        # TODO [part e]: see spec above
        doc = self.data[idx]
        truncated_len = random.randint(4, int(self.block_size * 7 / 8))
        truncated_doc = doc[:truncated_len]
        masked_content_len = int(torch.randint(low=1, high=2 * int(truncated_len / 4), size=(1,))[0])
        masked_content_index = int(torch.randint(low=0, high=truncated_len - int(truncated_len / 4) + 1, size=(1,))[0])
        
        prefix, masked_content, suffix = truncated_doc[0:masked_content_index], truncated_doc[masked_content_index:
            masked_content_index + masked_content_len], truncated_doc[masked_content_index + masked_content_len:]
        
        # [prefix] MASK_CHAR [suffix] MASK_CHAR [masked_content] [pads]
        masked_string = prefix + self.MASK_CHAR + suffix + self.MASK_CHAR + masked_content
        masked_string = masked_string + self.PAD_CHAR * (self.block_size - len(masked_string))
        
        input_string, output_string = masked_string[0:-1], masked_string[1:]
        
        return torch.tensor([self.stoi[c] for c in input_string], dtype=torch.long), torch.tensor([self.stoi[c] for c in output_string], dtype=torch.long)
```
### (f) Pretrain, finetune, and make predictions.
#### run.py pretrain
```python
if args.function == 'pretrain':
    assert args.writing_params_path is not None
    # TODO [part f]:
    # - Given:
    #     1. A corpus specified in args.pretrain_corpus_path
    #     2. An output path args.writing_params_path for the model parameters
    # - Goals:
    #     1. Pretrain the model on this corpus
    #     2. Save the resulting model in args.writing_params_path
    
    # - Make sure to use the following hyperparameters for pretraining:
    # Hyperparameters for pretraining:
    # max_epochs=650
    # batch_size=128
    # learning_rate=args.pretrain_lr
    # lr_decay=True
    # warmup_tokens=512*20
    # final_tokens=200*len(pretrain_dataset)*block_size
    # num_workers=4
    # writer=writer 
    assert args.pretrain_corpus_path is not None
    
    trainer_config = trainer.TrainerConfig(
        max_epochs=650,
        batch_size=128,
        learning_rate=args.pretrain_lr,
        lr_decay=True,
        warmup_tokens=512*20,
        final_tokens=200*len(pretrain_dataset)*block_size,
        num_workers=2,
        writer=writer
    )
    trainer = trainer.Trainer(model, pretrain_dataset, None, trainer_config)
    trainer.train()
    torch.save(model.state_dict(), args.writing_params_path)    
```
#### test
```python
# Pretrain the model
python src/run.py pretrain vanilla wiki.txt --writing_params_path vanilla.pretrain.params
# Finetune the model
python src/run.py finetune vanilla wiki.txt --reading_params_path vanilla.pretrain.params --writing_params_path vanilla.finetune.params --finetune_corpus_path birth_places_train.tsv
# Evaluate on the dev set; write to disk
python src/run.py evaluate vanilla wiki.txt --reading_params_path vanilla.finetune.params --eval_corpus_path birth_dev.tsv --outputs_path vanilla.pretrain.dev.predictions
# Evaluate on the test set; write to disk
python src/run.py evaluate vanilla wiki.txt --reading_params_path vanilla.finetune.params --eval_corpus_path birth_test_inputs.tsv --outputs_path vanilla.pretrain.test.predictions
```
![[Pasted image 20240401231050.png]]
直接从 %2 到 21.2%
### (g)Write and try out a more efficient variant of Attention

The transformer model uses a `self-attention scoring function based on dot products`, this involves a rather intensive computation that’s quadratic in the sequence length. . If we can reduce the length of the sequence passed on the self-attention module, we should observe significant reduction in compute. 


PerceiverAR [1] proposes a solution to make the model more efficient by
`reducing the sequence length of the input to self-attention for the intermediate layers`. 
In the first layer, the input sequence is projected onto a lower-dimensional basis. Subsequently,all self-attention layers operate in this smaller subspace. The last layer projects the output back to the original input sequence length. In this assignment, we propose a simpler version of the PerceiverAR transformer model.

#### CausalSelfAttention layer
![[Pasted image 20240401211934.png]]

```python
class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    I believe I could have just used torch.nn.MultiheadAttention but their documentation
    is all but absent and code ugly so I don't trust it, rolling my own here.
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # torch.tril(input)：这个函数会返回输入张量的下三角部分，其余部分将被置为0。在这里，它将全 1 矩阵转换为下三角矩阵。
        # self.register_buffer("mask", tensor)：
		# 这个函数用于将一个张量注册为模型的缓冲区。模型的缓冲区在进行模型保存和加载时会被自动保存和加载。在这里，它创建了一个名为     
		# "mask" 的缓冲区，并将其初始化为下面介绍的张量。
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x):
        B, T, C = x.size() # batch_size, seq_len, hidden_size

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) #进行矩阵乘法注意只是对最后两维
        
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, -1e10)
        att = F.softmax(att, dim=-1) 
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y
```

#### Transformer Block
```python
class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = attention.CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
```

####  Perceiver model architecture
1. replace the first transformer Block in the model with the `DownProjectBlock`.  This block reduces the length of the sequence from 
`ℓ to m.`
3. This is followed by a series of regular transformer blocks, which would now perform self-attention on the reduced sequence length of m.
4. We replace the last block of the model with the `UpProjectBlock`, which takes in the m length output of the previous block, and projects it back to the original sequence length of ℓ.
![[Pasted image 20240401222334.png]]
#### Casual Cross Attention
处理q和k, v 前不同系数的x.
```python
class CausalCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x_kv, x_q): # x_q 变成了 C
        Bk, Tk, Ck = x_kv.size()
        Bq, Tq, Cq = x_q.size()
        
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        
        # keys of x1
        k = self.key(x_kv).view(Bk, Tk, self.n_head, Ck // self.n_head).transpose(1, 2) # (B, nh, Tk, hs)
        
        # query with x2
        q = self.query(x_q).view(Bq, Tq, self.n_head, Cq // self.n_head).transpose(1, 2) # (B, nh, Tq, hs)
        
        # values from x1
        v = self.value(x_kv).view(Bk, Tk, self.n_head, Ck // self.n_head).transpose(1, 2) # (B, nh, Tk, hs)

        # causal self-attention;  (B, nh, Tk, hs) x (B, nh, hs, Tq) -> (B, nh, Tq, Tk)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        B = max(Bk, Bq)
        
        att = att.masked_fill(self.mask[:,:,:Tq,:Tk] == 0, -1e10) 
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, Tq, Tk) x (B, nh, Tk, hs) -> (B, nh, Tq, hs)
        y = y.transpose(1, 2).contiguous().view(B, Tq, Cq) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y
```
#### DownProjectBlock
![[Pasted image 20240401222334.png]]
注意：这里的C就是一个参数矩阵
```python
class DownProjectBlock(nn.Module):
    """Transformer block used for down projection
    `self.C` will be used to compute the Query vector for the cross attention
    layer.
    """
    def __init__(self, config):
        super().__init__()
        ### YOUR CODE HERE
        ### Hint: Copy over the code from Block and make necessary modifications.
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.C = nn.Parameter(nn.init.xavier_uniform_(torch.empty(1, config.bottleneck_dim, config.n_embd)))
        self.attn = attention.CausalCrossAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )
        ### END YOUR CODE
    def forward(self, x_input):
        """Hint: perform cross-attention between x_input and self.C.
        Use the layernorm layers on C, and then on the input to the MLP.
        """
        ### YOUR CODE HERE
        ### Hint: Copy over the code from Block and make necessary modifications.
        ### Should be around 3-5 lines.
        x = x_input
        x = x + self.attn(self.ln1(x), self.ln1(self.C))
        x = x + self.mlp(self.ln2(x))
        ### END YOUR CODE
```

#### UpProjectBlock
![[Pasted image 20240401225800.png]]
```python
class UpProjectBlock(nn.Module):
    """Transformer block used for up projection.
    
    Initialize similarly to the regular transformer Block class,
    while using the CausalCrossAttention layer instead of the regular
    CausalSelfAttention layer.
    """
    def __init__(self, config):
        super().__init__()
        ### YOUR CODE HERE
        ### Hint: Copy over the code from Block and make necessary modifications.
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = attention.CausalCrossAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )
        ### END YOUR CODE
    
    def forward(self, y, x_input):
        """Hint: perform cross-attention between previous layer's output y and
        x_input. 
        Use the layernorm layers on y, and then on the input to the MLP.
        """
        ### YOUR CODE HERE
        ### Hint: Copy over the code from Block and make necessary modifications.
        ### Should be around 3-5 lines.
        x = self.attn(self.ln1(y), x_input)
        x = x + self.mlp(self.ln2(x))
        return x
```

#### test
```python
# Pretrain the model
python src/run.py pretrain perceiver wiki.txt --bottleneck_dim 64 --pretrain_lr 6e-3 --writing_params_path perceiver.pretrain.params
# Finetune the model
python src/run.py finetune perceiver wiki.txt --bottleneck_dim 64 --reading_params_path perceiver.pretrain.params --writing_params_path perceiver.finetune.params --finetune_corpus_path birth_places_train.tsv
# Evaluate on the dev set; write to disk
python src/run.py evaluate perceiver wiki.txt --bottleneck_dim 64 --reading_params_path perceiver.finetune.params 
--eval_corpus_path birth_dev.tsv --outputs_path perceiver.pretrain.dev.predictions
# Evaluate on the test set; write to disk
python src/run.py evaluate perceiver wiki.txt --bottleneck_dim 64 \
--reading_params_path perceiver.finetune.params \
--eval_corpus_path birth_test_inputs.tsv \
--outputs_path perceiver.pretrain.test.predictions
```

```python
"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier


Originally forked from Andrej Karpathy's minGPT.

CS224N 2022-23: Homework 5

John Hewitt <johnhew@stanford.edu>
Ansh Khurana <anshk@stanford.edu>
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

import attention

torch.manual_seed(1)


class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    perceiver = False
    bottleneck_dim = None

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)


class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = attention.CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class DownProjectBlock(nn.Module):
    """Transformer block used for down projection.
    
    Initialize similarly to the regular tranformer Block class,
    while using the CausalCrossAttention layer instead of the regular
    CausalSelfAttention layer.
    
    You also need to initialize the parameter for the basis vectors `self.C` here.
    Initialize `self.C` with appropriate dimensions and xavier_uniform initalization.
    
    self.C should be 1 x bottleneck_dim x n_embd. We need the first dimension 
    for appropriate broadcasting along the batch_size dimension of the input 
    sequence.
    
    `self.C` will be used to compute the Query vector for the cross attention
    layer.
    """
    def __init__(self, config):
        super().__init__()
        ### YOUR CODE HERE
        ### Hint: Copy over the code from Block and make necessary modifications.
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = attention.CausalCrossAttention(config)
        self.C = nn.Parameter(nn.init.xavier_uniform_(torch.empty(1, config.bottleneck_dim, config.n_embd)))

        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )
        ### END YOUR CODE

    def forward(self, x_input):
        """Hint: perform cross-attention between x_input and self.C.
        Use the layernorm layers on C, and then on the input to the MLP.
        """
        ### YOUR CODE HERE
        ### Hint: Copy over the code from Block and make necessary modifications.
        ### Should be around 3-5 lines.
        x = self.attn(self.ln1(x_input), self.ln1(self.C))
        x = x + self.mlp(self.ln2(x))
        return x
        ### END YOUR CODE
    
    
class UpProjectBlock(nn.Module):
    """Transformer block used for up projection.
    
    Initialize similarly to the regular transformer Block class,
    while using the CausalCrossAttention layer instead of the regular
    CausalSelfAttention layer.
    """
    def __init__(self, config):
        super().__init__()
        ### YOUR CODE HERE
        ### Hint: Copy over the code from Block and make necessary modifications.
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = attention.CausalCrossAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )
        ### END YOUR CODE
    
    def forward(self, y, x_input):
        """Hint: perform cross-attention between previous layer's output y and
        x_input. 
        Use the layernorm layers on y, and then on the input to the MLP.
        """
        ### YOUR CODE HERE
        ### Hint: Copy over the code from Block and make necessary modifications.
        ### Should be around 3-5 lines.
        x = self.attn(self.ln1(y), x_input)
        x = x + self.mlp(self.ln2(x))
        return x
        ### END YOUR CODE
    

class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.perceiver = config.perceiver
        if config.perceiver:            
            input_block_size = config.block_size
            
            # input sequence based causal mask
            self.down_block = DownProjectBlock(config)
            
            # bottleneck basis based causal mask
            config.block_size = config.bottleneck_dim
            self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer-2)])
            
            # reset value of the block size back to the original.
            config.block_size = input_block_size
            self.up_block = UpProjectBlock(config)
            
            
        else:
            self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)

        print("number of parameters: {}".format(sum(p.numel() for p in self.parameters())))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size (%d, %d) is exhausted." % (t, self.block_size)

        # forward the GPT model
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        x_input = self.drop(token_embeddings + position_embeddings)
        
        if self.perceiver:
            x = self.down_block(x_input)
        else:
            x = x_input
        
        # always compute through the blocks
        x = self.blocks(x)
        
        if self.perceiver:
            x = self.up_block(x, x_input)
            
        x = self.ln_f(x)
        logits = self.head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0)

        return logits, loss
```

# Default Project

## Sentiment Analysis
Sentiment analysis can be utilized to determine individual feelings towards particular products, politicians, or within news reports.
### Stanford Sentiment Treebank
`Stanford Sentiment Treebank`  consists of 11,855 single sentences extracted from movie reviews.

 The dataset was parsed with the Stanford parser2 and includes a total of 215,154 unique phrases from those parse trees, each annotated by 3 human judges. 
 Each phrase has a label of `negative, somewhat negative, neutral, somewhat positive, or positive.`\
![[Pasted image 20240402213227.png]]
## Paraphrase Detection
改写检测（Paraphrase Detection）是一种自然语言处理（NLP）任务，旨在确定两个文本片段是否表达相同的含义，尽管它们可能使用了不同的词汇、句法结构或表达方式。在文本匹配和语义相似性方面，改写检测在许多NLP应用中都是一个关键的子任务。
paraphrase detection thus essentially seeks to determine whether particular words or phrases convey the same semantic meaning.
### Quora released dataset
Quora released a dataset that labeled whether different questions were paraphrases of each other.
 ![[Pasted image 20240402214014.png]]
##  Semantic Textual Similarity (STS)
语义文本相似性（Semantic Textual Similarity，STS）是自然语言处理（NLP）领域的一个任务，旨在衡量两个文本片段之间的语义相似性。与改写检测类似，但STS更注重于评估两个文本片段之间的语义相似程度，而不仅仅是是否具有相同的含义。STS通常被用作测量自然语言处理系统中诸如句子嵌入、语义表示学习等技术的性能。

STS differs from paraphrasing in it is not a yes or no decision; rather STS allows for degrees of similarity. For example, on a scale from 5 (same meaning) to 0 (not at all related), the following sentences have the following relationships to each other.

The goal models trained on the STS task (often using a `cosine similarity metric` based on their word embbedding) would be to appropriately train a model to predict the similarity of each of the above sentences in terms of their semantic content.


![[Pasted image 20240402214340.png]]


## The jobs

 1. first implement some of the key aspects of the original BERT model including multi-head self-attention as well as a Transformer layer.
 2. you will utilize your completed BERT model to perform sentiment analysis on the `Stanford Sentiment Treebank dataset` as well as
another dataset of movie reviews.
 3.  Finally, in the latter half of this project, you will fine-tune and otherwise extend the BERT model to create sentence embeddings that can perform well across a wide range of downstream tasks.

## minBERT

### structure
#### Tokenization (tokenizer.py)
tokenizing and converting each token to ids
#### Embedding Layer (bert.BertModel.embed)
After tokenizing and converting each token to ids, the BERT model subsequently utilizes a `trainable` embedding layer across each token.

later used embeddings = token embeddings + the segmentation embeddings + the position embeddings

the positional embeddings are utilized to encode the position of different words within the input.
Like the token embeddings, position embeddings are learned embeddings that are learned for each of the 512 positions in a given BERT input.
#### BERT Transformer Layer (bert.BertLayer)
![[Pasted image 20240403220612.png]]
#### Multiheaded Self-Attention (bert.BertSelfAttention.attention)
![[Pasted image 20240403220946.png]]

#### Position-wise Feed-Forward Networks
#### BERT output (bert.BertModel.forward)
#### Training BERT
The original version of BERT was trained using two unsupervised tasks on Wikipedia articles.
![[Pasted image 20240403221716.png]]
##### Masked Language Modeling 
In order to train BERT to extract deep bidirectional representations, the training procedure masks some percentage (`15%` in the original paper) of the word piece tokens and attempts to predict them. 

Specifically, the final hidden vectors corresponding to the masked tokens are fed into an output softmax layer over the vocabulary and are subsequently predicted. 

To prevent a mismatch between initial pre-training and later fine-tuning（预训练时有[MASK] token,下游任务没有）, in the training procedure the “masked” tokens are not always replaced by the [MASK] token. Rather the training data generator chooses 15% of the token positions at random for prediction, then in 80% of these cases the token is replaced [MASK], in 10% of cases the token is replaced with a random token, and in another 10% of cases, the token will remain unchanged.
##### Next Sentence Prediction
In order to allow BERT to understand the relationships between two sentences, BERT is further fine-tuned on the Next Sentence Prediction task.

Specifically across training with these sentence pairs, the BERT model, 50% of the time is shown the actual next sentence, and 50% of the time it is shown a random sentence. The BERT model then predicts across these pairs, whether the second inputted sentence was actually the next sentence.


 ### Code To Be Implemented
 #### BERT Multi-head Self-Attention  bert.SelfAttention.attention
```python
class BertSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # initialize the linear transformation layers for key, value, query
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)
    # this dropout is applied to normalized attention scores following the original implementation of transformer
    # although it is a bit unusual, we empirically observe that it yields better performance
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transform(self, x, linear_layer):
    # the corresponding linear_layer of k, v, q are used to project the hidden_state (x)
    bs, seq_len = x.shape[:2]
    proj = linear_layer(x)
    # next, we need to produce multiple heads for the proj 
    # this is done by spliting the hidden state to self.num_attention_heads, each of size self.attention_head_size
    proj = proj.view(bs, seq_len, self.num_attention_heads, self.attention_head_size)
    # by proper transpose, we have proj of [bs, num_attention_heads, seq_len, attention_head_size]
    proj = proj.transpose(1, 2)
    return proj

  def attention(self, key, query, value, attention_mask):
    # each attention is calculated following eq (1) of https://arxiv.org/pdf/1706.03762.pdf
    # attention scores are calculated by multiply query and key 
    # and get back a score matrix S of [bs, num_attention_heads, seq_len, seq_len]
    # S[*, i, j, k] represents the (unnormalized)attention score between the j-th and k-th token, given by i-th attention head
    # before normalizing the scores, use the attention mask to mask out the padding token scores
    # Note again: in the attention_mask non-padding tokens with 0 and padding tokens with a large negative number 

    # normalize the scores
    # multiply the attention scores to the value and get back V'
    # next, we need to concat multi-heads and recover the original shape [bs, seq_len, num_attention_heads * attention_head_size = hidden_size]
	bs, num_attention_heads, seq_len, ahs = key.size()
    S = torch.matmul(query, torch.transpose(key, -1, -2)) / ahs ** 0.5
    masked_S = attention_mask + S
    weight = F.softmax(masked_S, -1) # bs*h*n*n
    # multiply the attention scores to the value and get back V'
    scores = torch.matmul(weight, value) # bs*h*n*ahs
    scores = torch.transpose(scores, 1, 2).contiguous() 
    # concat multi-heads and recover the original shape [bs, seq_len, num_attention_heads * attention_head_size = hidden_size]
    scores = scores.view(bs, seq_len, -1)
    return scores
    
  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # first, we have to generate the key, value, query for each token for multi-head attention w/ transform (more details inside the function)
    # of *_layers are of [bs, num_attention_heads, seq_len, attention_head_size]
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)
    # calculate the multi-head attention 
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value
```
 ####  BertLayer
 ```python
 class BertLayer(nn.Module):
  def __init__(self, config):
    super().__init__()
    # multi-head attention
    self.self_attention = BertSelfAttention(config)
    # add-norm
    self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
    # feed forward
    self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
    self.interm_af = F.gelu
    # another add-norm
    self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

  def add_norm(self, input, output, dense_layer, dropout, ln_layer):
    """
    this function is applied after the multi-head attention layer or the feed forward layer
    input: the input of the previous layer
    output: the output of the previous layer
    dense_layer: used to transform the output
    dropout: the dropout to be applied 
    ln_layer: the layer norm to be applied
    """
    # Hint: Remember that BERT applies to the output of each sub-layer, before it is added to the sub-layer input and normalized 
    ### TODO
    
    raise NotImplementedError


  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: either from the embedding layer (first bert layer) or from the previous bert layer
    as shown in the left of Figure 1 of https://arxiv.org/pdf/1706.03762.pdf 
    each block consists of 
    1. a multi-head attention layer (BertSelfAttention)
    2. a add-norm that takes the input and output of the multi-head attention layer
    3. a feed forward layer
    4. a add-norm that takes the input and output of the feed forward layer
    """
    ### TODO
    raise NotImplementedError
```

### 4.1 Datasets
#### Stanford Sentiment Treebank (SST) dataset
![[Pasted image 20240405132757.png]]
#### CFIMDB dataset
![[Pasted image 20240405132832.png]]

### Code To Be Implemented: Sentiment Classification with BERT embeddings

```python
class BertSentimentClassifier(torch.nn.Module):
    '''
    This module performs sentiment classification using BERT embeddings on the SST dataset.
    In the SST dataset, there are 5 sentiment categories (from 0 - "negative" to 4 - "positive").
    Thus, your forward() should return one logit for each of the 5 classes.
    '''
    def __init__(self, config):
        super(BertSentimentClassifier, self).__init__()
        self.num_labels = config.num_labels
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Pretrain mode does not require updating bert paramters.
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True
        self.classfier = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_size, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, self.num_labels)
        )
        
    def forward(self, input_ids, attention_mask):
        '''Takes a batch of sentences and returns logits for sentiment classes'''
        # The final BERT contextualized embedding is the hidden state of [CLS] token (the first token).
        # HINT: you should consider what is the appropriate output to return given that
        # the training loop currently uses F.cross_entropy as the loss function.
        output_dict =  self.bert(input_ids, attention_mask) # 使用bert 进行向量化
        pooler_output = output_dict['pooler_output'] # 取出bert返回的最后的句子语义
        logits = self.classfier(pooler_output) # 用来进行分类
        return logits
```


### Adam Optimizer
![[Pasted image 20240405150129.png]]
###  Extensions and Improvements for Additional Downstream Tasks

调整bert的embedding 使其能同时进行多个任务 。 sentiment analysis, paraphrase detection, and semantic textual similarity
#### Quora Dataset
![[Pasted image 20240405165522.png]]

#### SemEval STS Benchmark Dataset
![[Pasted image 20240405165541.png]]

#### Main parts

##### multitask_classifier.MultitaskBERT:
A class that imports the weights of a pre-trained BERT model and can predict sentiment, paraphrases, and semantic textual similarity.

```python
class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:
    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        # You will want to add layers here to perform the downstream tasks.
        # Pretrain mode does not require updating bert paramters.
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True
        ### TODO
        raise NotImplementedError


    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        ### TODO
        raise NotImplementedError


    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        ### TODO
        raise NotImplementedError


    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        ### TODO
        raise NotImplementedError


    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''
        ### TODO
        raise NotImplementedError
```


##### multitask_classifier.train_multitask()

A function for training your model. It is largely your choice how to train your model. As a baseline, you will find the original code from classifier.py to train your model on the SST sentiment dataset.


```python
## Currently only trains on sst dataset
def train_multitask(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Load data
    # Create the data and its corresponding datasets and dataloader
    sst_train_data, num_labels,para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')

    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)

    # Init model
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc = 0

    # Run for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for batch in tqdm(sst_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            b_ids, b_mask, b_labels = (batch['token_ids'],
                                       batch['attention_mask'], batch['labels'])

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            b_labels = b_labels.to(device)

            optimizer.zero_grad()
            logits = model.predict_sentiment(b_ids, b_mask)
            loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / (num_batches)

        train_acc, train_f1, *_ = model_eval_sst(sst_train_dataloader, model, device)
        dev_acc, dev_f1, *_ = model_eval_sst(sst_dev_dataloader, model, device)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)

        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")

```

##### datasets.SentenceClassificationDataset:
A class for handling the SST sentiment dataset.

```python
class SentenceClassificationDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):

        sents = [x[0] for x in data]
        labels = [x[1] for x in data]
        sent_ids = [x[2] for x in data]

        encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])
        labels = torch.LongTensor(labels)

        return token_ids, attention_mask, labels, sents, sent_ids

    def collate_fn(self, all_data):
        token_ids, attention_mask, labels, sents, sent_ids= self.pad_data(all_data)

        batched_data = {
                'token_ids': token_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'sents': sents,
                'sent_ids': sent_ids
            }

        return batched_data
```
##### evaluations.test_multitask()
 A function for testing your model. You should call this function after loading in an appropriate checkpoint of your model.

### 改进方法

如果直接顺序将任务所有的数据对bert进行微调，会发现效果很差，可能是因为后面任务的数据会导致前面部分数据
参数的遗忘。 所以这个任务主要就是实现一种多任务模型共享一份bert的参数从而达到一个很好的效果。


思路：对于三个任务，在进行pretrain 阶段 固定bert的参数为bert_base 的参数。进行训练改变对于每个任务后面的classfier 的参数
finetune阶段： 不再对 bert 的参数进行固定而是 都进行训练跟新，但是 只训练有限的轮次。

预测阶段：对不同的任务,对bert产生的向量进行不同的处理
对于`情感分析`任务：直接将CLS 对应的向量 通过分类器进行父类, loss 使用 crossEntropy
对于`改写检测`:  将两个句子对应的向量进行按照元素相减， 然后将两个向量和相减以后向量的绝对值拼接， 将拼接以后的输入classfier
loss 使用 crossEntropy
对于`语义相似度计算`:  同上可以将向量cos 相似度然后 再进行拼接。
loss 使用 mse
#### finetune 策略 轮训finetune:
每次取一组：
非轮训：每次取一个。
- `--rlayer`：在para和sts之间引入共享层，relational layer
效果最好：不共享参数 + 轮训进行微调














