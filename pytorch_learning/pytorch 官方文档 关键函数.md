
# torch.nn
## LSTM
https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#
注意输出包含隐藏状态和cell状态

注意输入和输出
### 输入
![[Pasted image 20231022135630.png]]
### 输出
![[Pasted image 20231022135707.png]]
## LSTM Cell
https://pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html#torch.nn.LSTMCell
Cell  每次只进行运算一个数据。

## pack_padded_sequence
可以将Tensor 的维度 从 (src_len, batch_size, embedding_size) 转换到（batch_sum_len, embedding_size)

batch_sum_len 是指 这一批数据所有的词的长度， 不包括填充的部分。

## squeeze  和 unsqueeze

```python
e_t = enc_hiddens_proj.bmm(dec_hidden.unsqueeze(2)).squeeze(2)
```
1. `enc_hiddens_proj` 是编码器隐藏状态的投影，通常在注意力机制中用于计算注意力分数。这个张量的形状通常为 `(batch_size, seq_length, hidden_dim)`，其中 `batch_size` 是批次大小，`seq_length` 是编码器序列的长度，`hidden_dim` 是隐藏状态的维度。

2. `dec_hidden` 是解码器的当前隐藏状态，通常具有形状 `(batch_size, hidden_dim)`。

3. `dec_hidden.unsqueeze(2)` 将解码器隐藏状态的形状从 `(batch_size, hidden_dim)` 扩展为 `(batch_size, hidden_dim, 1)`，这是为了与 `enc_hiddens_proj` 进行批次矩阵乘法（batch matrix multiplication）做准备。

4. `enc_hiddens_proj.bmm(dec_hidden.unsqueeze(2))` 执行批次矩阵乘法，将 `enc_hiddens_proj` 与 `dec_hidden.unsqueeze(2)` 进行点积运算。这将为每个示例（批次中的每个示例）计算一个注意力分数。

5. `squeeze(2)` 操作用于将结果张量的最后一个维度（尺寸为1的维度）压缩掉，从 `(batch_size, seq_length, 1)` 变为 `(batch_size, seq_length)`。这是为了获得最终的注意力分数张量，其中每个元素表示解码器当前隐藏状态对编码器不同位置的注意力分数。

综合来说，这行代码的目的是计算解码器当前隐藏状态与编码器各个位置之间的注意力分数。这是注意力机制的一部分，通常在序列到序列的任务中用于确定解码器在生成输出时应该关注编码器的哪些部分。

## PyTorch中的contiguous
PyTorch 提供了is_contiguous、contiguous (形容词动用)两个方法 ，分别用于判定Tensor是否是 contiguous 的，以及保证Tensor是contiguous的。

某些Tensor操作（如`transpose、permute、narrow、expand`）与原Tensor是共享内存中的数据，不会改变底层数组的存储，但原来在语义上相邻、内存里也相邻的元素在执行这样的操作后，在语义上相邻，但在内存不相邻，即不连续了（is not contiguous）。

如果想要变得连续使用contiguous方法，如果Tensor不是连续的，则会重新开辟一块内存空间保证数据是在内存中是连续的，如果Tensor是连续的，则contiguous无操作。
1. torch.view等方法操作需要连续的Tensor。
transpose、permute 操作虽然没有修改底层一维数组，但是新建了一份Tensor元信息，并在新的元信息中的 重新指定 stride。torch.view 方法约定了不修改数组本身，只是使用新的形状查看数据。如果我们在 transpose、permute 操作后执行 view，Pytorch 会抛出错误.

为什么不在view 方法中默认调用contiguous方法?
因为历史上view方法已经约定了共享底层数据内存，返回的Tensor底层数据不会使用新的内存，如果在view中调用了contiguous方法，则可能在返回Tensor底层数据中使用了新的内存，这样打破了之前的约定，破坏了对之前的代码兼容性。为了解决用户使用便捷性问题，PyTorch在0.4版本以后提供了reshape方法，实现了类似于 tensor.contigous().view(*args)的功能，如果不关心底层数据是否使用了新的内存，则使用reshape方法更方便。 


## stride()
在PyTorch中，`t.stride()`是一个张量的方法，用于返回张量在每个维度上的步长（stride）。步长表示在每个维度上移动一个元素所需的张量存储单位数量。 

例如，对于一个二维张量（矩阵），其步长是一个二元组，分别表示在行和列上移动一个元素所需的存储单位数量。

下面是一个简单的示例，说明了如何使用`stride()`方法：

```python
import torch

# 创建一个3x3的张量
t = torch.tensor([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

# 获取张量的步长
strides = t.stride()

print("Tensor strides:", strides)
```

输出会是：

```
Tensor strides: (3, 1)
```

这表明在行上移动一个元素需要3个存储单位，而在列上移动一个元素只需要1个存储单位。
