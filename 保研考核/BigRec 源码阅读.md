# 新技术模块
## wandb
Weights & Biases（简称wandb）是一个用于机器学习实验跟踪和可视化的工具和平台。它提供了一个简单的Python API，可以轻松地将实验数据发送到云端，并通过网络应用程序进行访问和可视化。以下是wandb的一些核心功能：

1. **实验跟踪和记录**：wandb能够自动跟踪机器学习实验，包括超参数、指标、模型架构等，并将这些信息保存在云端，以便后续查看和比较 。

2. **结果可视化**：提供了丰富的可视化功能，包括曲线图、散点图、直方图等，帮助用户更好地理解实验结果和模型性能 。

3. **模型检查点和版本控制**：可以保存模型检查点，并生成唯一的版本号，方便回溯和比较不同的实验结果 。

4. **协作和共享**：邀请团队成员参与实验、查看结果，并进行讨论和反馈。允许将实验和结果与其他人共享，使他们可以在不同的环境中重现和使用您的工作 。

5. **集成多种框架**：支持与各种机器学习框架（如TensorFlow、PyTorch、Keras等）和机器学习工具（如scikit-learn）集成，并提供了方便的API，方便进行实验管理和结果跟踪 。

使用wandb时，可以通过几行代码快速集成到现有项目中。例如，使用`wandb.init()`初始化一个新的实验，使用`wandb.log()`记录实验中的指标和日志信息，使用`wandb.finish()`结束实验记录。此外，wandb还支持模型的监视（`wandb.watch()`）、文件保存（`wandb.save()`）和从云端恢复实验记录的模型参数或文件（`wandb.restore`）等功能 。

## tokenizer
在自然语言处理（NLP）和机器学习中，tokenizer（分词器）是一个关键组件，它的主要作用是将文本数据转换为模型可以理解和处理的格式。
# 不清楚参数

## cutoff-len
cutoff_len参数通常用于定义处理或考虑序列数据（如文本）时的最大长度。
## micro_batch_size
在提供的代码片段中，micro_batch_size被用作transformers.Trainer的参数之一，指定了在训练和评估过程中，每次迭代使用的样本数量。此外，代码中还计算了gradient_accumulation_steps，这是基于batch_size和micro_batch_size的比例来确定的。这意味着，即使每次只处理少量样本，通过梯度累积，模型仍然可以模拟较大批次大小的训练效果。
## resume_from_checkpoint
恢复中断的训练：如果训练过程由于某些原因（如硬件故障、电源中断等）被意外中断，可以使用最后一个检查点从中断的地方继续训练，而不是从头开始。

尝试不同的训练参数：在训练过程中，可能需要调整学习率、优化器或其他超参数。从检查点恢复训练可以确保这些更改仅影响训练过程的这一部分，而不会抹去之前已经完成的训练进度。

迁移学习：在迁移学习场景中，可能会在预训练模型的基础上进行额外的训练。使用 resume_from_checkpoint 可以加载预训练的模型权重，然后在此基础上继续训练。


## python 解包字典
在Python中，`**`用于解包字典。在你提供的代码片段中：

```python
user_prompt = generate_prompt({**data_point, "output": ""})
```

这行代码意图是创建一个新的字典，这个字典首先是通过解包`data_point`字典来初始化的，然后在这个基础上添加或更新一个`"output"`键，其值被设置为一个空字符串`""`。

具体来说：

- `data_point`是一个字典，包含了一些键值对。
- `{**data_point}`部分将`data_point`字典解包，将其内容放入到一个新的字典中。
- `{**data_point, "output": ""}`将这个新字典和一个新的键值对`"output": ""`合并，形成一个新的字典。

如果`data_point`中已经包含了`"output"`这个键，那么在这个新字典中，`"output"`的值将会被更新为`""`（空字符串）。如果`"output"`键不存在于`data_point`中，那么它将被添加到新字典中。

这种用法在Python中很常见，特别是在需要基于现有字典快速创建新字典时。在这个上下文中，它可能用于生成一个用于训练的语言模型的提示，其中`"output"`键可能被用来控制是否在生成的提示中包含输出文本。


字典解包（Dictionary Unpacking）是Python中的一种语法特性，它允许你在一个表达式中将字典的键值对直接提取出来，用于创建新的字典或者作为函数调用的关键字参数。这种特性在Python 3.5及以后的版本中引入。

### 字典解包的两种常见用法：

1. **创建新字典时的解包**：
   当你想要基于已有的字典创建一个新的字典，并且需要添加或修改一些键值对时，可以使用字典解包。这可以通过在字典名前加上`**`来实现。

   ```python
   original_dict = {'a': 1, 'b': 2}
   new_dict = {'c': 3, **original_dict}
   ```

   上述代码中，`original_dict`的内容被解包并添加到`new_dict`中。结果是`new_dict`变成了`{'c': 3, 'a': 1, 'b': 2}`。

2. **函数调用时的解包**：
   当调用函数时，如果需要将字典中的键值对作为关键字参数传递，可以使用字典解包。

   ```python
   def my_function(a, b, c):
       print(a, b, c)

   params = {'b': 2, 'c': 3}
   my_function(a=1, **params)
   ```

   在这个例子中，`params`字典中的`'b'`和`'c'`键值对被解包并作为关键字参数传递给`my_function`函数。

### 字典解包的注意事项：

- 解包的字典必须有有效的键值对。
- 如果在解包时与目标字典或函数调用中存在重复的键，解包字典中的键值对会覆盖已有的。
- 解包只能在字典字面量的上下文中使用，不能用于其他数据结构。

字典解包提供了一种方便的方式来操作和传递字典数据，使得代码更加简洁和易于阅读。


# 模型下载
/data/liudaoxuan/Grounding4Rec/qwen/Qwen1___5-0___5B

# llama-7b
https://www.modelscope.cn/models/skyline2006/llama-7b/files

```c++
#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('skyline2006/llama-7b', cache_dir='./')
```
## 通义千问1.5-1.8B

```c++
#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('qwen/Qwen1.5-1.8B', cache_dir='./')
```
## 通义千问1.5-0.5B
```python
#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('qwen/Qwen1.5-0.5B', cache_dir='./')
```

# 代码运行过程
## 常用指令
![](../img/Pasted%20image%2020240729124524.png)
![](../img/Pasted%20image%2020240729124546.png)


## vscode debug
# vscode 如何debug python torchrun deepspeed

## 最优雅的方式

### 安装
1. 安装包 `pip install debugpy -U`
2. 安装vscode关于python的相关插件
### 写配置
一般情况下，大家都是使用deepspeed、torchrun运行代码。参数都特别多，然后都是使用`sh xxxx.sh`启动脚本。

#### 在python代码里面（最前面加上这句话）

```python
import debugpy
try:
    # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
    debugpy.listen(("localhost", 9501))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
except Exception as e:
    pass

```
#### 在vscode的launch.json的configuration里面，加上这个配置

```json
{
            "name": "sh_file_debug",
            "type": "debugpy",
            "request": "attach",
            "connect": {
                "host": "localhost",
                "port": 9501
            }
        },
```
🚨 上面的端口号都写一样。别搞错了。
## 启动

1. 就正常启动，直接`sh xxx.sh`
2. 在你需要debug的python文件，打上debug断点。
2. 你看打印出来的东西，是不是出现`Waiting for debugger attach`.一般来说，都很快，就出现了。
3. 再在vscode的debug页面，选择`sh_file_debug`进行debug。
4. 就基本上完成了。确实是很方便。
5. **debug结束之后，别忘记把代码里面的 添加的代码，注销掉**
## 使用镜像源

```python
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple transformers==4.37.0

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple peft==0.3.0
```
## 后台运行程序

```python
nohup ./train_qwen.sh >qinwen_18.log 2>&1 &

nohup ./train_qwen.sh >qinwen_05.log 2>&1 &

nohup ./train.sh >llama.log 2>&1 &

nohup ./inference_qwen18.sh >inference_qwen18.log 2>&1 &
nohup ./inference_qwen05.sh >inference_qwen05.log 2>&1 &


nohup ./train_qwen.sh >qinwen_05_13.log 2>&1 &
nohup ./train_qwen14.sh >qinwen_05_14.log 2>&1 &
nohup ./train_qwen55.sh >qinwen_05_55.log 2>&1 &


nohup ./train_qwen.sh >qinwen_05_18.log 2>&1 &

nohup ./inference_tem.sh >inference_tem_5000.log 2>&1 &
nohup ./inference_D3.sh >inference_D3.log 2>&1 &

nohup ./inference_D3m.sh >inference_D3m_2000.log 2>&1 &
nohup ./inference_D3m.sh >inference_D3m_5000.log 2>&1 &
nohup ./inference_tem.sh >inference_tem_1000.log 2>&1 &
nohup ./inference_tem_qwen18.sh >inference_tem_qwen18_5000.log 2>&1 &


python ./evaluate.py --input_dir ./book_result


nohup ./inference_D3m.sh >inference_D3m.log 2>&1 &
```

## Preprocess
按照步骤进行处理即可
```python
python process.py gao --category "Book" --metadata ./path_to_metadata.json --reviews ./path_to_reviews.json --K 5 --st_year 2017 --st_month 10 --ed_year 2018 --ed_month 11 --output True

python process.py --category="Books"
nohup python process.py --category="Books" >inf_qwen18.log 2>&1 &
```
2024-07-30 19:55:56.045 | INFO     | __main__:gao:172 - interaction_list: 853747
2024-07-30 19:56:20.036 | INFO     | __main__:gao:196 - Train Books: 682997
2024-07-30 19:56:20.042 | INFO     | __main__:gao:197 - Valid Books: 85375
2024-07-30 19:56:20.048 | INFO     | __main__:gao:198 - Test Books: 85375
2024-07-30 19:56:20.050 | INFO     | __main__:gao:199 - Done!
https://huggingface.co/blog/zh/constrained-beam-search

# Grounding4Rec

For item embedding, due to the quota of the git LFS, you can use the [link](https://rec.ustc.edu.cn/share/78de1e20-763a-11ee-b439-a3ef6ed8b1a3) with password 0g1g.
### Environment
```
pip install -r requirements.txt
```

```python
pip install -b /data/liudaoxuan/tmp 
```

```python
pip config set global.cache-dir "/data/liudaoxuan/pip_cache"
```
### Preprocess
Please follow the process.ipynb in each data directory.
### Training on Single Domain
```
Grounding4Rec/skyline2006/llama-7b
for seed in 0 1 2
do
    for lr in 1e-4
    do
        for dropout in 0.05    
        do
            for sample in 1024
            do
                echo "lr: $lr, dropout: $dropout , seed: $seed,"
                CUDA_VISIBLE_DEVICES=$1 python train.py \
                    --base_model "[\"./skyline2006/llama-7b\"]" \
                    --train_data_path "[\"./data/movie/train.json\"]"   \
                    --val_data_path "[\"./data/movie/valid_5000.json"]" \
                    --output_dir /model/movie/${seed}_${sample} \
                    --batch_size 128 \
                    --micro_batch_size 4 \
                    --num_epochs 50 \
                    --learning_rate $lr \
                    --cutoff_len 512 \
                    --lora_r 8 \
                    --lora_alpha 16\
                    --lora_dropout $dropout \
                    --lora_target_modules '[q_proj,v_proj]' \
                    --train_on_inputs \
                    --group_by_length \
                    --resume_from_checkpoint 'XXX' \
                    --seed $seed \
                    --sample $sample 
            done    
        done
    done
done
```
### Training on Multi Domain
```
for seed in 0 1 2
do
    for lr in 1e-4
    do
        for dropout in 0.05    
        do
            for sample in 1024
            do
                echo "lr: $lr, dropout: $dropout , seed: $seed,"
                CUDA_VISIBLE_DEVICES=$1 python train.py \
                    --base_model YOUR_LLAMA_PATH/ \
--train_data_path "[\"./data/movie/train.json\", \"./data/game/train.json\"]"  \
                    --val_data_path "[\"./data/movie/valid_5000.json\", \"./data/game/valid_5000.json\"]"  \
                    --output_dir ./model/multi/${seed}_${sample} \
                    --batch_size 128 \
                    --micro_batch_size 4 \
                    --num_epochs 50 \
                    --learning_rate $lr \
                    --cutoff_len 512 \
                    --lora_r 8 \
                    --lora_alpha 16\
                    --lora_dropout $dropout \
                    --lora_target_modules '[q_proj,v_proj]' \
                    --train_on_inputs \
                    --group_by_length \
                    --resume_from_checkpoint 'XXX' \
                    --seed $seed \
                    --sample $sample 
            done    
        done
    done
done
                    
```

### Training on Multiple GPU Card
We provide our accelerate config in ./config/accelerate.yaml
```
accelerate config # Please set up your config
for seed in 0 1 2
do
    for lr in 1e-4
    do
        for dropout in 0.05    
        do
            for sample in 1024
            do
                echo "lr: $lr, dropout: $dropout , seed: $seed,"
                CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch train.py \
                    --base_model YOUR_LLAMA_PATH/ \
--train_data_path "[\"./data/movie/train.json\", \"./data/game/train.json\"]"  \
                    --val_data_path "[\"./data/movie/valid_5000.json\", \"./data/game/valid_5000.json\"]"  \
                    --output_dir ./model/multi/${seed}_${sample} \
                    --batch_size 128 \
                    --micro_batch_size 4 \
                    --num_epochs 50 \
                    --learning_rate $lr \
                    --cutoff_len 512 \
                    --lora_r 8 \
                    --lora_alpha 16\
                    --lora_dropout $dropout \
                    --lora_target_modules '[q_proj,v_proj]' \
                    --train_on_inputs \
                    --group_by_length \
                    --resume_from_checkpoint 'XXX' \
                    --seed $seed \
                    --sample $sample 
            done    
        done
    done
done
```

### Inference
```
#  Taking movie as an example
python inference.py \
    --base_model YOUR_LLAMA_PATH/ \
    --lora_weights YOUR_LORA_PATH \
    --test_data_path ./data/movie/test/test_5000.json \
    --result_json_data ./movie_result/movie.json
```

### Evaluate
```
# Taking Movie as an example
# Directly
python ./evaluate.py --input_dir ./book_result

 
# CI Augmented
python ./data/movie/adjust_ci.py --input_dir ./movie_result # Note that you need to have your own SASRec/DROS model (Specify the path in the code)
```

论文中也没说具体怎么实现保证生成的item 是

![](../img/Pasted%20image%2020240731200447.png)

# transformers 知识点
在`transformers`库中，`tokenizer`是文本处理的重要组成部分，它负责将原始文本转换为模型可以理解的格式（如token IDs），并进行解码以将生成的token IDs转换回文本。以下是一些常用的`tokenizer`操作：

## tokenizer
1. **初始化Tokenizer**:
   - 加载预训练模型的tokenizer。
     ```python
     from transformers import AutoTokenizer
     tokenizer = AutoTokenizer.from_pretrained("model_name")
     ```

2. **编码文本（Encoding）**:
   - 将文本转换为token IDs。
     ```python
     input_ids = tokenizer.encode("Hello, world!", return_tensors="pt")
     ```

3. **解码Token IDs（Decoding）**:
   - 将token IDs转换回文本。
     ```python
     decoded_text = tokenizer.decode(input_ids, skip_special_tokens=True)
     ```

4. **特殊Token处理**:
   - 获取或设置特殊token，如`[PAD]`、`[CLS]`、`[SEP]`等。
     ```python
     tokenizer.pad_token = '[PAD]'
     tokenizer.cls_token = '[CLS]'
     tokenizer.sep_token = '[SEP]'
     ```

5. **批量编码（Batch Encoding）**:
   - 同时对多条文本进行编码。
     ```python
     inputs = ["Hello, world!", "Transformers are amazing!"]
     batch_input_ids = tokenizer(inputs, return_tensors="pt", padding=True)
     ```

6. **填充（Padding）和裁剪（Truncation）**:
   - 对文本序列进行填充或裁剪以满足模型的输入要求。
     ```python
     padded_input_ids = tokenizer.encode("Hello", return_tensors="pt", padding='longest')
     truncated_input_ids = tokenizer.encode("Hello, this is a very long sentence that will be truncated", return_tensors="pt", truncation=True)
     ```

7. **设置填充和裁剪参数**:
   - 配置填充和裁剪的行为，如最大长度、方法等。
     ```python
     tokenizer.padding_side = "right"
     tokenizer.max_length = 512
     tokenizer.truncation_side = "right"
     ```

8. **获取词汇表大小（Vocabulary Size）**:
   - 获取tokenizer的词汇表大小。
     ```python
     vocab_size = tokenizer.vocab_size
     ```

9. **保存Tokenizer**:
   - 将tokenizer保存到文件系统。
     ```python
     tokenizer.save_pretrained("path_to_save_tokenizer")
     ```

10. **加载Tokenizer**:
    - 从文件系统加载tokenizer。
    ```python
    tokenizer = AutoTokenizer.from_pretrained("path_to_tokenizer")
    ```

11. **转换为模型输入**:
    - 将编码后的文本转换为模型的输入格式。
     ```python
     inputs = {
         "input_ids": batch_input_ids,
         "attention_mask": batch_attention_mask,  # 通常与input_ids一起生成
     }
     ```

12. **使用不同的解码策略**:
    - 使用不同的解码方法，如贪婪解码、束搜索（Beam Search）等。
     ```python
     generated_text = tokenizer.decode(input_ids, skip_special_tokens=True)
     ```

13. **处理多语言文本**:
    - 对于支持多语言的tokenizer，可以处理不同语言的文本。
     ```python
     tokenizer = AutoTokenizer.from_pretrained("multilingual_model_name")
     ```

14. **获取Token信息**:
    - 获取特定token的信息，如ID或文本。
     ```python
     token_id = tokenizer.convert_tokens_to_ids("[CLS]")
     token_text = tokenizer.convert_ids_to_tokens(token_id)
     ```

这些操作为使用`transformers`库进行NLP任务提供了强大的文本处理能力，确保文本数据可以有效地输入到模型中，并从模型输出中得到有意义的文本结果。


