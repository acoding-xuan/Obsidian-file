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


![](../img/Pasted%20image%2020240801161357.png)