[Markdown 基本语法 | Markdown 官方教程](https://markdown.com.cn/basic-syntax/)

[扩展语法]https://markdown.com.cn/extended-syntax/tables.html
# 任务列表
- [x] Write the press release
- [x] a
- [x] b
- [x] c
# 粗体（Bold）和 斜体

要加粗文本，请在单词或短语的前后各添加两个星号（asterisks）或下划线（underscores）。如需加粗一个单词或短语的中间部分用以表示强调的话，请在要加粗部分的两侧各添加两个星号（asterisks）。

| Markdown语法                   | HTML                                      | 预览效果                       |
| ---------------------------- | ----------------------------------------- | -------------------------- |
| `I just love **bold text**.` | `I just love <strong>bold text</strong>.` | I just love **bold text**. |
| `I just love __bold text__.` | `I just love <strong>bold text</strong>.` | I just love **bold text**. |
| `Love**is**bold`             | `Love<strong>is</strong>bold`             | Love**is**bold             |

## 粗体（Bold）用法最佳实践
Markdown 应用程序在如何处理单词或短语中间的下划线上并不一致。为兼容考虑，在单词或短语中间部分加粗的话，请使用星号（asterisks）。

| ✅  Do this       | ❌  Don't do this |
| ---------------- | ---------------- |
| `Love**is**bold` | `Love__is__bold` |

## 斜体（Italic）

要用斜体显示文本，请在单词或短语前后添加一个星号（asterisk）或下划线（underscore）。要斜体突出单词的中间部分，请在字母前后各添加一个星号，中间不要带空格。

#   带有其它元素的块引用

块引用可以包含其他 Markdown 格式的元素。并非所有元素都可以使用，你需要进行实验以查看哪些元素有效。

```
> #### The quarterly results look great!
>
> - Revenue was off the chart.
> - Profits were higher than ever.
>
>  *Everything* is going according to **plan**.
```

渲染效果如下：

>  ####  The quarterly results look great!
> 
> - Revenue was off the chart.
> - Profits were higher than ever.
> 
> _Everything_ is going according to **plan**.




  # 分割线
---

