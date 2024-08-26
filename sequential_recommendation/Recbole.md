
recbole.utils.case_study.full_sort_topk(_uid_series_, _model_, _test_data_, _k_, _device=None_)[[source]](https://recbole.io/docs/_modules/recbole/utils/case_study.html#full_sort_topk)[¶](https://recbole.io/docs/recbole/recbole.utils.case_study.html#recbole.utils.case_study.full_sort_topk "Permalink to this definition")

Calculate the top-k items’ scores and ids for each user in uid_series.

Note

The score of [pad] and history items will be set into -inf.

Parameters

- **uid_series** (_numpy.ndarray_) – User id series.
  
- **model** ([_AbstractRecommender_](https://recbole.io/docs/recbole/recbole.model.abstract_recommender.html#recbole.model.abstract_recommender.AbstractRecommender "recbole.model.abstract_recommender.AbstractRecommender")) – Model to predict.
  
- **test_data** ([_FullSortEvalDataLoader_](https://recbole.io/docs/recbole/recbole.data.dataloader.general_dataloader.html#recbole.data.dataloader.general_dataloader.FullSortEvalDataLoader "recbole.data.dataloader.general_dataloader.FullSortEvalDataLoader")) – The test_data of model.
  
- **k** (_int_) – The top-k items.
  
- **device** (_torch.device__,_ _optional_) – The device which model will run on. Defaults to `None`. Note: `device=None` is equivalent to `device=torch.device('cpu')`.
  

Returns

- topk_scores (torch.Tensor): The scores of topk items.
  
- topk_index (torch.Tensor): The index of topk items, which is also the internal ids of items.
  

Return type

tuple

```python
# @Time   : 2020/12/25
# @Author : Yushuo Chen
# @Email  : chenyushuo@ruc.edu.cn
# UPDATE
# @Time   : 2020/12/25
# @Author : Yushuo Chen
# @email  : chenyushuo@ruc.edu.cn

"""
recbole.utils.case_study
#####################################
"""

import numpy as np
import torch

from recbole.data.interaction import Interaction


[docs]@torch.no_grad()
def full_sort_scores(uid_series, model, test_data, device=None):
    """Calculate the scores of all items for each user in uid_series.

    Note:
        The score of [pad] and history items will be set into -inf.

    Args:
        uid_series (numpy.ndarray or list): User id series.
        model (AbstractRecommender): Model to predict.
        test_data (FullSortEvalDataLoader): The test_data of model.
        device (torch.device, optional): The device which model will run on. Defaults to ``None``.
            Note: ``device=None`` is equivalent to ``device=torch.device('cpu')``.

    Returns:
        torch.Tensor: the scores of all items for each user in uid_series.
    """
    device = device or torch.device("cpu")
    uid_series = torch.tensor(uid_series)
    uid_field = test_data.dataset.uid_field
    dataset = test_data.dataset
    model.eval()

    if not test_data.is_sequential:
        input_interaction = dataset.join(Interaction({uid_field: uid_series}))
        history_item = test_data.uid2history_item[list(uid_series)]
        history_row = torch.cat(
            [torch.full_like(hist_iid, i) for i, hist_iid in enumerate(history_item)]
        )
        history_col = torch.cat(list(history_item))
        history_index = history_row, history_col
    else:
        _, index = (dataset.inter_feat[uid_field] == uid_series[:, None]).nonzero(
            as_tuple=True
        )
        input_interaction = dataset[index]
        history_index = None

    # Get scores of all items
    input_interaction = input_interaction.to(device)
    try:
        scores = model.full_sort_predict(input_interaction)
    except NotImplementedError:
        input_interaction = input_interaction.repeat_interleave(dataset.item_num)
        input_interaction.update(
            test_data.dataset.get_item_feature().to(device).repeat(len(uid_series))
        )
        scores = model.predict(input_interaction)

    scores = scores.view(-1, dataset.item_num)
    scores[:, 0] = -np.inf  # set scores of [pad] to -inf
    if history_index is not None:
        scores[history_index] = -np.inf  # set scores of history items to -inf

    return scores


[docs]def full_sort_topk(uid_series, model, test_data, k, device=None):
    """Calculate the top-k items' scores and ids for each user in uid_series.

    Note:
        The score of [pad] and history items will be set into -inf.

    Args:
        uid_series (numpy.ndarray): User id series.
        model (AbstractRecommender): Model to predict.
        test_data (FullSortEvalDataLoader): The test_data of model.
        k (int): The top-k items.
        device (torch.device, optional): The device which model will run on. Defaults to ``None``.
            Note: ``device=None`` is equivalent to ``device=torch.device('cpu')``.

    Returns:
        tuple:
            - topk_scores (torch.Tensor): The scores of topk items.
            - topk_index (torch.Tensor): The index of topk items, which is also the internal ids of items.
    """
    scores = full_sort_scores(uid_series, model, test_data, device)
    return torch.topk(scores, k)
```


# evaluator
https://recbole.io/docs/_modules/recbole/evaluator/evaluator.html#Evaluator
```python
"""
recbole.evaluator.evaluator
#####################################
"""

from recbole.evaluator.register import metrics_dict
from recbole.evaluator.collector import DataStruct
from collections import OrderedDict


[docs]class Evaluator(object):
    """Evaluator is used to check parameter correctness, and summarize the results of all metrics."""

    def __init__(self, config):
        self.config = config
        self.metrics = [metric.lower() for metric in self.config["metrics"]]
        self.metric_class = {}

        for metric in self.metrics:
            self.metric_class[metric] = metrics_dict[metric](self.config)

[docs]    def evaluate(self, dataobject: DataStruct):
        """calculate all the metrics. It is called at the end of each epoch

        Args:
            dataobject (DataStruct): It contains all the information needed for metrics.

        Returns:
            collections.OrderedDict: such as ``{'hit@20': 0.3824, 'recall@20': 0.0527, 'hit@10': 0.3153, 'recall@10': 0.0329, 'gauc': 0.9236}``

        """
        result_dict = OrderedDict()
        for metric in self.metrics:
            metric_val = self.metric_class[metric].calculate_metric(dataobject)
            result_dict.update(metric_val)
        return result_dict

```python
def calculate_metric(self, dataobject):  
	pos_index, pos_len = self.used_info(dataobject)  
	result = self.metric_info(pos_index, pos_len) # 对pos_index 进行加和。  
	metric_dict = self.topk_result("recall", result)  
	return metric_dict
```

```python
def topk_result(self, metric, value):  
"""Match the metric value to the `k` and put them in `dictionary` form.  

Args:  
metric(str): the name of calculated metric.  
value(numpy.ndarray): metrics for each user, including values from `metric@1` to `metric@max(self.topk)`.  
  
Returns:  
dict: metric values required in the configuration.  
"""  
metric_dict = {}  
avg_result = value.mean(axis=0)  
for k in self.topk:  
key = "{}@{}".format(metric, k)  
metric_dict[key] = round(avg_result[k - 1], self.decimal_place)  
return metric_dict
```

![[Pasted image 20230924153102.png]]
![[Pasted image 20230924153137.png]]
![[Pasted image 20230924163751.png]]


## 模型参数

|   |   |
|---|---|
|BERT4Rec|learning_rate in [0.0003, 0.0005, 0.001, 0.003, 0.005]|



## 调参结果

### gru

![](C:\Users\86155\AppData\Roaming\Typora\typora-user-images\image-20231130210043836.png)

### narm

![image-20231130212342479](C:\Users\86155\AppData\Roaming\Typora\typora-user-images\image-20231130212342479.png)

## SAS

![image-20231130214016933](C:\Users\86155\AppData\Roaming\Typora\typora-user-images\image-20231130214016933.png)

### Bert




### 规律总结

GRU 和 NARM 相比习惯于推荐流行度偏高的，命中的流行度也偏高


## Amazon_magazine
#### bert
![[Pasted image 20231207155327.png]]
#### gru
![[Pasted image 20231207161536.png]]
#### narm
![[Pasted image 20231207161812.png]]
#### sas
![[Pasted image 20231207162742.png]]


{'ft_ratio': 0, 'learning_rate': 0.008694123647198243}

{'ft_ratio': 0.5, 'learning_rate': 0.00918699476394836}


## 规律
### ml-1m
#### bert vs sas
命中用户的流行度差不多
sas 习惯于推荐流行度高的 item， 但命中的物品的流行度却偏低

#### gru vs sas
命中用户的流行度差不多
gru 推荐的item 流行度偏高，命中的也偏高。
#### gru vs narm
narm 命中用户的流行度偏高
gru 推荐的item 流行度偏高，命中的也偏高。

### Amazon_Magazine

#### bert vs sas
bert命中用户的流行度偏高
bert和sas推荐的物品流行度差不多(如果算均值还是bert更高)， 但命中的物品的流行度sas 偏低

#### gru vs sas
命中用户的流行度差不多
gru 推荐的item 流行度偏低，命中的也偏低。
#### gru vs narm
gru 命中用户的流行度偏高
narm 推荐的item 流行度偏高，命中的偏低。


### Amazon_Beauty 调参结果

#### bert

best params:  {'ft_ratio': 0.1, 'learning_rate': 0.003}

#### sas
best params:  {'learning_rate': 0.0003}
best result: 
{'model': 'SASRec', 'best_valid_score': 0.0167, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.0496), ('mrr@10', 0.0167), 

#### narm
best params:  {'learning_rate': 0.001, 'num_layers': 2}
best result: 
{'model': 'NARM', 'best_valid_score': 0.0204, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.0424), ('mrr@10', 0.0204), 
#### gru

best params:  {'learning_rate': 0.001, 'num_layers': 1}
best result: 
{'model': 'GRU4Rec', 'best_valid_score': 0.018, 'valid_score_bigger': True, 'best_valid_result': OrderedDict([('recall@10', 0.0379),



## 数据统计部分
RecBole-master/recbole/data/utils.py/data_preparation

```python
with open(fr'E:\RecBole-master_test\RecBole-master\dataset_map\{dataset.dataset_name}.json', 'w') as file:  
	json.dump(dataset.field2token_id, file)
```
**Dataset:** amazon-books_seq

**Data filtering:** delete interactive records with rating less than 3

**K-core filtering:** delete inactive users or unpopular items with less than 10 interactions

**Evaluation method:** chronological arrangement, leave one out split data set and full sorting

**Evaluation metric:** Recall@10, NGCG@10, MRR@10, Hit@10, Precision@10

## amazon-toys-games
### gru
![[Pasted image 20240326001819.png]]
2048 2048
### narm
![[Pasted image 20240326001859.png]]
2048 2048
### sas
![[Pasted image 20240328203655.png]]
2048 2048
### bert
![[Pasted image 20240326124805.png]]
1024 1024
### s3
## amazon_beaty
### gru
512 512
![[Pasted image 20240328203802.png]]
## narm
512 512
![[Pasted image 20240328203835.png]]
### bert
256 256
![[Pasted image 20240329194744.png]]
### sas
512 512
![[Pasted image 20240330105537.png]]
### s3
256 256

