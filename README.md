This is for Web3 lab

The testingdata fold is a small set of data based on the courseware Page46

In this procedure, -1 represents no record

### Item-based recommendation

> 2021.2.24 17:35 UTC+08

添加了基于物品的预测，输出文件存储在/testingdata/Item_based_recommend_matrix.npy

### User-Based recommendation

> 2021.2.26 18:03 UTC+08

添加了基于用户的预测：

```python
    def __init__(self, k=10, useRelation=True):
        self.dataDir = '../testingdata/'
        self.testDataPath = '../testingdata/testing.dat'
        self.K = k
        self.useRelation = useRelation
```

默认k=10，输出地址根据以上初始化数据更改

有如下命令行参数：

- `-k`: 指定k值
- `--relation`：存在此flag则使用权重，否则不使用权重

例：

```sh
python ./user_based_predict.py -k 2 --relation # 使用关系权重，2-最近邻推荐
```