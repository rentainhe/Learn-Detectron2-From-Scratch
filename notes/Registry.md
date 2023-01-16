## Registry：通过注册机制优雅地调用代码中的不同组件

本文是关于detectron2下yacs配置系统的一个拓展文章，同样希望通过几个实际应用的场景，对detectron2下**yacs配置系统**所搭配的一些使用作进一步地介绍，以下是本文的目录, 大家可以选择性地阅读自己想看的章节, 也可以按照顺序阅读：

## Contents
- [Registry：通过注册机制优雅地调用代码中的不同组件](#registry通过注册机制优雅地调用代码中的不同组件)
- [Contents](#contents)
  - [注册机制的简单介绍](#注册机制的简单介绍)
  - [为什么我们需要注册机制？](#为什么我们需要注册机制)
  - [注册机制的简单实现](#注册机制的简单实现)
  - [Detectron2下注册机制的基本用法](#detectron2下注册机制的基本用法)
    - [创建一个注册实例来存放相关的module](#创建一个注册实例来存放相关的module)
    - [注册模型](#注册模型)
  - [Brief Summary](#brief-summary)
- [Reference](#reference)

### 注册机制的简单介绍

注册机制的核心功能是：**以全局字典的形式，存储用户定义的类和函数**

### 为什么我们需要注册机制？

要理解这个问题，首先我们要知道字典形式会给我们带来什么好处，我们先设想一个应用场景，我们希望使用config中的某个key，来控制调用的模型，一个最naive的实现是，将我们core code下实现的所有模型，写入一个build函数里，并且通过config判断调用的是哪个模型，一个简单的伪代码如下:
```python
from model import RetinaNet, FCOS

def build_model(cfg):
    if cfg.MODEL.NAME == "retinanet":
        return RetinaNet()
    elif cfg.MODEL.NAME == "fcos":
        return FCOS()
    else:
        raise NotImplementedError("only support retinanet / fcos now.")
```

此时我们可以通过修改我们config下的`NAME`来调用不同的模型
```python
cfg.MODEL.NAME = "retinanet"

# 得到retinanet模型
model = build_model(cfg)
```

这样的实现很简单直观，但是**拓展性太差**，举个例子，当用户希望通过config控制自己新构建一个新的模型的时候，需要hack的部分很多：
```python
from model import RetinaNet, FCOS

from my_net import MyNet  # 添加一个新的模型

def build_model(cfg):
    if cfg.MODEL.NAME == "retinanet":
        return RetinaNet()
    elif cfg.MODEL.NAME == "fcos":
        return FCOS()
    
    # hack进build_model函数
    elif cfg.MODEL.NAME == "my_net":
        return MyNet()
    
    else:
        raise NotImplementedError("only support retinanet / fcos now.")
```

更加优雅的形式是：我们希望存在一个**字典(model-factory)**，这个字典具备以下几个功能：
- **每个key对应着我们定义好的一个模型**
- **可以方便地logging这个字典中已经存在的模型**
- **可以方便地添加模型到字典中**

如果我们存在这样的一个**字典(model-factory)**，那么我们调用模型的流程就会得到极大的简化，一个简单的伪代码如下：
```python
from model import model_factory

print(model_factory)  # 输出的是一个dict
"""
{
    "fcos": FCOS,
    "retinanet": RetinaNet,
}
"""
```
调用`model_factory`中存在的模型
```python
cfg.MODEL.NAME = "retinanet"
model = model_factory(cfg.MODEL.NAME)(args)
```
向`model_factory`中添加模型
```python
from model import model_factory
from my_net import MyNet

model_factory["my_net"] = MyNet()
```

### 注册机制的简单实现

要实现一个满足我们上述功能的字典其实非常简单，一个简单可用的代码如下：

```python
from typing import Any
from tabulate import tabulate

class Registry(object):
    def __init__(self, name: str):
        # 我们这个字典的name
        self._name = name

        # 定义一个 name to object 的mapping
        self._obj_map = {}
    
    def register(self, name: str, obj: Any):
        # 判断我们是否已经注册过这个name，不能重复注册
        assert name not in self._obj_map, "An object named '{}' was already registered in '{}' registry!".format(name, self._name)

        # 将对应的object添加到字典中
        self._obj_map[name] = obj

    def get(self, name: str):
        ret = self._obj_map.get(name)
        # 判断是否存在，不存在需要报错提示
        if ret is None:
            raise KeyError(
                "No object named '{}' found in '{}' registry!".format(name, self._name)
            )
        return ret

    # 这边用tabulate美化一下print的输出结果
    def __repr__(self) -> str:
        table_headers = ["Names", "Objects"]
        table = tabulate(
            self._obj_map.items(), headers=table_headers, tablefmt="fancy_grid"
        )
        return "Registry of {}:\n".format(self._name) + table
```
我们通过定义`Registry`类，实现了以下几个对应的功能：
- 在初始化时可以通过显示传入`name`参数，对我们这个`factory`命名
- 通过实现`register`函数，可以将用户定义的新模型注册到我们定义好的`Registry`类中
- 通过`get`函数传入`object name`来调用得到对应的`object`
- 最后通过自定义`__repr__`函数美化一下我们的输出结果，并且可以**List**出我们已经注册好的所有模块

在定义好`Registry`类后，我们可以极其便捷地添加新定义好的模型，我们用`torchvision`下的模型举例:
```python
from torchvision.models.resnet import resnet18, resnet34

model_factory = Registry("model")
model_factory.register("resnet18", resnet18)
model_factory.register("resnet34", resnet34)
print(model_factory)

>>>
Registry of model:
╒══════════╤═══════════════════════════════════════╕
│ Names    │ Objects                               │
╞══════════╪═══════════════════════════════════════╡
│ resnet18 │ <function resnet18 at 0x7f72ffdd43b0> │
├──────────┼───────────────────────────────────────┤
│ resnet34 │ <function resnet34 at 0x7f72ffdd4440> │
╘══════════╧═══════════════════════════════════════╛
```
我们也可以很方便地通过`model_factory`去调用模型:
```python
import torch

model = model_factory.get("resnet18")(num_classes=10)  # 可以传入自己想修改的参数

# 随机定义一组输入
img = torch.randn(1, 3, 224, 224)
preds = model(img)
print(preds.shape)
>>> torch.Size([1, 10])
```

### Detectron2下注册机制的基本用法
注册机制在很多优质项目中都实现过，包括mmcv，detectron2等，这边简单列举一下detectron2下注册机制的基本用法，对于其中一些python syntax的内容（例如装饰器等）我们后续会专门推出文章介绍相关的基础知识

#### 创建一个注册实例来存放相关的module
```python
from fvcore.common.registry import Registry

# 创建一个model factory来存放模型相关的函数和类
MODEL_FACTORY = Registry("MODEL")
```

#### 注册模型
有两种方式可以将模型注册到我们定义好的`MODEL_FACTORY`中:

- 显式地调用`register()`方法
```python
from torchvision.models.resnet import resnet18, resnet34

MODEL_FACTORY.register(resnet18)
MODEL_FACTORY.register(resnet34)

# 打印已注册的模型
print(MODEL_FACTORY)
>>>
Registry of MODEL:
╒══════════╤═══════════════════════════════════════╕
│ Names    │ Objects                               │
╞══════════╪═══════════════════════════════════════╡
│ resnet18 │ <function resnet18 at 0x7f7b369b9560> │
├──────────┼───────────────────────────────────────┤
│ resnet34 │ <function resnet34 at 0x7f7b369b95f0> │
╘══════════╧═══════════════════════════════════════╛
```

- 通过**装饰器**注册用户定义的模型

```python
import torch.nn as nn

@MODEL_FACTORY.register()
class MyNet(nn.Module):
    def __init__(self):
        self.conv = nn.Conv2d(3, 16, 1, 1, 1)
    
    def forward(self, x):
        return self.conv(x)

# 打印已注册的模型
print(MODEL_FACTORY)
>>>
Registry of MODEL:
╒══════════╤═══════════════════════════════════════╕
│ Names    │ Objects                               │
╞══════════╪═══════════════════════════════════════╡
│ resnet18 │ <function resnet18 at 0x7f07ee70e710> │
├──────────┼───────────────────────────────────────┤
│ resnet34 │ <function resnet34 at 0x7f07ee70e7a0> │
├──────────┼───────────────────────────────────────┤
│ MyNet    │ <class '__main__.MyNet'>              │
╘══════════╧═══════════════════════════════════════╛
```

### Brief Summary
灵活地使用注册机制可以**很方便地管理代码中的核心组件**，通过定义不同的Registry实例，来管理代码中的不同组件，例如我可以通过定义`OPTIM_FACTORY`, `DATASET_FACTORY`, `MODEL_FACTORY`来分别管理优化器，数据集以及模型。在后续的文章中我们会结合detectron2下的`configurable`组件更加深入介绍一下detectron2下的完整的模型调用流程。

## Reference
- [fvcore.common.registry](https://github.com/facebookresearch/fvcore/blob/main/fvcore/common/registry.py)
