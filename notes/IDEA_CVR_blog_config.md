## YACS vs LazyConfig: Detectron2下的两代配置系统介绍

为了方便阅读, 这边是相关的content的index, 大家可以根据自己的兴趣阅读相关的内容
### Contents
- [YACS vs LazyConfig: Detectron2下的两代配置系统介绍](#yacs-vs-lazyconfig-detectron2下的两代配置系统介绍)
  - [Contents](#contents)
  - [Introduction](#introduction)
  - [Basic Usage of YACS](#basic-usage-of-yacs)
    - [如何通过yacs创建config](#如何通过yacs创建config)
    - [Config System的核心需求: 增删查改](#config-system的核心需求-增删查改)
    - [如何合理地更新和保护定义好的Config](#如何合理地更新和保护定义好的config)
    - [Summary of yacs](#summary-of-yacs)
  - [Basic Usage of LazyConfig](#basic-usage-of-lazyconfig)
    - [LazyConfig Introduction Example](#lazyconfig-introduction-example)
    - [What is "lazy" in LazyConfig?](#what-is-lazy-in-lazyconfig)
    - [LazyConfig: 在保证基本功能的前提下更加灵活](#lazyconfig-在保证基本功能的前提下更加灵活)
  - [LazyConfig与Yacs的简单对比](#lazyconfig与yacs的简单对比)
  - [Summary](#summary)
  - [References](#references)

### Introduction
配置系统(Config System)作为open-source library的一个很重要的组件, 一个好的配置系统能给用户的使用体验带来的好处是很明显的:
- 方便修改大部分实验所需要的超参
- 比较方便用户管理和回溯每次实验
- 让项目结构更加清晰

所以作为一个open-source codebase, 首先需要明确自身需要采用哪套Config System, 这是一切的基础, 而`yacs`和`LazyConfig`作为detectron2下使用过的两套配置系统, 这篇文章讲针对这两套配置系统进行相关的介绍, 以及这两套系统背后对应的一些开源思想, 文章不会从特别细枝末节的功能开始讲解, 而是从一些**常见的使用场景**出发, 希望让读者读起来轻松一些, 有些必要的功能可以在后续使用的时候接触到了再进行相关的了解. 并且会给出一些简单的example, 帮助大家理解与感受d2下第二代LazyConfig为用户带来的便捷性.

大家可能已经注意到, detectron2的最新版本下已经用了最新的LazyConfig完全替代了之前的yacs版本的`.yaml`格式的config, 但是为了兼容早期的baseline, 并没有完全删除yacs的使用, 呈现出了兼容的状态, 我们基于detectron2开发的 [detrex](https://github.com/IDEA-Research/detrex), 完全采用了LazyConfig的配置系统, 以及搭配LazyConfig的training engine: [lazyconfig_train_net.py](https://github.com/IDEA-Research/detrex/blob/main/tools/train_net.py), 在新的这套配置下, 一个完整的训练代码只有百行不到, 并且config的可读性相对于之前的yacs来说有了本质的提升, 可以**极大地减少用户阅读代码的负担**. 这篇blog表达的内容有限, 只是作为第一篇blog让大家更好的理解LazyConfig以及yacs的区别与好处, 并且帮助大家更好地上手detectron2与detrex. 后续还会有更多的blog对detrex和detectron2的设计做更多的介绍.


### Basic Usage of YACS
`yacs`是4年前rbg大佬团队开发的用于`Detectron`, `maskrcnn-benchmark`以及早期`Detectron2`的一个轻量化配置系统, 其使用与可读性较好的`.yaml`文件有着紧密的联系. 在讨论一个config system的时候, 我们首先可以了解一下这个config的**基本格式**.

#### 如何通过yacs创建config
所有的配置在`yacs`下都可以通过`CfgNode`这个类来定义, 如果我们希望得到下面这样`.yaml`排版格式的config配置:
```python
MODEL:
  BACKBONE: R50
  NORM: BN
NAME: Test
```
我们可以看到config最外层总共有两个参数, 一个`MODEL`一个`NAME`, 并且MODEL下还有两个子节点`BACKBONE`以及`NORM`, 那么我们用`yacs`便可以轻松地创建出这样层级关系的config:

```python
from yacs.config import CfgNode as CN

cfg = CN()
cfg.NAME = "Test"

cfg.MODEL = CN()
cfg.MODEL.BACKBONE = "R50"
cfg.MODEL.NORM = "BN"
```
当我们`print(cfg)`, 就可以得到刚刚想要的config格式了:
```python
MODEL:
  BACKBONE: R50
  NORM: BN
NAME: Test
```

#### Config System的核心需求: 增删查改
配置系统的一个基本需求当然是希望用户可以很方便地对其参数进行访问与更新, 在`yacs`中, 我们很直观对这些配置系统进行相应地增删查改, 依旧以之前我们创建的config为例:
```python
from yacs.config import CfgNode as CN

cfg = CN()
cfg.NAME = "Test"

cfg.MODEL = CN()
cfg.MODEL.BACKBONE = "R50"
cfg.MODEL.NORM = "BN"
```
现在我们定义了一组我们需要仓库需要的配置, 我们可以对其进行以下几个基本操作:
```python
# 1. 访问某个参数
print(cfg.NAME)
>>> Test

# 2. 修改某个参数
cfg.NAME = "Update"
print(cfg.NAME)
>>> Update

# 3. 新增一个参数
cfg.NEW_PARAM = "New"
print(cfg.NEW_PARAM)
>>> New

```
删除一个参数倒是没有特别的方法, 目前看来只能从定义的部分直接删除, 但是`cfg.clear()`方法是可以将所有的配置清空

#### 如何合理地更新和保护定义好的Config

我们了解完了一个config配置的定义和基本操作(增删查改)后, 随之而来的问题便是, 面对这么一个灵活的配置系统, 应该如何有效避免我们在代码的某处不小心对其进行了修改, 从而导致实验出错, `yacs`提供了几个接口帮助我们尽可能地避免这个问题

**`CfgNode.clone()`: 返回一份复制的config, 对其进行修改不会影响你最初定义的config内容**

```python
from yacs.config import CfgNode as CN

cfg = CN()
cfg.NAME = "Test"

# 将cfg中的内容完整地复制给new_cfg
new_cfg = cfg.clone()
print(new_cfg.NAME)
>>> Test

# 对new_cfg进行修改, 不会影响原cfg中的参数
new_cfg.NAME = "New"
print(new_cfg.NAME)
>>> New

print(cfg.NAME)
>>> Test
```
`.clone()`方法, 一方面可以作为**对原来config的一种保护**, 另一方面也是**快速创建一个新的config的方式**.

**`CfgNode.freeze()`与`CfgNode.defrost()`: 作为一个开关, 可以保护定义好的params不可被修改**
对于配置参数的保护, `yacs`提供了另一种方法, 即可以通过`freeze()`方法冻结整个配置系统, 使其在freeze之后无法进行任何修改, 这也保证了整个过程中我们的配置不会有任何的改动
```python
from yacs.config import CfgNode as CN

cfg = CN()
cfg.NAME = "Test"

# 保护cfg下定义的所有参数不可被修改
cfg.freeze()
cfg.NAME = "New"
>>> AttributeError: Attempted to set NAME to New, but CfgNode is immutable

# 可以通过defrost让超参重新变得可修改
cfg.defrost()
cfg.NAME = "New"
print(cfg.NAME)
>>> New
```

#### Summary of yacs

`yacs`作为一个及其lightweight的config system, 可以说是麻雀虽小五脏俱全, 其中一些基本的应用如果熟悉了, 有助于帮助到大家平时的实验:
- config的增删查改
- 如何load某个`.yaml`文件中的config
- 如何dump出config的内容到某个`.yaml`文件
- 如何通过命令行传参修改config中的内容, 可以直接通过`bash`修改参数

虽然`yacs`的功能足够全面了, 但是其难免会存在部分问题, `detectron2`的早期设计, 旨在尽可能整套训练所涉及到的内容都通过config system控制, 但是`yacs`在本身的语法上有所限制, 并且detectron2最初将所有的基本config都定义到了 [detectron2/config/default.py](https://github.com/facebookresearch/detectron2/blob/main/detectron2/config/defaults.py) 下, 并在训练的过程中会将所有config都dump下来, 虽然带有一部分的注释, 但是在可读性上依旧不是很好, 存在一部分臃肿冗余的参数.

### Basic Usage of LazyConfig

在最近的detectron2更新下, 注意到引入了一种LazyConfig机制, 全面替换之前的`yacs` config system. 在基本功能都有所保证的情况下, LazyConfig相比于之前的`yacs`具有了**更简洁, 更准确, 更高效**的特性. detectron2在结构设计上不断能有新的简洁的设计, 在代码设计上不断地做减法, 让人很敬佩, 接下来这边简单介绍一下LazyConfig的使用, 帮助新接触detectron2的用户熟悉一下这套配置系统.


#### LazyConfig Introduction Example

我们首先通过一个例子入手, 直观地比较一下两种config方式的区别, 假设我们现在的需求是, **需要一个简单的卷积层, 并且我们我们需要可以灵活地控制其中的参数, 例如`stride`, `kernel_size`, `padding`等**, 我们对比一下两套配置方案是如何完成这样的事情:

**yacs Config System**
在yacs下, 需要在config中指定**所有**我们需要调整的参数, 这意味着我们的config中如果面对需要调整的新的参数, 就必须新增一项内容
```python
import torch.nn as nn
from yacs.config import CfgNode as CN

# 创建一个config node, 并且罗列我们需要控制的参数
cfg = CN()
cfg.in_channels = 16
cfg.out_channels = 16
cfg.kernel_size = 3
cfg.stride = 1
cfg.padding = 1


# 通过cfg传入超参, 实例化一个卷积层
conv_layer = nn.Conv2d(
    in_channels=cfg.in_channels,
    out_channels=cfg.out_channels,
    kernel_size=cfg.kernel_size,
    stride=cfg.stride,
    padding=cfg.padding,
)

# 打印cfg
print(cfg)
>>> in_channels: 16
    kernel_size: 3
    out_channels: 16
    padding: 1
    stride: 1

# 打印conv_layer
print(conv_layer)
>>> Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
```

**LazyConfig System**
在LazyConfig System下, 以上一个10行代码才能完成的事情, 只需要2行即可:
```python
import torch.nn as nn
from detectron2.config import LazyCall, instantiate

# 通过LazyCall创建一个config对象
conv_config = LazyCall(nn.Conv2d)(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)

# 通过instantiate实例化
conv_layer = instantiate(conv_config)

# 打印conv_layer
print(conv_layer)
>>> Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
```

#### What is "lazy" in LazyConfig?
LazyConfig中的核心思想是Lazy, Lazy表示一种**延迟**状态, 这里的表述可能会让人云里雾里, 我们结合上一个小节的example, 对lazyconfig中的核心思想做进一步解释, 我们可以看到在上文创建卷积层的这个example中, 我们新接触到了两个函数, `LazyCall`以及`instantiate`, 这也是LazyConfig中的核心用法, 我们和直接通过`nn.Conv2d`创建一层卷积层进行一下简单的对比:
```python
import torch.nn as nn
from detectron2.config import LazyCall, instantiate

# 通过LazyCall创建一个config对象
conv_config = LazyCall(nn.Conv2d)(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
conv_layer = instantiate(conv_config)

# 直接通过nn.Conv2d创建conv_layer
conv_layer = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
```
用最通俗的语言表达就是, LazyCall将实例化这个过程拆分成了两个步骤, 将一步即可实例化的过程拆分成了两个状态, 也就是所谓的"延迟":
1. 通过`LazyCall`将需要实例化的对象包裹一下, 传入对应的参数
2. 通过`instantiate`来进行实例化

我们可以打印一下使用`LazyCall`包裹的对象, 观察一下具体的内容:
```python
import torch.nn as nn
from detectron2.config import LazyCall, instantiate

# 通过LazyCall创建一个config对象
conv_config = LazyCall(nn.Conv2d)(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)

print(conv_config)
>>> {'in_channels': 16, 'out_channels': 16, 'kernel_size': 3, 'stride': 1, 'padding': 1, '_target_': <class 'torch.nn.modules.conv.Conv2d'>}
```
我们可以观察到LazyCall是返回给我们一个字典的结构形式, 当然其不是真正的dict, 我们去打印type可以发现是一个`omegaconf.dictconfig.DictConfig`对象, 我们暂时可以不需要去了解, 有感兴趣的小伙伴可以去看看`omegaconf`这个repo. 这个`DictConfig`对象包含了几个key:
- 一个特殊的名为`_target_`的key, 表示我们需要实例化的类, 在这个例子中是`torch.nn.modules.conv.Conv2d`
- 实例化`_target_`所需要的所有的参数, 都以`key-value`的形式保存在这个`DictConfig`对象中

那么所谓的`Lazy`顾名思义, **在我们真正实例化这个类之前, 它将一直以`DictConfig`对象的形式存在**. 也就是说我们不马上去实例化这个对象, **在需要的时候再调用`instantiate`函数进行实例化**即可. 这种形式给我们带来了几个好处:
- 我们可以**在实例化这个对象前, 对其任意的value进行修改**, 甚至是`_target_`的value
- **可读性好**, 从读者阅读的角度而言, 和直接调用`nn.Conv2d`的区别不大, 只是额外wrap了一层, 方便我们可以通过config去控制修改超参


#### LazyConfig: 在保证基本功能的前提下更加灵活
在解释完什么是Lazy之后, 我们直接进入最基本的使用, 让我们看看LazyConfig是如何满足一个配置系统的基本需求的:

**config的修改**
如果我们需要得到一个卷积核大小为5的卷积层
```python
import torch.nn as nn
from detectron2.config import LazyCall, instantiate

# 通过LazyCall创建一个config对象
conv_config = LazyCall(nn.Conv2d)(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)

# 修改kernel_size
conv_config.kernel_size = 5

# 实例化这个对象, 得到kernel_size=5的卷积层
conv_layer = instantiate(conv_config)
print(conv_layer)
>>> Conv2d(16, 16, kernel_size=(5, 5), stride=(1, 1), padding=(1, 1))
```

**config的增加**
如果我们需要将这个卷积替换为分组卷积, 意味着需要新指定一个`groups`参数
```python
import torch.nn as nn
from detectron2.config import LazyCall, instantiate

# 通过LazyCall创建一个config对象
conv_config = LazyCall(nn.Conv2d)(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)

# 新增一个groups参数, 用来控制分组卷积的组数
conv_config.groups = 16

# 实例化这个对象, 得到groups=16的分组卷积
conv_layer = instantiate(conv_config)
print(conv_layer)
>>> Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16)
```
LazyCall的flexible也会带来一个问题, 当你新增一个不属于需要实例化的那个类的参数的时候, 会报相应的错误, LazyCall不存在超参检查的机制, 需要用户对传入的参数足够了解:
```python
import torch.nn as nn
from detectron2.config import LazyCall, instantiate

# 通过LazyCall创建一个config对象
conv_config = LazyCall(nn.Conv2d)(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)

# 新增了一个不属于nn.Conv2d的参数并且实例化
conv_config.embed_dim = 16
conv_layer = instantiate(conv_config)
```
会提示以下错误:
```bash
Error when instantiating torch.nn.modules.conv.Conv2d!
TypeError: __init__() got an unexpected keyword argument 'embed_dim'
```


**config的删除**
如果我们不需要传入padding这个参数, 使用默认值, 但是我们原有的config中已经指定了的话, 那么我们可以在实例化之前`del`了这个参数, 而在yacs中需要删除原始config中对应的字段.
```python
import torch.nn as nn
from detectron2.config import LazyCall, instantiate

# 通过LazyCall创建一个config对象
conv_config = LazyCall(nn.Conv2d)(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)

# 删除padding这个参数
del conv_config.padding

# 实例化这个对象
conv_layer = instantiate(conv_config)
print(conv_layer)
>>> Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1))
```

**直接替换我们实例化的对象**
更有趣的功能是, 如果我们不需要Conv2d了, 需要一个一维卷积(Conv1d), 我们可以直接修改`_target_`参数:
```python
import torch.nn as nn
from detectron2.config import LazyCall, instantiate

# 通过LazyCall创建一个config对象
conv_config = LazyCall(nn.Conv2d)(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)

# 修改target参数, 将实例化的对象改为Conv1D
conv_config._target_ = nn.Conv1D

# 实例化这个对象, 得到groups=16的分组卷积
conv_layer = instantiate(conv_config)
print(conv_layer)
>>> Conv1d(16, 16, kernel_size=(3,), stride=(1,), padding=(1,))
```
非常神奇的发现我们实例化的对象从Conv2d变成了Conv1d

### LazyConfig与Yacs的简单对比
在上面的内容中, 我们简单介绍了一下`yacs`和`LazyConfig`的基本使用, 我们可以很直观地看出来, 很多时候我们在使用yacs作为config system的时候, 我们在构建整套codebase需要的组件时, 我们的config会变得**越发臃肿且可读性变差**, 假设我们需要控制codebase下的多个模型, 但真正在执行程序的时候只会运行其中一个模型, 以`Conv2d`和`MultiheadAttention`为例, 我们首先要面对的一个麻烦事就在于, 需要将这两个模型可控制的所有参数都记录到config中:

```python
import torch.nn as nn
from yacs.config import CfgNode as CN

# 创建一个config node, 并且罗列我们需要控制的参数
cfg = CN()

# 创建一个子config node来控制Conv2d所需要的参数
cfg.CONV = CN()
cfg.CONV.in_channels = 16
cfg.CONV.out_channels = 16
cfg.CONV.kernel_size = 3
cfg.CONV.stride = 1
cfg.CONV.padding = 1

# 创建一个子config node来控制MultiheadAttention参数
cfg.ATTN = CN()
cfg.ATTN.num_heads = 8
cfg.ATTN.embed_dim = 256

# 创建一个参数来控制我们需要创建的模型
cfg.MODEL = "conv"

# 构建一个build_model()函数来根据config返回我们需要的模型
def build_model(cfg):
    if cfg.MODEL == "conv":
        model = nn.Conv2d(
            in_channels=cfg.CONV.in_channels,
            out_channels=cfg.CONV.out_channels,
            kernel_size=cfg.CONV.kernel_size,
            stride=cfg.CONV.stride,
            padding=cfg.CONV.padding,
        )
    elif cfg.MODEL == "attn":
        model = nn.MultiheadAttention(
            embed_dim=cfg.ATTN.embed_dim,
            num_heads=cfg.ATTN.num_heads
        )
    else:
        raise NotImplementedError("only implement conv and attn")
    return model

model = build_model(cfg)
print(model)
>>> Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
```
并且如果不给到用户足够多的提示, 例如`cfg.MODEL`这个参数, 用户很可能无法知道我们的模型库里具体有多少个模型, 并且应该如何调用, 给用户带来了更多的困扰. 用`LazyConfig`可以很直接地避免这些问题, 如果用户需要构建一个conv的config, 或者是attn的config, 可以按照以下的操作:

```python
import torch.nn as nn
from detectron2.config import LazyCall, instantiate

# 通过LazyCall创建一个conv config对象
conv_config = LazyCall(nn.Conv2d)(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)

# 通过LazyCall构建一个attn config对象
attn_config = LazyCall(nn.MultiheadAttention)(embed_dim=256, num_heads=8)

# 根据我们的需要实例化对应的模型
model = instantiate(attn_config)
```

并且`LazyCall`在语法上更加贴合原生的python语法的使用, 即 `import需要的对象` -> `创建对应的config` -> `在需要的时候实例化`, 整体上更像是一个即插即用的config插件, 用户可以对这整个过程中的每一步做更加灵活精细且**直观**地控制. 但是LazyConfig本身在某种程度上过于flexible, 缺少了yacs中`frozen()`与`defrost()`的保护机制, 所以在使用的过程中也需要注意, 尽可能避免在一些小细节的地方做了不必要的修改.

### Summary
这篇文章只能算是一个引子, 简单对比了两个配置系统的基本使用与理解, 其实其背后的设计思想都各有好处, 具体还是需要在使用的过程中去感受, 后续会搭配更多的文章详细介绍一些更高级的用法, 这边再对文章的内容作一个简单的对比与总结:

**Yacs与LazyConfig在功能与灵活性上的比较**

| Func | Yacs | LazyConfig |
|:---|:---:|:---:|
| Config的增删查改 | :heavy_check_mark: | :heavy_check_mark: |
| Config的可读性 | 一般 | 好 |
| Config的灵活性 | 一般 | 好 |
| 代码量 | 多 | 少且直观 |
| 对于config的保护 | `frozen()`, `copy()`, `defrost()` | 无 |

**配置系统需要具有的基本功能: 学习了解一些基本的使用可以很直观地帮助到大家高效地做实验**
- 配置参数的**增删查改**
- 如何**合并**多组配置参数
- 如何合并**不同格式**的config配置, **merge from `List`, `Dict`, `.yaml`** ,e.g.
- 如何**导入(load)** 与 **导出(dump)** 配置
- 如何**通过命令行修改**参数

### References
- [d2 lazyconfig system](https://detectron2.readthedocs.io/en/latest/tutorials/lazyconfigs.html)
- [d2 yacs config system](https://detectron2.readthedocs.io/en/latest/tutorials/configs.html)
