#!/usr/bin/enviroments python
# -*- coding:utf-8 _*-
"""
@File:   _common_.py
@IDE:    Pycharm
@Des:
"""
from typing import Dict, TypeVar

import pytorch_lightning
import torch

from src.config import BaseConfig, BaseModelConfig
from src.utils.factory import Factory

__all__ = [
    'BaseModel',
    'ModelFactory'
]


class BaseModel(pytorch_lightning.LightningModule):
    """
    后续的基类
    """
    name: str = "Model"
    conf: BaseModelConfig

    def __init__(self, conf: BaseModelConfig):
        super().__init__()
        self.name = conf.name
        self.conf = conf


BaseModelClass = TypeVar('BaseModelClass', bound=BaseModel)


class ModelFactory(Factory):
    def __init__(self):
        super().__init__(module_name='模型')

    def restore_model(self, config: BaseConfig, *args, **kwargs) -> BaseModelClass:
        """
        用于测试过程直接加载相应的模型
        :param config:
        :return:
        """
        obj = torch.load(config.data.model_load_path)
        if isinstance(obj, Dict):  # 说明是state_dict
            model_cls = self._reflect_cls(config.model.name)
            model = model_cls(config=config, *args, **kwargs)
            obj = model.load_state_dict(obj)
        return obj

    def product(self, config: BaseConfig, is_resumed=False, *args, **kwargs) -> BaseModelClass:
        """
        获得实例化好的模型和状态
        :param config: 配置
        :param is_resumed: 是否是断点继续
        :return:
        """
        # model = model_cls(config.model)
        # if is_resumed:
        #     status = self._load_status(status_path=config.data.status_load_target_path)
        #     model.load_state_dict(status.model_state_dict)
        # else:
        #     status = self._load_status(config=config)
        model = super().product(config.model.name, *args, **dict(kwargs, config=config.model))
        return model
