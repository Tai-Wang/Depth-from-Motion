# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.core.bbox import AssignResult, BaseAssigner, MaxIoUAssigner
from .atss_3dcenter_assigner import ATSS3DCenterAssigner

__all__ = [
    'BaseAssigner', 'MaxIoUAssigner', 'AssignResult', 'ATSS3DCenterAssigner'
]
