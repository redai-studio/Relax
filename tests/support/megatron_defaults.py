# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""测试侧镜像的 Megatron 参数默认值。

这些字段由 Megatron 的解析器声明，不属于 Relax 参数，但 Relax 业务代码会直接读取。
CPU 单测不安装 Megatron，无法从上游取默认值，因此这里镜像一份，**仅用于构造测试参数
对象**；镜像值与真实上游是否一致由
``tests/backends/megatron/test_args_compat.py::test_megatron_test_defaults_match_upstream``
校验（装 Megatron 时才跑）。

新增字段时必须同时确认上游默认值来源，否则镜像会变成第二份"凭印象写的默认值"。
"""

from typing import Any


MEGATRON_TEST_DEFAULTS: dict[str, Any] = {
    "context_parallel_size": 1,
    "dynamic_context_parallel": False,
    "seq_length": None,
    # profiler 相关字段由 Megatron 解析器提供（见 arguments.py 中
    # "--record-memory-history ... defined in Megatron" 的注释）。
    # ProfilingConfig.profile_ranks 的默认值是 []（ArgumentGroupFactory 用 factory
    # 结果作 argparse 默认值）；需要覆盖旧 None 行为的用例请显式传 profile_ranks=None。
    "use_pytorch_profiler": False,
    "record_memory_history": False,
    "profile_ranks": [],
    "profile_step_start": 10,
    "profile_step_end": 12,
}
