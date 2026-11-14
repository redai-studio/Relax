# 独立权重发布与消费

训练端在优化器更新完成后导出逻辑权重，发布完整版本或无损差量。消费端独立获取指定版本，在本地重建标准 safetensors checkpoint，再由现有 SGLang Loader 加载。

## 启用方式

通过 `--weight-version-config` 指定发布配置，通过 `--weight-consumer-config` 指定消费者配置。省略这两个参数时沿用现有同步路径。

SharedFS 消费者使用 `transport.type=sharedfs`，配置 `transport.store` 和 `transport.disk_budget`。发布端和消费端使用可共同访问的仓库路径，各自配置临时目录与空间预算。

## 发布与安装

源端复用现有完整转换器导出逻辑张量，再逐块选择 COPY、SPARSE 或 RAW 表示。发布返回句柄；后台写入完成并确认源快照后，返回对应的权重版本。

消费者在暂停生成前完成下载与 checkpoint 重建。安装时按实际 worker 的 TP/PP/EP 布局加载并核对 Dense 或 MoE 参数；加载、缓存刷新和生成探针完成后恢复请求，并持久化 ACTIVE 回执及消费游标。

逐版订阅保留尚未消费的历史；消费者可从持久游标继续追赶。保留的规范权重内容可生成同一目标版本的 FULL 包，用于冷启动及安装恢复。容量不足或安装异常时返回目标版本、当前阶段及重试操作。

## 验证

```bash
python -m pytest -q tests/cpu/distributed/weight_sync
python -m pytest -q tests/cpu/backends/sglang/test_weight_version_verification.py
```

[FULL checkpoint 指南](./weight-checkpoints.md) 提供训练导出与独立 Loader 的 GPU 运行入口。
