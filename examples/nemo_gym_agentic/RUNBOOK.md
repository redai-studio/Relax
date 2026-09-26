# NeMo Gym 运行入口

旧的通用 runbook 同时包含多个已过时拓扑，容易把 Gym Ray、Relax Ray 和不同 recipe 的参数混用。
运行流程现已按 recipe 拆分：

- [Calendar：数据、本机 Gym、远程训练三步接入](recipes/calendar/README.md)
- [GSM8K：无工具协议 smoke](recipes/gsm8k/README.md)
- [Workplace Assistant：有状态工具环境](recipes/workplace-assistant/README.md)
- [R2E-Gym：OpenHands + Apptainer](recipes/r2e-gym/README.md)

总体架构、镜像关系、session 关联和当前验证边界见[中文总览](README.md)。
