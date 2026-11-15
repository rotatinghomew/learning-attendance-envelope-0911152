# 课室使用分析

面向老年学习服务的内部资料分析工具。当前支持版本资料检查、课室使用分析的描述性汇总（overview）、占用边界分析（occupancy_bounds）以及分析历史。它只描述已有观察与可行安排的边界，不自动生成干预结论。`learning/domain.py` 包含业务校验及汇总逻辑，`learning/occupancy.py` 包含占用边界建模；资料源和应用是同一机器的两个本地 HTTP 进程，分别保留来源契约和分析消费契约。

occupancy_bounds 把“同一学员同一时刻只能在一个教室”、课程时间窗口、刷卡进/出区间与教室容量放在一起，给出到课人头、同时占用峰值和座位分钟的可行上下界；没有刷卡只代表证据不足，不代表缺席。它支持用 `parameters.scenario` 描述调房/调容量后的情景并与基线比较；资料本身无解时会定位相互矛盾的记录。各时段极值来自独立优化问题，不能拼成一张不存在的完整到课表。

## 安装和验证

使用 Python 3.11.2，数值依赖固定在 requirements.txt；部署时先安装，不在服务启动或测试过程中下载。

```bash
python3 -m pip install -r requirements.txt
python3 -m compileall -q learning scripts tests
python3 -m unittest discover -s tests -v
python3 scripts/demo.py
```

最后一条命令启动两个真实服务，等待就绪后读取目录、执行当前汇总、保存并回读结果（含一次 occupancy_bounds 基线与调房情景分析），然后退出。它使用临时数据库，不会覆盖已有分析；服务就绪即继续，最长就绪期限为 65 秒，占用边界分析含线性规划求解，整体通常需要二十秒左右。使用容器时先构建再运行，构建不计入业务演示时间：

```bash
docker build -t attendance-envelope:local .
docker run --rm attendance-envelope:local
```

## 持续运行

```bash
python3 -m learning.provider
python3 -m learning.service
```

以上命令在不同终端执行，应用默认把结果写入 var/analyses.sqlite3。数据字段、缺失含义和版本差异见 docs/data-contract.md；应用接口与幂等约定见 docs/api.md。扩展业务分析时应保持原有 overview 和已保存分析可查询。
