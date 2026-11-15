# 分析服务接口

资料服务在 8765 端口提供 `/releases`、`/releases/term-a` 和版本内的表分页。表页携带 revision；客户端必须读取同一清单版本并验证摘要，不能合并不同版本的页。

应用服务在 8766 端口提供 `GET /catalog`、`POST /analyses`、`GET /analyses`、`GET /analyses/{id}`。当前目录有 overview 课程资料汇总与 occupancy_bounds 占用边界分析。分析实现登记在 `learning.domain.ANALYSES`，函数接收完整 tables 和 parameters，返回可序列化对象。

occupancy_bounds 的 parameters 可省略，或携带 `scenario` 数组描述场地调整情景；每项形如 `{"session": "s03", "room": "r9", "capacity": 30}`，只允许覆盖会话的 room 或 capacity，不允许改动 bookings/scans 证据。结果包含 baseline、可选 scenario 与 comparison；当资料或情景无解时对应配置的 `feasible` 为 false，并在 `contradictions` 中定位相互矛盾的记录（刷卡窗口冲突、同学员区间不兼容、强制在场超容）。人头与同时占用峰值由刷卡强制在场和逐分钟可行性组合判定，座位分钟由分钟级流守恒线性规划求解，边界依据保存在 result 内（含每条刷卡的 forced_minutes）。各极值分别求解，不能拼成一张完整到课表。

创建请求字段为 request_key、release、analysis 和可选的 parameters 对象。request_key 是调用方的幂等编号：相同完整请求返回同一分析，相同编号不同内容返回 409。分析返回 id、request、revision 和 result。不同 request_key 是不同留存记录。缺少字段、未知分析类型或无效资料返回 422；查询不存在的记录返回 404。JSON 不允许 NaN 或 Infinity，无法估计的量使用 null 并在结果中说明原因。

保存后的分析内容不因资料更新而重算。新分析可在 result 内保存统计对象、参数、诊断及所需的版本化结构；不可覆盖其他分析或跳过资料完整性校验。数据库只保存本地分析，不是资料服务的业务真源。
