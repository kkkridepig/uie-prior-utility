# V3 现场差异审计

核验UTC 2026-10-04T17:08:34Z。当前commit cb9b104e724f304053d2e658f54deec8ba115e2f 与框架记录一致；起始工作区干净，新建研究分支 experiment/prior-utility-cde-v3-20261004。未检测到研究训练进程；厂商 torch 2.0.0a0+nv2303 保持原路径。

父checkpoint哈希与85000步身份核验通过。旧预算 14.783688 h，仅作历史；新上限72 h，预留12 h。剩余磁盘 4.95 GiB，需要delta checkpoint与容量预检。

旧BASE_CONT日程终点10000，新轮5000，必须重训同轮控制。旧keyed_noise忽略外部seed；v3单独键控。旧C同位置sigmoid不满足QK交叉注意力，v3新实现；旧D浅层phase是允许简化，不列实现错误。

角色图数：{'adapter_fit': 490, 'route_fit': 141, 'route_cal': 71, 'source_dev': 91, 'legacy_exposed_regression': 97}。61/356绑定已知场景；保留旧测试身份为legacy_exposed_regression，不称新盲测。DA/VGG上游暴露未知；新留出候选待审计。

时钟差异：会话环境日期2026-10-05，服务器date -u当前为2026-10-04 UTC。协议ID沿用户指定20261004；事件时间记录服务器实测UTC，不人为改写成会话日期。设备账本使用实际累计占用秒数。
