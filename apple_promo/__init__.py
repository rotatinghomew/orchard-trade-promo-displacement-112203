"""苹果节联合券的净销量增量、库存与投放批准分析。

各模块只使用标准库，口径全部来自资料包本身：

- :mod:`apple_promo.packet`   资料读取与口径（净销量、退款更正、评估时点）
- :mod:`apple_promo.panel`    窗口面板、同轮对照与轮换拉丁方识别
- :mod:`apple_promo.censor`   售罄截断的缺单补回
- :mod:`apple_promo.evidence` 分项可接回总量的证据汇总
- :mod:`apple_promo.inventory` 按真实组件扣减、会员保留量与实收入库
- :mod:`apple_promo.scenario` 同时投放的新增与挤出范围、试算与缩量
- :mod:`apple_promo.ledger`   批准时点快照与恢复
- :mod:`apple_promo.cli`      经办人命令行入口
"""

__version__ = "1.0.0"
