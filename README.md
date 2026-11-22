# 果品直销缺口的受约束配方需求分析

`资料/演练.json` 保存本地合成的业务记录，不含个人信息及实际企业数据。各资料集的字段、单位和有效范围见[资料格式](资料格式.md)。需要 Python 3.11.2，仅使用标准库，从项目根目录运行。

查看资料包的记录数量：

```sh
python3 inspect_packet.py
```

编译与测试：

```sh
python3 -m compileall -q inspect_packet.py tests
python3 -m unittest discover -s tests -v
```

当前入口只读取资料包并列出记录规模；测试检查输入记录间的引用和格式，不给出业务分析或决策结果。
