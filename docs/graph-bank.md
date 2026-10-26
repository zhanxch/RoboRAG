# TRAIN 图库与检索

唯一默认路径为 `data/graph-bank/current`。本机已从 `work/v3-basic-suite/work/basic-suite/bank` 复制原始 TRAIN 图库，未重标注或重建索引。精确 JSON 文件哈希保存在 `results/selected-evidence.json`；原始 HDF5 仍通过图中的相对 `source.path` 读取。

该图库包括其他已接入任务；检索先按任务筛选。fold_clothes 为 3 条示范、24 个阶段、189 个窗口；push_T 为 3 条示范、25 个阶段、266 个窗口。两者均没有额外的 interaction 标注。默认库的“当前”表示选定数据快照，不意味着运行期间持续写入。

## 数据层次

`graphs/*.json` 保存每条历史回合的阶段链和 cue/action/feedback。cue 是执行前情境，action 是历史操作，feedback 是历史结果。`bank.json` 是图库目录，`search-index.json` 服务浏览检索，`execution-index.json` 服务模型检索，包含 graph/phase/chunk 三层记录。

plan 模式返回整条经验的初始情境、阶段标题和历史结果；action 模式返回更局部的阶段和窗口信息。排序使用 BM25、显式条件与多样性；中文标签和词汇别名对结果有影响。底层可选的 state/stage 条件尚未在当前 MCP 请求中启用。

`read_action_reference` 展开已检索到的引用。默认不返回完整动作数组；需要时显式设置 `include_actions=true`。start/end/close/open 选择阶段开始、结束或夹爪事件附近窗口；push_T 额外返回阶段首、中、末及相关腕部视图。历史坐标必须适应当前图像，不自动执行示范。

## 查看、构建与恢复

```bash
python -m roborag browse-bank
python -m roborag fetch --help
python -m roborag build-bank --help
python -m roborag index-bank --help
```

浏览器默认监听本机 8772 端口。fetch 从固定版本下载并校验训练示范；build-bank 会调用模型做标注，`--prepare-only` 只准备输入；index-bank 不调用模型，但会重写索引及必要元数据。

新标注或重建实验应使用新的 `--output` / `--bank` 目录，并声明新的数据版本。重建得到的语义内容不保证等同于冻结图库，不能覆盖 current 后沿用旧成绩。新机器要精确复现，需要恢复原始图 JSON、索引和 HDF5，并核对所选证据清单；模型重新标注不是精确恢复。

`bank/interactions.py` 保留同一检索器对结构化阶段/角色注释的支持，但默认验证路径不启用。它是可测试的实验能力，不是已有性能改善结论。
