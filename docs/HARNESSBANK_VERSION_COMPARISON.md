# HarnessBank 三版本过程对照记录

本文可公开阅读；下文 `data/audits/` 链接指向保留在本地开发工作区的原始证据，不随 GitHub 仓库发布。原始数据缺席公开仓库不改变本记录的单例结论边界。

## 2026-09-07 正式化状态与历史边界

正式默认现为 v20 `rubric-union`，显式对照为 `full-history`。旧的只按锚点裁剪的聚焦实现、相同锚点加标签/索引的路由实现及其共享启动器已移除；重建说明和冻结清单保留。下文准备、恢复和运行阶段的记录按原时间顺序保留，其中“待执行”“长期保留多套机制”等是当时状态，不是当前授权或可选功能。清理范围及恢复边界见 [正式化记录](V20_FORMALIZATION.md)。

## 四组 Reflection 单独对照：入口与结论边界

在三版完整运行之后，另行固定最后一次 Master 决策前的状态、核验问题、Rubric、锚点、模型和现代 v20 职责提示，只比较四种 Reflection 输入。它不是四套历史 Prompt/控制器的完整重跑。四组均一次成功，无重试；之后没有执行 Master、Worker 或 Synthesis。

| 输入组 | Findings | Prompt Tokens | 总 Tokens |
| --- | ---: | ---: | ---: |
| 全历史 | 33 | 21,888 | 22,777 |
| 旧聚焦 | 2 | 2,075 | 2,649 |
| 旧路由（相同两条加标签/索引） | 2 | 2,222 | 2,671 |
| v20 Rubric 并集加锚点 | 18 | 11,141 | 12,191 |

完整入口：[复盘](../data/audits/harnessbank-reflection-ablation-20260907-01/review.md)、[原始清单](../data/audits/harnessbank-reflection-ablation-20260907-01/manifest.json)、[共同源状态](../data/audits/harnessbank-reflection-ablation-20260907-01/source_state.json)。各组实际发送输入与输出：

- 全历史：[prompt](../data/audits/harnessbank-reflection-ablation-20260907-01/full-history/prompt.json)、[result](../data/audits/harnessbank-reflection-ablation-20260907-01/full-history/result.json)
- 聚焦：[prompt](../data/audits/harnessbank-reflection-ablation-20260907-01/rubric-focused/prompt.json)、[result](../data/audits/harnessbank-reflection-ablation-20260907-01/rubric-focused/result.json)
- 路由：[prompt](../data/audits/harnessbank-reflection-ablation-20260907-01/rubric-routing/prompt.json)、[result](../data/audits/harnessbank-reflection-ablation-20260907-01/rubric-routing/result.json)
- v20：[prompt](../data/audits/harnessbank-reflection-ablation-20260907-01/current-v20/prompt.json)、[result](../data/audits/harnessbank-reflection-ablation-20260907-01/current-v20/result.json)

本例 v20 比全历史少 **46.48% 总 Tokens**，保留主要判断，并更完整地利用了 K=1 候选选择与 K=3 最终评估的限定。该限定本来就在 Worker finding `r1-t3-f4` 中，不是 Reflection 新发现的论文事实。全历史同样表现良好、没有明显偏题；小切片也能保留核心限定，但无法利用未提供的其他证据。本例支持的是上下文效率取舍，不支持稳定显著的能力提升、全面领先或整个 Agent 的性能增益。所选案例及锚点有目的性，四组各一次，未测 Master 是否会自然选择这些 Rubric，也未测对最终综合的影响。

原四组记录中的 `result.report.context_mode` 等部分字段沿用了 dataclass 默认值；真实组别和输入范围应看 `group`、manifest、prompt 和 `calls[].user_prompt`，不能据辅助元数据重新分类。历史文件未修写。

[对照脚本说明](../variants/reflection-ablation-20260907/README.md)：原脚本现已调整为只使用正式运行时的 `rubric-union` 与 `full-history` 两种原生输入，并要求新输出目录。新清单版本为 `reflection-context-comparison-v2`，旧四组清单仅用于历史审计；其中原脚本、测试、运行时及提示哈希不改，也不宣称当前脚本可逐字节复现旧实验。当前两组直接使用正式 payload，包括各模式自身的上下文与职责约束；不能把它们误称为旧四组同职责提示实验的原样重跑。

## 最终汇总：三组有效运行均已完成

三组均正常 DECIDE、运行后冻结校验通过，且都只执行首批取证后的 post_method_model Reflection。没有真实第二次 Reflection；当前版也没有选 REFLECT 或 READ_PAPER_AND_REFLECT。聚焦版 01 的基础设施失败另行保留，不混入下表质量样本。

| 有效运行 | READ 批次 / Worker 任务 | Master / 全部逻辑调用 | Tokens | Reflection | 来源关联 |
| --- | --- | --- | ---: | ---: | --- |
| 聚焦重建版 02 | 4 / 7 | 5 / 21 | 206,225 | 1 | 一条未关联 disposition |
| 路由重建版 01 | 2 / 6 | 3 / 17 | 174,581 | 1 | 完整 |
| 当前 v20 01 | 3 / 6 | 4 / 18 | 191,163 | 1 | 一条未关联 disposition |

总体判断：三组均抓住完整系统收益与 Gene Bank 独立因果证据之间的区别，均没有把未报告细节直接当作无效反证。本次路由版最紧凑，相对聚焦版少 15.34% Tokens；v20 比聚焦版少 7.30%，但比路由版多 9.50%。这些是整体版本各一次的观察，不是 Rubric 单因素的因果估计。

聚焦版更细查 judge 可靠性与 confirmation/Table 3 关系，但角色配置再拆 judge 一轮、消融关系后来另查，存在局部重复。路由版两批覆盖主干并新增完整搜索重复性，较早收束；不过 confirm 变体含义及 proposer/evolver 的部分措辞仍不充分准确，结构来源完整不代表语义判断完全正确。v20 对正向与负向跨模型迁移的共存、显示值舍入边界更清楚，但没有同样深入核验角色配置与 confirmation，且第三轮再次返回已经检查过的 SWE-bench 舍入问题。没有证据认定 v20 整体胜出。

v20 后续 Master payload 明示 reflection_budget.remaining=1、request_available=true。模型认为主要机制竞争解释已由首次反思处理，剩余问题需要查原文或新增论文实验，因此没有再选反思。本次只能确认第二次反思可选却未被使用，不能证明其隔离输入无效，更不能验证三种第二次输入设计的优劣。三组首次反思均有价值：将 semantic bank 与普通 screened elite archive 的竞争解释带入最终归因边界，但没有执行全面跨 Worker 关系复核。

墙钟分别 406.253、511.487、651.616 秒；请求尝试分别 22、19、20 次，均含服务连接重试。并行等待及慢响应明显影响耗时，不能据此直接排列架构速度。聚焦版另有失败 01：返回 usage 2,548 Tokens，102.932 秒，不含失败请求未知计费。

逐轮任务、读页、信息流、重复成本和来源问题详见：[聚焦版复盘](../data/audits/harnessbank-20260906-focused-reconstructed-02/review.md)、[路由版复盘](../data/audits/harnessbank-20260906-routing-reconstructed-01/review.md)、[v20 复盘](../data/audits/harnessbank-20260906-current-v20-01/review.md)。以下内容保留为准备和运行过程的时间顺序记录。

## 历史进展：独立恢复与逐组启动

**后续运行更新：** 路由重建版 01 已正常 DECIDE，17 次调用、174,581 Tokens、511.487 秒，仅一次 Reflection；来源关联完整，运行后冻结校验通过。其两次 Evidence 请求在连接错误后自动重试成功，耗时不能全部归因于机制。最后一组 current-v20 已随即启动，输出目录为 data/audits/harnessbank-20260906-current-v20-01。以下预检段落保留为历史记录。

**运行更新：** 聚焦重建版 01 因 Terra 首轮两次连接错误结束，没有 Worker、Reflection 或有效 FinalJudgment。Luna Synthesis 返回 2,548 Tokens 但因空发现输出校验失败；总耗时 102.932 秒，运行后源码/PDF 核验一致。详见 ../data/audits/harnessbank-20260906-focused-reconstructed-01/review.md。用户明确授权额外重试，聚焦版 02 已以相同冻结配置完成：正常 DECIDE，4 个 READ 批次、7 个 Worker、5 次 Master、1 次 Reflection，共 21 个逻辑调用、22 次请求尝试，已返回 206,225 Tokens，406.253 秒。最终 11 条发现，来源状态 incomplete，仅一条 unlinked_disposition:r1-t3-f4；运行后冻结核验通过。两个结果目录独立保留。不要把 01 当作机制质量样本或把已返回 Tokens 当作完整供应商计费。

最新进展：聚焦重建版已通过 239 项完整离线测试，23 次历史调用的通用静态字段与 system prompt 对照通过，独立审查允许启动；源码／提示资产／说明／测试共 25 个文件已冻结。一次启动审批被中断且未创建结果目录，确认未执行后重新发起授权，第一组真实运行已开始，输出目录为 data/audits/harnessbank-20260906-focused-reconstructed-01。此前“尚未调用”的段落均是预检阶段历史记录，不代表当前状态。原版没有运行。

用户随后明确授权依据记录独立恢复两套历史机制，并要求不删除代码、长期保留多套机制。以下“未找到备份”的预检记录保留为过程记录，不再是等待授权的阻塞项。

已建立 variants/harnessbank-20260906/，其中 current-v20 是当前源码和测试的独立副本；Python 源码逐文件复制后哈希一致，从副本自身目录运行全部 290 项测试通过。rubric-focused-reconstructed 与 rubric-routing-reconstructed 正在恢复和做离线核对。当前仍无新付费模型调用。

采用独立目录而不是覆盖主 src；新增运行入口将记录逐调用事件，并在运行前后核对冻结文件。恢复版显式保留 reconstructed 标识，不宣称源码与旧版逐字节相同。原版仍取消，不会顺带运行。

## 恢复过程中的离线质量关卡

首轮历史恢复副本分别通过 230、231 项复制回归与新增测试，但人工代码复核发现：聚焦版增量 Master 仍保留只挑最后一个 Worker 建议的 next(...)；两个恢复版也尚未实现历史版已有的最终 source_finding_ids / finding_dispositions。首轮报告对建议传递的描述不准确，已撤回其作为付费运行就绪证据的效力。这说明复制的旧测试通过不能证明中间版本恢复完整。

目前在补齐上述核心行为，并改用历史真实 payload 的静态提示／输出契约逐项匹配测试，不仅检查版本名或笼统的语义相似。两个目录及中间恢复代码均保留，不删除主项目代码，不启动有已知身份偏差的付费对照。

启动器审查还要求冻结恢复说明和测试文件，并增加实际 run 路径的离线覆盖。额外宽泛安全扫描可能因普通路径擦除整段事件提示，因此事件输出必须沿用实验的原有脱敏策略，不把普通路径自动当作密钥；角色输入不受记录回调影响。

## 当前状态与授权范围

2026-09-06：用户同意按顺序测试 Rubric 聚焦版、Rubric 路由版、当前版，并要求详细记录每一版整个过程的效果与问题。原先计划中的原版运行已被用户明确取消，以避免额外模型成本。原版只作历史参考，不纳入本次同条件新实验。

截至本次预检结束，尚未启动任何真实模型调用，没有生成 HarnessBank 新实验结果。只读进程检查未列出正在运行的 Python 或 uv 进程。未进行联网搜索、Phase 6、论文代码执行或 Git 写操作；没有读取或展示 .env 内容。

## 版本恢复预检：尚未满足正式对照条件

现有 Git 历史只有初始公开版本和正式外部审计版本，分支与远端引用均指向 2aed007，stash 为空。检查 data/baselines、data/profiles、data/eval、data/audits 中的源码、补丁和归档候选后，目前仅找到原版隔离源码与 source.zip，没有找到两个中间版本的完整源码快照。此结论限于已经检查的位置，不代表其他备份位置一定不存在。

两个中间版本的历史 manifest 保存了源码 SHA256，result 保存了真实调用记录；这些资料能支持核对或重建，但哈希本身不能恢复源码，调用记录也不能完整还原未执行的控制分支。不能把当前代码切换 reflection_context_mode 后冒充历史架构，也不能把按记录重建的版本标为未经改动的旧版。

| 计划组 | 已有记录 | 当前可运行性 |
| --- | --- | --- |
| Rubric 聚焦版 | onedayagent-20260906-rubric-focused-01；Master v16、Reflection v7 | 未找到完整源码快照，等待恢复来源或重建授权 |
| Rubric 路由版 | onedayagent-20260906-rubric-routing-01；Master v17、Reflection v8 | 未找到完整源码快照，等待恢复来源或重建授权 |
| 当前版 | 工作区；Master v20-restored-convergence、Reflection v10 | 源码存在；尚未冻结本次实验快照，也未启动实验 |
| 原版 | 2aed007 隔离源码存在 | 用户取消，不运行 |

关键核验哈希：

| 历史版本 | paper_agent_runtime.py SHA256 | paper_agent.py SHA256 |
| --- | --- | --- |
| 聚焦版 | 89adbefe7f7326b0f549bddc9b704cfa9ac84e0aee619de5f01177a9e8112671 | dccdd8e010deaa9d23886cb7792112f92e53986489325cdd781e8b0cfcad15e0 |
| 路由版 | 3b9253d885f281308a901e98e88a8d90ae19668de07e1f50e10d2d846a1b59b1 | 845c08f30e39b82d0cb522a57900ec0ec80bff90ade2097b906c3f18d4b09cfc |

此前提出四版本可直接对照时，没有先确认中间版本源码是否保存，这是准备阶段的疏漏。当前选择先暂停付费调用并说明缺口，而不是消耗预算运行身份不明确的版本。

## 待执行的统一实验口径

论文使用 paper/self-evolution/harnessbank.pdf。三组统一 PDF、论文自身主张导向的调查目标、角色模型、temperature、Worker 并行度、最大轮数和 Reflection 上限。计划沿用 Terra Master/Reflection、Luna Locator/Evidence/Synthesis、temperature=1、Worker 并行度二、最多五轮和两次 Reflection。运行前必须核对实际调用参数并记录 PDF 与源码哈希。

各版本自身的提示、上下文组织、触发条件和动作接口保留。若为统一目标或输出安全作适配，必须逐项记录，不能称为逐字节历史重跑。任何重建组都必须独立标明 reconstructed，说明与历史输入可核对的部分、无法确认的部分及离线验证结果。

轮数口径需要特别区分：旧控制器的 pre_decide Reflection 会在同一读取轮次内重新询问 Master，当前独立 REFLECT 批次则占用一轮。三组 max_rounds=5 是相同配置数值，不是严格相同的 Master 调用或推理预算。必须另外报告实际 READ 批次、Master 调用、Reflection 与组合动作，不能把这个整体版本对照称为等计算量消融。

按聚焦版、路由版、当前版顺序各运行一次，不并行启动三个完整审计，不因结果不好选择性重跑。若发生服务失败，记录逻辑调用、请求尝试、已返回 usage 及结果保留情况；额外重跑或 Synthesis 回放单独标注，不能混入原始成功率和成本。

不向模型注入历史结果、人工已知问题或本记录，不强制第二次 Reflection，也不运行外部证据审计。第一轮比较是探索性整体版本对照，不是单因素消融，也不据单个样本宣布稳定提升。

## 每版必须形成的过程记录

每版记录实际启动条件、冻结来源、离线预检、运行状态和结果目录。按每个 Master 决策轮次，解释它当时掌握了什么，为什么选择该动作，各 Worker 问题、所读页面、关键发现、建议以及成本；说明下一轮是否收到并利用这些建议，而不是只列任务数量。

对 Reflection，记录触发时点、输入范围、聚焦 Rubric 与锚点、实际收到的相关及反向证据、推理提出的修正或竞争解释。继续追踪 Master 是否采纳、是否转成准确的缺失证据任务、是否重复取证，以及最终报告是否保留。未发生第二次 Reflection 时明确记录原因与未测到的功能；次数本身不作为质量分数。

对关键发现，追踪“原文或 Worker 事实—进入 Master 的信息—跨报告核验—后续行动—最终结论”的完整链路。来源 ID 与 disposition 只用于结构追踪，不替代语义核验。特别留意模型角色与比较配置的关联，但将其作为人工评估线索，不注入 Agent，也不预设某个版本必须犯错或某个角色必须纠错。

对重复消耗，结合问题语义、已读证据、页面重叠与实际新增信息判定。相同页码不自动等于浪费；必须解释重复是否带来新证据或重要纠正。分别报告明确重复调用的直接 Tokens、疑似低收益支出和无法量化的累计上下文成本，不能把整轮 Master 成本全部当作可节省。

最后记录正常 DECIDE、预算耗尽、NEEDS_HUMAN、Synthesis 结果、结构与来源警告、实际错误及未解决问题。区分“运行保存完成”和“控制流程正常完成”，区分“报告发现更多”和“判断质量更好”。

## 横向比较与结论边界

三版结果全部完成后，比较关键证据覆盖、信息传递与利用、有效跨 Worker 核验、反思到行动的转换、发现保留、重复取证、收尾以及按角色分解的成本。报告具体改善和退步及其对应 trace 依据，不只比较最终结论和总 Tokens。

若没有第二次 Reflection，本次不能验证第二次聚焦输入的效果。若不同版本一次运行表现不同，也不能排除模型随机性与整套提示变化的影响。后续局部固定状态对照或追加重复实验需单独决定，不在当前三次运行中自动扩展。

## 实验结果

聚焦版：01 基础设施失败保留；用户授权的 02 已正常完成，详细过程复核已保存在其 review.md。结果目录分别为 data/audits/harnessbank-20260906-focused-reconstructed-01 与 -02。

路由版：修正新增测试中 PaperPage 构造参数遗漏后，241 项离线测试全部通过；独立审查通过，25 个文件的快照已冻结并校验。真实运行 01 已正常完成，17 次调用、174,581 Tokens，来源关联完整。结果目录为 data/audits/harnessbank-20260906-routing-reconstructed-01。此次修正只涉及测试数据，不改变运行机制。

当前版：路由版完成后立即启动 01，现已正常 DECIDE，18 次调用、191,163 Tokens、651.616 秒，运行后冻结校验通过。一条来源警告 unlinked_disposition:r1-t2-f6。结果目录为 data/audits/harnessbank-20260906-current-v20-01，详细 review.md 已完成。

横向结论：三组均完成，见文首最终汇总；原版取消，未追加实验。

## 人工原文核验基线（不注入模型）

本次本地 HarnessBank 为 9 页，首页标注 arXiv:2607.13683v2、2026-07-30。PDF SHA256 为 360efedc16753f6bfe062cd58b1e2f7e9984bbdd36a8cc79ed363af686aeaeb2。旧 HarnessBank 实验 manifest 未直接列出 PDF SHA256；因此在进一步验证前，不宣称新旧 PDF 字节相同。本次三组必须使用这个相同冻结 PDF。

第 5 页实验设置写默认任务骨干为 Qwen3.6-27B，Claude Opus 4.8 为 evolver；同页基线段落写 GEPA/DGM 的 task agent 和 proposer 为 Qwen3.6-27B。两段关系值得核对，但不能预设必须通过第二次 Reflection 才能识别：它们在同一页，一个 Worker 就可能正确联系起来。重要的是结论与证据边界，而不是强行把功劳归给某个角色。

第 4 页提供 Gene Bank 的父代 argmax、重组、逐格竞争、停止及训练/测试职责；第 5 页提供有效性、激活、显著性筛选。第 7 页有门控消融、false elites 与停止轮数观察，也有来自不同 cell 的机制组合陈述。因此“缺少 bank-disabled / 等规模无结构档案对照”不能扩大成“没有任何档案相关观察”；同样，激活 beacon 证明执行发生，不直接证明因果贡献。

第 6—7 页跨模型实验既有不匹配补丁收益减弱，也有相似病理下的迁移收益。报告应同时保留这两种观察，避免将“不是普遍最优 harness”误写成“所有跨模型迁移均无效”。SWE-bench 明确标为小样本 preliminary，不能将“七个领域全部显著”作为事实。上述是人工核验线索，不是必须逐项出现的固定模型答案表。
