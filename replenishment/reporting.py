"""Chinese report text with explicit data coverage and study limitations."""
import pandas as pd


def markdown_table(frame, columns=None):
    if frame.empty:
        return "无适用记录。"
    if columns:
        frame = frame[columns]
    rows = [[str(x) for x in frame.columns]]
    def display(value, column):
        if pd.isna(value):
            return "n.a."
        if isinstance(value, float):
            return f"{value:.1%}" if any(s in str(column) for s in ("share", "rate", "wape", "bias_pct")) else f"{value:,.3f}"
        return str(value)
    rows += [[display(x, col) for col, x in zip(frame.columns, row)] for row in frame.itertuples(index=False, name=None)]
    return "\n".join(["| " + " | ".join(rows[0]) + " |", "| " + " | ".join(["---"] * len(rows[0])) + " |"] +
                     ["| " + " | ".join(row) + " |" for row in rows[1:]])


def summary_text(context, audit, classification_summary, panel, plan, evaluation, quality):
    weekly = context["analysis_grain"] == "weekly"
    selected = evaluation.get("selected_models", pd.DataFrame())
    review_count = int(selected.get("phasing_out_review", pd.Series(dtype=bool)).fillna(False).sum())
    periods = int(panel["date"].nunique())
    n_months = int(panel["date"].dt.to_period("M").nunique())
    latest_end = panel["period_end"].max().date()
    coverage_gap = (pd.Timestamp(context["analysis_end"]) - pd.Timestamp(latest_end)).days
    prediction = evaluation.get("forecast_16_weeks", pd.DataFrame())
    future_dates = (f"本次未来16周覆盖 {prediction['week_start'].min().date()} 至 {prediction['week_end'].max().date()}。"
                    if not prediction.empty else "没有生成周级预测。")
    status = "已完成真实周销量建模与最后16周测试；结果为初步样本外证据" if weekly and not evaluation.get("test_detail", pd.DataFrame()).empty else "仅完成月度清洗、描述性分析；未实施周级预测实验"
    return f"""# 毕业设计需求预测分析

运行状态：{status}。

## 分析期与筛选

- reference_date：{context['reference_date']}。
- analysis_start：{context['analysis_start']}；analysis_end：{context['analysis_end']}。
- 当前不完整月份 {context['excluded_current_month']} 已完全排除；跨边界周也整周排除，避免引入当前月销量。
- 最终只分析 Normal 和 Phasing Out 产品；Phasing Out 保留独立标识。
- 分析窗口严格为12个完整自然月；实际历史可用期并不等于已覆盖整个12个月。
- 真实数据粒度：{context['source_grain']}；分析粒度：{context['analysis_grain']}。
- 最终数据最早日期：{panel['date'].min().date()}；最晚周/月开始日：{panel['date'].max().date()}；最后期间结束日：{latest_end}。
- 最终可用期间数：{periods}；有数据的自然月数量：{n_months}。
- SKU唯一键：销售组织 + 库位 + 物料。空库位保留为 `<UNSPECIFIED>`，表示未提供，不能据此开展独立仓库决策。

{markdown_table(audit)}

表中 snapshot rows 指工作簿产品行，SKU-period rows 指展开后的销量期间行，不能混为同一种记录。

## 字段与数据质量

真实字段、两行/多行表头、字段类型、合并区域、状态候选及日期列判定见 data_dictionary.xlsx、workbook_inspection.json、field_candidates.json。
无年份的月名（例如 Oct）不根据列位置反推年份；不会与周销量相加，避免重复计量。
库存、在途、欠单和公司预测为当前快照/未来排程，未提供可验证的历史生效日期或预测发布时间，均不进入历史模型和历史库存回测。
产品状态作为当前产品主数据进行筛选；缺少历史状态生效日期，因此回测是“当前合格产品队列的回溯测试”，不是历史全产品组合回测。
缺少单位字段时不推断件/箱转换，也不计算跨物料可比的总金额。数量误差总体汇总仅作源数量单位的组合描述，按SKU评价优先。

{markdown_table(quality)}

重复快照键（含 Parent/Son）、负销量/可能退货、缺失销量、单位冲突的整个SKU被隔离，明细在 quarantined_records.csv。原始值保持不变，未静默累加、删除重复行或把负数改为零。
最终分类和建模只基于通过检查的合格SKU；这一排除会带来队列选择偏差，需要业务确认退货、层级口径后再扩大样本。
仅在已有连续来源覆盖区间内补齐某个SKU的无销量周；不会把最早来源之前未知的周当成零需求。未知整周缺失会停止建模。
没有上架日期/停产生效日期，默认合格SKU在已提供的周覆盖区间内可观察，需后续用生命周期字段验证。

## 需求类型

ADI = 观察期间数 / 正需求期间数。
CV² = Var(D+) / Mean(D+)²，其中 D+ 仅包含大于零的真实销量，方差 ddof={context['variance_ddof']}。
阈值 ADI={context['adi_threshold']}、CV²={context['cv2_threshold']}，均可配置；等于阈值归入高ADI/高CV²组。
Smooth 为低ADI低CV²；Erratic 为低ADI高CV²；Intermittent 为高ADI低CV²；Lumpy 为高ADI高CV²；全零序列为 Zero。
只有一个正需求点时 CV² 暂记0且 cv2_reliable=False；分类暂定，不能声称波动已可靠估计。
产品状态与需求类型为两种独立维度。分类在真实分析粒度上计算，月分类不能直接当作周分类。

{markdown_table(classification_summary)}

## 异常大单

Sen slope 为全部两两时间点斜率的中位数。比较前半段和后半段的原始销量中位数。
只用正销量计算 IQR 和 U = Q75 + 1.5 × IQR（倍数可配置）；正需求样本不足时不替换。
如果 Sen slope > 配置阈值且后半段Median高于前半段，超限高值仅标记，不替换。
其余统计确认超限点分别保留 Raw、截断到U的 Winsorized、替换为历史非零销量中位数的 Median。
这里的确认是可复现统计规则，尚不等于已获业务订单确认。完整窗口异常值比较只用于描述与最终估计；验证/测试中的阈值、趋势和中位数都只从各自训练窗口重算。
cleaned_last_12_complete_months.csv、测试指标和后续库存实际需求始终使用未被异常值替换的真实销量。

## 预测、验证与测试

{markdown_table(plan)}

候选为 MovingAverage、SES、Croston、SBA、TSB，分别比较 Raw/Winsorized/Median。
先保留最后16个真实周为独立测试集，再在此前历史上做滚动原点验证；默认验证步长1周、预测步长1周。不同历史窗口共用相同验证原点，保证可比。
选择依据为验证MAE，分数相同时优先26周；所有平滑参数固定在配置中，没有在测试集调参。
模型选择后仅对已选方案汇总正式测试结果。其他候选测试比较标为描述性结果，不用于再选模型。
验证/测试分组的需求类型仅用相应原点前的训练销量计算；全窗口最终分类不参与历史模型选择。
MAE = mean(abs(predicted-actual))；WAPE = sum(abs(error))/sum(abs(actual))；Bias = mean(predicted-actual)，正值表示高估。
真实需求总和为0时 WAPE/Bias百分比为 n.a.，不伪装为0。
13/26周候选若共同验证点少于最低要求，标为样本不足。52周只有在足够训练、验证和16周测试点都存在时才能比较，不借用窗口外历史。
仅有约12个月历史时，52周模型通常无法同时获得严格独立评估点；不会强行训练。
公司现有预测虽然有历史周表头，但缺少每个预测的发布时间/版本，且当前周预测/月预测可能使用分析期结束后的信息，故标为不可比较；不补造历史预测，不把月预测拆成周预测。

已选模型独立16周测试指标（类型仅由测试前训练窗口计算）：

{markdown_table(evaluation.get('selected_test_summary', pd.DataFrame()), ['product_status', 'demand_type', 'skus', 'mae', 'wape', 'bias']) if not evaluation.get('selected_test_summary', pd.DataFrame()).empty else '无可用周级测试。'}

## 未来16周与库存回测可行性

{'当前真实周数据足以生成运行日期之后下一个完整周开始的16周初步预测；这不是16周验证步长的模型选择，长期精度证据来自独立16周测试，验证仍为1周。26周窗口验证点少，结论需要持续采集更多周后复核。' if weekly else '只有月度来源，不生成或模拟周销量。无法可靠实施未来16周周级实验，需补充真实周销量。'}
正式未来预测从 reference_date 之后下一个周一开始，保留连续16个完整周。训练结束与未来起点之间的时滞仅用预测跨越，不引入当前月份真实销量、不填造实际销量；各SKU的 unobserved_bridge_weeks 和 lead_from_training_weeks 记录这一时滞。最后完整周结束距 analysis_end 尚有 {coverage_gap} 天；边界周因混有当前月/窗口外日期而被排除，这一数据时滞不应被解释为零销量。
{future_dates} 独立16周测试评估的是观测后第1至16周；本次未来起点含额外桥接时滞，更长提前期尚无直接验证证据。
预测输出允许未来日期，它们是估计值；任何模型输入、验证真实值和特征仍严格限制在分析窗口内。
Phasing Out 有 {review_count} 个SKU触发增长或长串零销量后仍持续正预测的复核标识。预测不自动转换为补货单，需独立制定减量/停止补货规则。
当前数据不足以实施可信的每周滚动库存回测：缺少分析期内每周库存起点、收货/在途历史及实际到货时间、缺货/未满足需求、历史欠单、订货记录和生效交期。
仅有销量无法区分零需求与缺货导致的零销售，也不能把模型处理销量替代真实消耗。

## 下一步补充

1. 核实公司周历是否为ISO 8601（当前显式配置为ISO，周一至周日）；如为财务周历，需提供周起止日期映射，不能直接复用本次日历结论。
2. 补充分析窗口前端缺少的真实周、当前最后完整月内边界周的逐日销量；当前未使用更早/当前月数据凑样本。
3. 解释Parent/Son汇总与明细口径，提供独立交易ID、物料层级和替代物料映射，确认哪些记录可合并。
4. 提供销售/退货分离数量、基本计量单位、包装换算规则和物料有效上市日期。
5. 提供逐期冻结的公司预测：SKU、target_week、issued_at、forecast_version、quantity。
6. 提供历史产品状态有效期、每周库存/收货/订单/欠单/缺货记录、真实交期、成本和服务目标后开展库存回测与价值分析。

原始Excel SHA-256：{context['source_sha256']}。已在运行结束时重新核对，未修改原始工作簿。
随机种子：{context['random_seed']}；本项目方法本身为确定性计算。运行配置及依赖版本保存在 run_manifest.json。
"""
