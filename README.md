# 毕业设计：可审计的补货需求预测

Python 3.10+。只读原始 Excel，动态取运行月份之前的12个完整自然月，仅保留 `Normal` 和 `Phasing Out`，完成真实周销量清洗、ADI/CV² 分类、异常大单比较、时间顺序验证与未来16周预测。不使用强化学习。

## 运行

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m replenishment --input "..\Replenishment Forecastint Report All Field (1).xlsx"
```

默认 `reference_date` 为系统日期，输出在项目 `outputs`。重现指定日期：

```powershell
python -m replenishment --reference-date 2026-10-23
python -m replenishment --inspect-only --reference-date 2026-10-23
python -m unittest discover -s tests -v
```

macOS/Linux 使用 `.venv/bin/python`。也可 `pip install -e .` 后执行 `replenishment-forecast`。
源文件路径、输出路径、配置路径均可通过 `--input`、`--output`、`--config` 指定。无需修改代码中的年月。
`get_analysis_period(reference_date=None, months=12)` 为独立函数；正式研究配置限定12个月。

## 已检查的真实结构

源文件为 `Replenishment Forecastint Report All Field (1).xlsx`，单工作表 `sheet1`，8,099条产品行、220列、两行表头。
产品状态确认为 `产品基本信息 | 产品状态`，不能用 `总部跨工厂物料状态` 替代。
SKU键为销售组织 + 库位 + 物料；物料按字符串读取并保留前导零。库位为空时明确保留未知标识。
`config.json` 中的字段映射来自对原始工作簿的只读检查。换工作簿后先执行 `--inspect-only`，检查字段候选，再修改配置；字段缺失或有多个可能候选时，程序停止正式分析。

真实周销量由“实际销量”组下带年份和周次的字段识别，日期不依赖列位置。无年份的 `Oct/Nov/...` 月列不反推年份、不加入周数据。
源工作簿未声明周历；目前明确采用可验证的 ISO 8601 周一至周日解释。公司财务周历不同时，需先提供映射；当前代码拒绝其他未经实现的日历。
分析窗口在所有统计之前应用，跨入当前月的周也整周排除，不将周销量拆成虚假逐日值。

## 质量规则

- 产品状态先去首尾空格、压缩多余空白、忽略大小写，只精确映射 Normal / Phasing Out。空值、近似词和其他状态全部排除。
- 日期筛选和状态筛选均先于SKU分类、异常值、训练和汇总。筛选日志同时记录产品行与SKU期间行，记录单位分开标明。
- 业务已确认忽略所有Son。先标准化真实“级别”字段并精确排除Son，保留Parent，不将父子相加；剩余未解决的重复键、缺失销量、无效物料或单位冲突才隔离。层级字段缺失时停止，避免漏掉Son。
- 周/月销量均已在退货到账/记账期间扣减，是净销量。保留正负原值，不再次扣退货、不移回原销售月、不把负数改为零，也不仅因负数隔离整个SKU。
- 零销量保留；已知连续来源区间内个别SKU缺周才补零。所有SKU同时缺少的整周属于未知来源覆盖，停止建模。来源最早日期之前不会补零。
- 当前库存、在途、欠单、公司预测缺少可验证的历史生效时间，故不进入历史分析或库存回测。产品状态只作为当前产品队列筛选，回溯结果存在生命周期/存活者选择偏差。
- 原始工作簿始终只读；开始和结束核对SHA-256。源文件不得放在输出目录内。

## 分类、异常值与预测

ADI=T/正需求期间数；CV²=正需求方差/正需求均值²。默认阈值1.32和0.49，采用总体方差(ddof=0)，等于阈值时归入高ADI/高CV²组。全零为Zero；单个正需求点的CV²暂记0并标记不可靠。Normal/Phasing Out独立分组。
分类依据为正净销量，含负值时标为暂定，不能由净额还原毛需求。没有正值但存在负值的纯退货序列单列为不可分类，不伪装为Zero；五类占比以全部合格SKU为分母。

Sen slope为所有两两时间点斜率的中位数。前后半段原销量Median与Sen slope均上升时，高销量只标记；其余正销量IQR超限点分别保留Raw、截尾到上限、替换为历史非零Median。至少4个正需求点才启用替换。阈值与趋势判断均可配置。
完整窗口异常比较只用于描述；验证/测试中的异常估计只用当时训练窗口重新计算。清洗文件和测试实际需求保持原值。
Raw/Winsorized/Median均保留原始负净销量，仅对符合规则的正净销量大单进行处理。

13/26/52周历史窗口必须先通过样本充足性检查。默认保留最后16周作为测试集，在更早数据上做1周滚动验证（至少3个原点、最多8个），所有可比窗口共用原点。验证MAE选择方法及Raw/Winsorized/Median版本，分数相同优先26周。测试结果不参与再选择。
单独输出窗口可用性、滚动验证、候选测试比较及已选模型测试。类型分组按各原点训练数据计算，避免用全窗口分类泄漏。

候选：MovingAverage、SES、Croston、SBA、TSB。平滑参数固定在配置，支持确定性重跑；Croston以第一个正需求大小和到该点的间隔初始化，SBA乘以(1-alpha/2)，TSB每期更新正需求概率。无正需求的训练窗预测0。
MovingAverage/SES支持原始正负净销量；Croston/SBA/TSB在训练窗含负净销量时标为不适用，并输出逐阶段适用性审计，绝不静默截断历史负值。补货数量预测输出端下限为0；实际值与测试目标不截断。
只比较在所有共同验证原点均适用的候选。历史测试起点的模型选择不查看测试实际；最终重估按原验证排名和当前训练窗适用性选择，分别记录历史已选模型和最终模型，测试误差不参与再选择。
MAE、WAPE、Bias均保留，Bias正值表示高估；零分母WAPE记为缺失，不记0。测试真实需求不进行异常替换。
净销量WAPE分母为原始净销量绝对值之和，避免正负值相互抵消。
公司预测缺少`issued_at/forecast_version`时只标为不可比较，不拿当前预测冒充历史预测。

最终未来16个完整周从运行日期之后下一个周一开始。训练末周与未来起点之间仅用预测跨越，记录未观察的桥接周数，不使用当前月份实际值。
当前方法为水平预测，跨时滞外推不产生额外已观测证据；最后16周测试并不直接证明桥接后所有更长提前期的精度。
Phasing Out检查相对最近13周异常增长、长串零销量后仍产生正预测；输出人工复核标识，不自动创建补货单。

只有月度来源时完成月清洗、月分类、月异常比较，明确停止周级实验；不拆分或模拟周销量。
长表格式必须明确配置 `grain`，程序不凭稀疏日期间距猜测月/周/交易粒度。

## 输出

至少生成：

- `data_dictionary.xlsx`：字段、缺失率、样例、类型、用途、合并表头、候选字段。
- `data_quality_report.xlsx`：质量问题、隔离SKU、覆盖范围及缺失输入。
- `filter_audit.xlsx`：动态日期、当前月排除、状态映射、记录/SKU数量及排除数量。
- `cleaned_last_12_complete_months.csv`：真实未被异常替换的合格需求；`material`为字符串，其他程序导入时同样需明确字符串类型。
- `sku_classification.xlsx`：需求类型、ADI、CV²、状态分组数量及占比。
- `outlier_comparison.xlsx`：Raw/Winsorized/Median与趋势证据。
- `forecast_evaluation.xlsx`：验证、测试、可用窗口、模型选择和不可比较基线。
- `analysis_summary.md`：本次结论、限制、后续数据需求。

附加CSV保存完整逐SKU验证/测试、16周预测、隔离记录、窗口可用性；`run_manifest.json`记录运行配置、依赖、输入指纹与验收状态。
`excluded_son_records.csv`记录被规则排除的Son；`method_applicability.csv`记录因负净销量而不适用的方法及阶段。
先检查manifest的 `status == "success"` 才接受当前运行结果。失败重跑会写入明确失败状态，目录中旧文件不能视为本次有效结果；源数据错误不会被隐藏。
每个Excel包含运行上下文，使用数值/日期类型，不输出可执行来源公式。

## 代码结构

`config.py` 配置；`periods.py` 动态日期与严格区间；`excel_io.py` 只读表头/字段；`status.py` 状态映射；`cleaning.py` 唯一键/质量/补周；`classification.py` ADI；`outliers.py` 训练期异常；`forecasting.py` 预测公式；`evaluation.py` 时间验证与测试；`export.py` 导出核验；`pipeline.py` 流水线；`reporting.py` 中文报告。

## 当前不能完成的库存实验

真实周销量可以支持初步需求预测；可信每周库存回测还需要每周库存起点、实际收货与到货日期、订单/欠单/缺货历史、生效交期、单位换算和价值字段。销量不是无约束需求，零销售可能来自缺货。
52周模型必须具有52周历史及独立评估点，不能从分析窗口之前或当前月取数据补足。

## 没有价格表时的计算口径

本项目目前只计算数量：逐SKU预测、MAE和Bias的单位与源销量一致；WAPE为数量误差比例，没有金额权重。例如实际净销量100、预测120，绝对误差20，单期WAPE为20%，完全不需要价格。
没有价格表时，不计算销售额、库存金额、利润、持有成本、缺货成本、补货金额或金额加权误差。未将缺失价格设成0或1，也未以销量ABC冒充价值ABC。
数量库存回测本身可在补齐库存/订单/到货记录后开展；金额成本或价值分析还需要带物料、币种、价格种类和生效日期的成本/售价表。

## 方法来源

- Syntetos, Boylan & Croston (2005), *On the categorization of demand patterns*, [DOI](https://doi.org/10.1057/palgrave.jors.2601841)：ADI/CV²分类框架。
- Teunter, Syntetos & Babai (2011), *Intermittent demand: Linking forecasting to inventory obsolescence*, [DOI](https://doi.org/10.1016/j.ejor.2011.05.018)：TSB与淘汰风险。
- [SAP TSB官方说明](https://help.sap.com/docs/SAP_HANA_PLATFORM/2cfbc5cf2bc14f028cfbe2a2bba60a50/82038471a3294627a6838f625dcdc2cd.html)：每期更新需求发生概率。

测试中的人工数据仅为隔离的单元测试夹具，从未用作毕业设计业务结果。原始Excel、分析输出和业务记录通过 `.gitignore` 排除，不同步到GitHub。
