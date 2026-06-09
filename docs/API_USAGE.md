# alpha_factory Python API 用法说明

本文档根据当前 `alpha_factory` 目录下真实源码扫描整理，排除了 `__MACOSX/`、`__pycache__/` 等解压或缓存文件。说明重点是每个 `.py` 文件的用途、主要调用入口和常见调用方式。

## 1. 总体使用链路

`alpha_factory` 的主流程可以理解为：

1. 用 `factor.storage` 或外部数据源准备 long/pivot 因子数据。
2. 用 `factor.dataloader` 按因子依赖加载数据。
3. 继承或调用 `factor.development` 中的因子基类和算子生成新因子。
4. 用 `factor.evaluation` 做质量、IC、PNL、相关性分析。
5. 用 `model` 或 `contrib.model` 训练模型并输出预测。
6. 用 `backtest` 把信号、成交价、可交易矩阵转成仓位和收益。

基础数据格式：

- long 格式：`DataFrame` 使用 `MultiIndex(["Datetime", "Contract"])`，列为一个或多个因子字段。
- pivot 格式：`DataFrame.index.name == "Datetime"`，`DataFrame.columns.name == "Contract"`。
- `Datetime` 默认要求带时间分量，例如 `2024-01-02 15:00:00`。

最小示例：

```python
from alpha_factory.factor.utils import pivot_to_long, long_to_pivot
from alpha_factory.factor.storage.disk import DailySeparatedDiskStorage

long_df = pivot_to_long(close_pivot, factor_name="Close")
storage = DailySeparatedDiskStorage(root_dir="./factor_store")
storage.save(table_name="price", factor_name="Close", data=close_pivot, freq="1day")
loaded_close = storage.load(table_name="price", factor_name="Close", freq="1day")
```

## 2. 根目录

### `main.py`

当前为空文件，没有可调用 API。项目 README 把它定义为未来入口，但目前不要依赖它。

## 3. `factor/` 因子模块

### `factor/utils.py`

用途：定义数据格式常量、long/pivot 校验与互转。

常量：

- `DATETIME_LEVEL = "Datetime"`
- `CONTRACT_LEVEL = "Contract"`
- `TRADE_END_TIME = "15:00:00"`
- `MULTI_INDEX = ["Datetime", "Contract"]`

主要 API：

- `validate_long_format(long_df, require_time_component=True, strict=False)`
- `validate_pivot_format(pivot_df, require_time_component=True, strict=False)`
- `pivot_to_long(pivot_df, factor_name="factor") -> DataFrame`
- `long_to_pivot(long_df, factor_name=None) -> DataFrame`
- `pivots_to_long(pivot_dfs, factor_names=None, use_tqdm=False) -> DataFrame`
- `long_to_pivots(long_df, factor_names=None, use_tqdm=False) -> List[DataFrame]`

调用示例：

```python
from alpha_factory.factor.utils import validate_pivot_format, pivot_to_long, long_to_pivots

validate_pivot_format(close_pivot)
long_df = pivot_to_long(close_pivot, "Close")
close_pivot, volume_pivot = long_to_pivots(long_df, ["Close", "Volume"])
```

### `factor/dataloader/base.py`

用途：定义加载器抽象接口。

主要 API：

- `BaseDataloader.load(dependencies, *args, **kwargs) -> Dict[str, DataFrame]`

一般不直接实例化，新增数据加载方式时继承它。

### `factor/dataloader/disk.py`

用途：从 `DailySeparatedDiskStorage` 读取因子数据，返回与 `FactorBase.dependencies` 对齐的字典。

主要 API：

- `DiskDataloader(storage=None, root_dir=None)`
- `load_table(table_name, factor_names, start_date=None, end_date=None, freq=None, use_tqdm=False, parallel=True) -> DataFrame`
- `load(dependencies, start_date=None, end_date=None, freq=None, use_tqdm=False, parallel=True) -> Dict[str, DataFrame]`

调用示例：

```python
from alpha_factory.factor.dataloader.disk import DiskDataloader

dataloader = DiskDataloader(root_dir="./factor_store")
data = dataloader.load(
    dependencies={"price": ["Close.1day", "Volume.1day"]},
    start_date="20240101",
    end_date="20241231",
)
price_long = data["price"]
```

字段名支持 `factor.freq` 格式；不带后缀时默认因子名本身，频率由 `freq` 参数或底层默认值决定。

### `factor/dataloader/qhdata.py`

当前为空文件，没有可调用 API。

### `factor/dataloader/__init__.py`

包级导出：

- `BaseDataloader`
- `DiskDataloader`

可直接调用：

```python
from alpha_factory.factor.dataloader import DiskDataloader
```

### `factor/development/base.py`

用途：因子研发核心基类。

主要 API：

- `UpdateMode.FULL`
- `UpdateMode.INCREMENTAL`
- `FactorBase(name, dependencies, description="", version="1.0")`
- `FactorBase.compute(data) -> DataFrame`：子类必须实现。
- `FactorBase.run(data) -> DataFrame`：标准执行流程，包含依赖校验、数据格式校验、计算、后处理。
- `AdaptiveUpdateFactor(..., update_mode=UpdateMode.FULL, lookback_steps=0, ...)`
- `AdaptiveUpdateFactor.run(data, calc_datetime=None, return_last=False) -> DataFrame`
- `ParallelFactor(..., n_jobs=-1, parallel_backend="loky", use_tqdm=False)`
- `ParallelFactor.split_compute_units(data)`：子类必须实现。
- `ParallelFactor.compute_one_unit(unit_data)`：子类必须实现。
- `HighFreqToDailyFactor(..., history_days=0, daily_timestamp="15:00:00", anchor_data_key=None)`
- `HighFreqToDailyFactor.series_to_pivot(series, index_value, index_name="Datetime", columns_name="Contract")`

新增普通因子示例：

```python
from alpha_factory.factor.development.base import AdaptiveUpdateFactor, UpdateMode

class MyFactor(AdaptiveUpdateFactor):
    def __init__(self):
        super().__init__(
            name="MyFactor",
            dependencies={"price": ["Close.1day"]},
            update_mode=UpdateMode.FULL,
        )

    def compute(self, data):
        price = data["price"]
        out = price[["Close"]].copy()
        out["my_factor"] = out["Close"].pct_change()
        return out[["my_factor"]]

factor = MyFactor()
result = factor.run({"price": price_long})
```

新增高频转日频因子时，继承 `HighFreqToDailyFactor`，一般只实现 `compute_one_unit(unit_data)`。

### `factor/development/ops.py`

用途：pivot 格式上的常用因子算子。所有函数输入通常是 `Datetime x Contract` 的 pivot `DataFrame`，输出仍是 pivot `DataFrame`。

异常策略：

- `set_safe_apply_error_policy(policy="warning") -> str`
- `get_safe_apply_error_policy() -> str`

元素级算子：

- 逻辑/检查：`dot_not`、`dot_isna`、`dot_notna`、`dot_isfinite`、`dot_notfinite`
- 单输入数学：`dot_abs`、`dot_sign`、`dot_log`、`dot_sqrt`、`dot_exp`、`dot_inverse`、`dot_reverse`、`dot_sigmoid`、`dot_tanh`、`dot_purify`、`dot_arctan`、`dot_arcsin`、`dot_arccos`、`dot_s_log_1p`
- 取整/裁剪：`dot_round`、`dot_floor_down`、`dot_floor`、`dot_ceil`、`dot_nanclip`
- 双输入运算：`dot_div`、`dot_mul`、`dot_add`、`dot_sub`、`dot_max`、`dot_min`、`dot_power`、`dot_spower`、`dot_slog`
- 比较/条件：`dot_gt`、`dot_ge`、`dot_lt`、`dot_le`、`dot_ne`、`dot_eq`、`dot_or`、`dot_and`、`dot_negmask`、`dot_if_else`

时序算子：

- 滚动统计：`ts_mean`、`ts_skew`、`ts_kurt`、`ts_max`、`ts_min`、`ts_median`、`ts_sum`、`ts_std`、`ts_cv`、`ts_ir`
- 滞后/变化：`ts_shift`、`ts_delta`、`ts_log_diff`、`ts_return`
- 标准化/排名/衰减：`ts_ema`、`ts_zscore`、`ts_rank`、`ts_decay_linear`
- 位置/分位/差异：`ts_idxmax`、`ts_idxmin`、`ts_quantile`、`ts_mad`、`ts_mindiff`、`ts_maxdiff`、`ts_av_diff`、`ts_mmdiff`
- 回归/相关：`ts_slope`、`ts_beta`、`ts_resi`、`ts_rsquare`、`ts_corr`、`ts_cov`
- 其他：`ts_prod`、`ts_countna`、`ts_scale`、`ts_step`、`ts_moment`、`ts_ffill`、`ts_eudldist`、`ts_jsdist`

横截面算子：

- `cs_rank`、`cs_zscore`、`cs_scale`、`cs_minmax`、`cs_normalize`、`cs_winsorize`、`cs_corr`
- `cs_group_neutralize(df, group_data=None, parallel=True, max_workers=None, use_tqdm=False)`
- `cs_resi(df, by, parallel=True, max_workers=None, use_tqdm=False)`
- `cs_beta(df, by, parallel=True, max_workers=None, use_tqdm=False)`
- `cs_rsquare(df, by, parallel=True, max_workers=None, use_tqdm=False)`

调用示例：

```python
from alpha_factory.factor.development.ops import ts_return, ts_std, cs_zscore

rtn = ts_return(close_pivot, periods=1)
vol = ts_std(rtn, window=20)
score = cs_zscore(vol)
```

### `factor/development/utils.py`

用途：研发辅助校验。

主要 API：

- `validate_input(df, expected_cols=None) -> DataFrame`

### `factor/storage/base.py`

用途：存储抽象接口。

主要 API：

- `BaseStorage.save(**kwargs)`
- `BaseStorage.load(**kwargs)`

### `factor/storage/disk.py`

用途：本地磁盘存储。目录结构是 `{root_dir}/{table_name}/{YYYYMMDD}/{factor_name}.{freq}.pkl`，并维护 `.metadata.json`。

主要 API：

- `DiskStorage(root_dir)`
- `create_table(table_name)`
- `delete_table(table_name)`
- `show_tables() -> List[str]`
- `show_factors(table_name) -> List[str]`
- `show_factor_meta(table_name, factor_name=None, freq=None) -> Dict`
- `rebuild_metadata(table_name) -> Dict`
- `DailySeparatedDiskStorage(root_dir)`
- `save(table_name, factor_name, data, start_date=None, end_date=None, freq=None, use_tqdm=False, parallel=True)`
- `load(table_name, factor_name, start_date=None, end_date=None, freq=None, use_tqdm=False, parallel=True) -> DataFrame`
- `update(table_name, factor_name, data, start_date=None, end_date=None, freq=None, inplace=True, use_tqdm=False, parallel=True)`
- `delete(table_name, factor_name, start_date=None, end_date=None, freq=None, use_tqdm=False, parallel=True)`

调用示例：

```python
from alpha_factory.factor.storage.disk import DailySeparatedDiskStorage

storage = DailySeparatedDiskStorage("./factor_store")
storage.save("price", "Close", close_pivot, freq="1day")
close = storage.load("price", "Close", start_date="20240101", end_date="20240131")
storage.update("price", "Close", close_update_pivot)
storage.delete("price", "Close", start_date="20240101", end_date="20240101")
```

单因子读取返回 pivot；多因子读取返回 long：

```python
long_df = storage.load("price", ["Close", "Volume"], freq="1day")
```

### `factor/storage/database.py`

用途：数据库 long 表存储，支持 MySQL 和 ClickHouse。

通用抽象 API：

- `DatabaseStorage(database)`
- `save(table_name, data, factor_name=None, chunk_size=5000)`
- `load(table_name, factor_names=None, start_datetime=None, end_datetime=None, contracts=None) -> DataFrame`
- `create_table(table_name, data_schema)`
- `delete_table(table_name)`
- `show_tables() -> List[str]`
- `show_factors(table_name) -> List[str]`
- `update(table_name, data, factor_name=None, chunk_size=5000)`
- `delete(table_name, start_datetime=None, end_datetime=None, contracts=None, factor_names=None)`

MySQL：

```python
from alpha_factory.factor.storage.database import MysqlStorage

storage = MysqlStorage(
    username="user",
    password="pwd",
    host="localhost",
    port=3306,
    database="alpha",
)
storage.save("factor_table", long_df)
loaded = storage.load("factor_table", factor_names=["Close"], start_datetime="2024-01-01")
```

ClickHouse：

```python
from alpha_factory.factor.storage.database import ClickhouseStorage

storage = ClickhouseStorage(host="localhost", port=8123, username="default", database="alpha")
storage.save("factor_table", long_df)
```

### `factor/evaluation/base.py`

用途：评估器抽象与 Plotly 可视化工具。

主要 API：

- `BaseAnalyzer.analyze(**kwargs)`
- `PlotlyMixin.show(fig, layout_kwargs=None)`
- `PlotlyMixin.get_series_fig(...)`
- `PlotlyMixin.get_bar_fig(...)`
- `PlotlyMixin.get_histogram_fig(...)`
- `PlotlyMixin.get_heatmap_fig(...)`

### `factor/evaluation/quality.py`

用途：因子质量分析。

主要 API：

- `FactorQualityAnalyzer(**kwargs)`
- `analyze(factor_mtx, tradeable_mtx=None, return_figure=False)`
- `detect_outliers(...)`
- `calc_coverage(...)`
- `calc_distribution(...)`
- `calc_unique_value_ratio(...)`
- `calc_histogram(...)`

### `factor/evaluation/ic.py`

用途：IC / RankIC 分析。

主要 API：

- `ICAnalyzer(label_mtx, **kwargs)`
- `analyze(factor_mtx, tradeable_mtx=None, return_figure=False, label_mtx=None)`
- `calc_daily_ic(...)`
- `calc_monthly_ic(...)`
- `check_label(label_mtx)`

### `factor/evaluation/pnl.py`

用途：按因子分组计算 PNL。

主要 API：

- `PNLAnalyzer(trade_price_mtx, trade_freq=1, **kwargs)`
- `analyze(factor_mtx, n_groups=5, tradeable_mtx=None, return_figure=False, trade_price_mtx=None)`
- `calc_group_pnl(...)`
- `check_price(price_mtx)`

### `factor/evaluation/corr.py`

用途：因子相关性和因子 PNL 相关性分析。

主要 API：

- `CorrAnalyzer(factor_bank, factor_pnl_bank=None, **kwargs)`
- `analyze(factor_mtx, factor_pnl, method="pearson", return_figure=False, factor_bank=None, factor_pnl_bank=None)`
- `calc_cs_corr(...)`
- `calc_pnl_corr(...)`
- `calc_pnl_corr_mtx(...)`
- `get_factor_clusters(factor_corr_mtx)`
- `check_factor_bank(factor_bank=None)`

评估示例：

```python
from alpha_factory.factor.evaluation.quality import FactorQualityAnalyzer
from alpha_factory.factor.evaluation.ic import ICAnalyzer

quality = FactorQualityAnalyzer().analyze(factor_pivot, tradeable_mtx=tradeable)
ic = ICAnalyzer(label_mtx=label_pivot).analyze(factor_pivot, tradeable_mtx=tradeable)
```

### `factor/__init__.py`、`factor/config/__init__.py`、`factor/development/__init__.py`、`factor/evaluation/__init__.py`、`factor/storage/__init__.py`

包初始化文件。其中 `factor/storage/__init__.py` 导出了 `BaseStorage`、`DiskStorage`、`ClickhouseStorage`、`MysqlStorage`；`factor/evaluation/__init__.py` 导出了评估器类。其他初始化文件当前没有额外逻辑。

## 4. `model/` 模型模块

### `model/base.py`

用途：统一模型协议和 MLflow 保存/加载能力。

主要 API：

- `MODEL_TYPE.SKLEARN`
- `MODEL_TYPE.LIGHTGBM`
- `MODEL_TYPE.PYTORCH`
- `BaseModel(model_type, model_name=None, **params)`
- `BaseModel.fit(X, y, **kwargs)`
- `BaseModel.predict(X, return_dataframe=True)`
- `BaseModel.evaluate(X_test, y_test) -> Dict[str, float]`
- `MLFlowMixin.save_to_mlflow(experiment_name="Default", run_name=None, tracking_uri=None, tags=None) -> (run_id, artifact_uri)`
- `MLFlowMixin.load_from_mlflow(tracking_uri, run_id)`
- `MLFlowCVMixin`：用于多 fold / CV 模型保存加载。

模型类约定：训练后设置 `self.is_fitted = True`，预测默认返回与 `X.index` 对齐的 `DataFrame`。

### `model/sklearn.py`

用途：sklearn 模型统一适配层。

主要 API：

- `SklearnBaseModel(model_name="SklearnBaseModel", **kwargs)`
- `fit(X, y, **fit_params)`
- `predict(X, return_dataframe=True)`
- `predict_proba(X, return_dataframe=True)`
- `evaluate(X_test, y_test)`
- `get_feature_importance() -> DataFrame`

该类要求子类在初始化时设置 `self.model` 为实际 sklearn estimator。

### `model/lightgbm.py`

用途：LightGBM 模型统一适配层。

主要 API：

- `LightGBMBaseModel(model_name="LightGBMBaseModel", **kwargs)`
- `predict(X, return_dataframe=True)`
- `evaluate(X_test, y_test)`
- `get_feature_importance(importance_type="gain") -> DataFrame`

### `model/torch.py`

用途：PyTorch 训练、预测、时序数据集封装。

主要 API：

- `EarlyStopping(patience=10, min_delta=0, restore_best_weights=True, greater_better=False)`
- `TimeSeriesDataset(X, y=None, seq_len=20, padding_value=0.0, datetime_level="Datetime", contract_level="Contract")`
- `TimeSeriesDataLoader(dataset, batch_size=-1, shuffle=False, drop_last=False, num_workers=0, prefetch_factor=2, include_metadata=True)`
- `PytorchBaseModel(num_epochs=100, batch_size=2048, gradient_clip=0.5, valid_ratio=0.1, shuffle=True, verbose=True, device=None, objective="mse", feval="mse", optimizer_type="adam", learning_rate=0.001, weight_decay=0.0, use_lr_scheduler=True, lr_scheduler_type="cosine", lr_scheduler_params=None, use_early_stopping=True, early_stopping_patience=10, model_name="PytorchBaseModel", **kwargs)`
- `PytorchBaseModel.fit(X, y, **kwargs)`
- `PytorchBaseModel.predict(X, return_dataframe=True)`
- `PytorchBaseModel.evaluate(X_test, y_test)`
- `PytorchBaseModel.get_deep_feature(X)`
- `PytorchBaseModel.param_num() -> int`
- `PytorchTimeSeriesBaseModel(seq_len=20, padding_value=0.0, drop_last=False, model_name="PytorchTimeSeriesBaseModel", **kwargs)`

具体 PyTorch 子类应提供 `self.model` 或重写构建网络逻辑。

### `model/metrics.py`

用途：模型评估指标。

主要 API：

- `get_regression_metrics(y_test, y_pred)`
- `get_classification_metrics(y_test, y_pred_proba)`

回归指标包含 R2、MSE、MAE 等；若输入是带 `Datetime` 多级索引的数据，还会计算 IC / RankIC 类截面相关指标。

### `model/manager.py`

用途：MLflow 实验和嵌套 run 管理。

主要 API：

- `RunType.TRAINING`
- `RunType.WINDOW`
- `RunType.PARAM_SET`
- `MLFlowExperimentManager(tracking_uri="./mlruns", experiment_name=None)`
- `set_tracking_uri(tracking_uri=None)`
- `set_experiment(experiment_name=None)`
- `start_run(run_name, tags=None, params=None, nested=False)`
- `training_run(run_name, tags=None)`
- `window_run(run_name, window_index, start_date, end_date, tags=None)`
- `param_set_run(run_name, param_index, params, tags=None)`
- `experiment_id`

调用示例：

```python
from alpha_factory.model.manager import MLFlowExperimentManager

manager = MLFlowExperimentManager("./mlruns", "demo")
with manager.training_run("train_lgb") as run_id:
    ...
```

### `model/train.py`

用途：时间序列滚动训练编排。

主要 API：

- `get_rolling_windows(all_dates, train_start_date, test_start_date, stride=120, rolling_type="sliding", horizon=2)`
- `rolling_training(model_class, model_params, X, y, train_start_date, test_start_date, stride=120, horizon=2, rolling_type="sliding", experiment_manager=None, tags=None, eval_on_test=True, save_artifacts=True, **fit_kwargs)`
- `param_grid_rolling_train(model_class, params_grid, X, y, train_start_date, test_start_date, stride=120, horizon=2, rolling_type="sliding", experiment_manager=None, tags=None, eval_on_test=True, save_artifacts=True, **fit_kwargs)`

调用示例：

```python
from alpha_factory.model.train import rolling_training
from alpha_factory.contrib.model.Ridge import RidgeModel

model = rolling_training(
    model_class=RidgeModel,
    model_params={"alpha": 1.0},
    X=X,
    y=y,
    train_start_date="2020-01-01",
    test_start_date="2022-01-01",
)
```

### `model/trainer.py`

用途：PyTorch 训练循环控制。

主要 API：

- `get_active_trainer()`
- `TorchTrainer(num_epochs, verbose=True)`
- `TorchTrainer.fit(train_epoch_fn, val_epoch_fn=None, model=None, early_stopping=None, epoch_logger=None) -> Dict[str, object]`
- 进度辅助：`create_progress_iterator(dataloader, phase)`、`set_batch_postfix(**kwargs)`

### `model/utils.py`

用途：模型数据类型转换、MLflow run 查询、模型恢复。

主要 API：

- `to_numpy(*datas)`
- `to_tensor(*datas, device=None)`
- `delete_failed_runs(experiment_ids, tracking_uri=None)`
- `get_tracking_uri_candidates(project_root=None) -> List[str]`
- `resolve_tracking_uri_for_run(run_id, tracking_uri_candidates=None, project_root=None) -> str`
- `get_run_info_by_id(run_id, tracking_uri=None) -> Dict`
- `get_model_class_registry() -> Dict[str, Type]`
- `resolve_model_class_from_run(run_id, tracking_uri=None, model_registry=None)`
- `load_model_by_run_id(run_id, tracking_uri=None, model_class=None, model_registry=None, project_root=None)`
- `get_experiment_id_by_name(experiment_name, tracking_uri=None) -> str`
- `get_all_experiment_infos(tracking_uri=None, include_deleted=False, max_results=5000) -> DataFrame`
- `search_runs_by_tags(experiment_name, tags, tracking_uri=None, max_results=100) -> DataFrame`
- `get_model_infos_by_experiment_name(experiment_name, tracking_uri=None, max_results=5000, only_model_runs=True, order_by=None) -> DataFrame`
- `get_child_run_ids(parent_run_id, experiment_name, tracking_uri=None, order_by=None) -> List[str]`
- `load_pickle_artifact_by_run_id(run_id, artifact_path="pred.pkl", tracking_uri=None)`

### `model/__init__.py`

当前为空文件，没有额外导出。

## 5. `contrib/model/` 可用模型

这些文件是可直接实例化的模型实现。

通用调用：

```python
model = SomeModel(...)
model.fit(X_train, y_train)
pred = model.predict(X_test)
metrics = model.evaluate(X_test, y_test)
run_id, artifact_uri = model.save_to_mlflow("experiment_name")
```

### `contrib/model/LinearRegression.py`

- `LinearRegressionModel(fit_intercept=True, model_name="LinearRegressionModel", **kwargs)`

### `contrib/model/Ridge.py`

- `RidgeModel(alpha=1.0, fit_intercept=True, max_iter=None, tol=0.001, solver="auto", random_state=None, model_name="RidgeModel", **kwargs)`
- `RidgeCVModel(alphas=None, fit_intercept=True, cv=5, scoring=None, model_name="RidgeCVModel", **kwargs)`

### `contrib/model/Lasso.py`

- `LassoModel(alpha=1.0, fit_intercept=True, max_iter=None, tol=0.001, selection="cyclic", random_state=None, model_name="LassoModel", **kwargs)`
- `LassoCVModel(alphas=None, fit_intercept=True, cv=5, precompute="auto", selection="cyclic", model_name="LassoCVModel", **kwargs)`

### `contrib/model/ElasticNet.py`

- `ElasticNetModel(alpha=1.0, l1_ratio=0.5, fit_intercept=True, precompute=False, max_iter=1000, copy_X=True, tol=0.0001, warm_start=False, positive=False, random_state=None, selection="cyclic", model_name="ElasticNetModel", **kwargs)`
- `ElasticNetCVModel(l1_ratio=None, eps=0.001, n_alphas=100, alphas=None, fit_intercept=True, normalize="deprecated", precompute="auto", max_iter=1000, tol=0.0001, cv=None, copy_X=True, n_jobs=None, positive=False, random_state=None, model_name="ElasticNetCVModel", **kwargs)`

### `contrib/model/OMP.py`

- `OMPModel(n_nonzero_coefs=None, tol=None, fit_intercept=True, normalize="deprecated", precompute="auto", model_name="OMPModel", **kwargs)`
- `OMPCVModel(fit_intercept=True, normalize="deprecated", max_iter=None, cv=5, n_jobs=None, model_name="OMPCVModel", **kwargs)`

### `contrib/model/PLSR.py`

- `PLSRModel(n_components=None, scale=True, max_iter=500, tol=1e-6, model_name="PLSRModel", **kwargs)`

### `contrib/model/KNeighbor.py`

- `KNNRegressorModel(n_neighbors=5, weights="uniform", algorithm="auto", leaf_size=30, p=2, model_name="KNNRegressorModel", metric="minkowski", metric_params=None, **kwargs)`
- `KNNClassifierModel(n_neighbors=5, weights="uniform", algorithm="auto", leaf_size=30, p=2, metric="minkowski", metric_params=None, model_name="KNNClassifierModel", **kwargs)`

### `contrib/model/LightGBM.py`

- `LightGBMModel(lgb_params=None, valid_ratio=0.2, feval=None, shuffle=True, random_state=42, early_stopping_rounds=50, verbose=100, model_name="LightGBMModel", **kwargs)`
- `LightGBMCVModel(lgb_params=None, n_splits=5, shuffle=True, feval=None, random_state=42, early_stopping_rounds=50, verbose=100, model_name="LightGBMCVModel", **kwargs)`
- `lgb_ic_metric(preds, train_data)`
- `lgb_rankic_metric(preds, train_data)`

### `contrib/model/MLP.py`

- `MLPModel(input_dim, hidden_dim=64, num_layers=3, output_dim=1, dropout=0.2, model_name="MLPModel", **kwargs)`
- `MLPCVModel(input_dim, hidden_dim=64, num_layers=3, output_dim=1, dropout=0.2, model_name="MLPCVModel", n_splits=5, random_state=42, **kwargs)`
- `ResidualBlock(hidden_dim, dropout=0.3)`
- `MLP(input_dim=101, hidden_dim=32, num_layers=3, dropout=0.2, output_dim=1)`

### `contrib/model/GRU.py`

- `MovingAverage(kernel_size)`
- `SeriesDecomposition(kernel_size)`
- `GRUModel(input_dim, hidden_dim=64, num_layers=1, output_dim=1, dropout=0.0, bidirectional=False, decomposition=False, decomposition_kernel_size=10, model_name="GRUModel", **kwargs)`
- `GRUCVModel(input_dim, hidden_dim=64, num_layers=1, output_dim=1, dropout=0.0, bidirectional=False, decomposition=False, decomposition_kernel_size=10, n_splits=5, random_state=42, model_name="GRUCVModel", **kwargs)`
- `GRUNet(input_dim, hidden_dim, num_layers, output_dim, dropout=0.0, bidirectional=False, decomposition=False, decomposition_kernel_size=10)`

### `contrib/model/Boost.py`

用途：多阶段残差模型。每一阶段拟合上一阶段残差，预测时累加各阶段预测。

主要 API：

- `BoostModel(model_classes, model_params, model_name="BoostModel")`
- `fit(X, y)`
- `predict(X, return_dataframe=True)`
- `evaluate(X_test, y_test)`
- `num_stages`

调用示例：

```python
from alpha_factory.contrib.model.Boost import BoostModel
from alpha_factory.contrib.model.Ridge import RidgeModel
from alpha_factory.contrib.model.Lasso import LassoModel

model = BoostModel(
    model_classes=[RidgeModel, LassoModel],
    model_params=[{"alpha": 1.0}, {"alpha": 0.01}],
)
model.fit(X_train, y_train)
pred = model.predict(X_test)
```

### `contrib/model/__init__.py`

当前为空文件，没有额外导出。

## 6. `contrib/factor/` 实验因子

### `contrib/factor/development/demo_factor.py`

示例因子：

- `RtnStatOrthSizeFactor(name, dependencies, windows=[5, 10, 20], update_mode=UpdateMode.DAILY, lookback_steps=21, description="", version="1.0")`
- `compute(data)`

注意：当前 `UpdateMode` 只有 `FULL` 和 `INCREMENTAL`，代码中的 `UpdateMode.DAILY` 与当前枚举不一致，直接运行可能报错，需要改成 `UpdateMode.INCREMENTAL` 或 `UpdateMode.FULL`。

### `contrib/factor/development/factor_set1.py`

用途：`AlphaAlgoSetFactor` 日频统计因子集合。

主要 API：

- `find_longest_consecutive(arr)`
- `AlphaAlgoSetFactor(name, dependencies, window=20, selected_factors=None, update_mode=UpdateMode.FULL, lookback_steps=0, description="", version="1.0", n_jobs=-1)`
- `compute(data) -> DataFrame`

典型依赖：`Close`、`High`、`Low`、`Open`、`Volume`。内部包含上下行波动、偏度、自相关、RSRS、成交量结构等多个私有计算函数。

### `contrib/factor/development/active_buy_sell.py`

用途：高频主动买卖概率因子，继承 `HighFreqToDailyFactor`。

主要 API：

- `CJSCDistFactor(name="CJSCDistFactor", dependencies={"price": ["Open.5min", "Close.5min", "Amount.5min", "Volume.5min"]}, window=20, selected_factors=None, update_mode=UpdateMode.FULL, lookback_steps=0, description="主动买入概率因子（高频转日频）", version="1.0", n_jobs=-1, parallel_backend="loky", use_tqdm=False, daily_timestamp="15:00:00", anchor_data_key=None, **kwargs)`
- `compute_one_unit(unit_data) -> DataFrame`

### `contrib/factor/development/feat_ig.py`

用途：高频信息几何日频因子。

主要 API：

- `HighFreqIGFactor(name="HighFreqIG", dependencies={"price": ["Close.5min", "Volume.5min"]}, update_mode=UpdateMode.FULL, lookback_steps=0, history_days=3, description=..., version="1.0", n_jobs=-1, parallel_backend="loky", use_tqdm=True, daily_timestamp="15:00:00", anchor_data_key=None, r_col="Close", l_col="Amount", window=48, z_win=48, erf_k=6, vol_regime_min_periods=6, **kwargs)`
- `compute_one_unit(unit_data)`

注意：默认依赖里写的是 `Volume.5min`，但默认 `l_col="Amount"`，调用前应确认输入列名与参数一致。

### `contrib/factor/development/rp_factor.py`

用途：高频价量因子，继承 `HighFreqToDailyFactor`。

主要 API：

- `RpFactor(name="RpFactor", dependencies={"price": ["Open.5min", "Close.5min", "Amount.5min", "Volume.5min"]}, window=30, m=9, selected_factors=None, update_mode=UpdateMode.FULL, lookback_steps=0, description="高频价量因子（高频转日频）", version="1.0", n_jobs=-1, parallel_backend="loky", use_tqdm=False, daily_timestamp="15:00:00", anchor_data_key=None, **kwargs)`
- `compute_one_unit(unit_data) -> DataFrame`

内部私有函数覆盖潮汐、峰谷成交量、排列熵、FFT 能量等特征。

### `contrib/factor/development/rpt_factor.py`

用途：Rpt 高频转日频因子集合。

主要 API：

- `RptFactor(name="RptFactor", dependencies={"price": ["Open.5min", "Close.5min", "Amount.5min", "Volume.5min"]}, window=20, selected_factors=None, update_mode=UpdateMode.FULL, lookback_steps=0, description="Rpt因子（高频转日频）", version="1.0", n_jobs=-1, parallel_backend="loky", use_tqdm=False, daily_timestamp="15:00:00", anchor_data_key=None, **kwargs)`
- `compute_one_unit(unit_data) -> DataFrame`

内部计算流动性、换手率相关性、FFT 周期能量等特征。

### `contrib/factor/development/ops_long.py`

用途：long 格式上的因子算子，与 `factor/development/ops.py` 的 pivot 算子类似。

主要 API：

- 元素级：`dot_abs`、`dot_sign`、`dot_log`、`dot_sqrt`、`dot_not`、`dot_div`、`dot_mul`、`dot_add`、`dot_sub`、`dot_max`、`dot_gt`、`dot_ge`、`dot_lt`、`dot_le`、`dot_ne`、`dot_or`、`dot_eq`
- 时序：`ts_mean`、`ts_ref`、`ts_skew`、`ts_kurt`、`ts_max`、`ts_cv`、`ts_min`、`ts_median`、`ts_sum`、`ts_std`、`ts_ema`、`ts_zscore`、`ts_delta`、`ts_return`、`ts_rank`、`ts_decay_linear`、`ts_idxmax`、`ts_idxmin`、`ts_quantile`、`ts_mad`、`ts_slope`、`ts_rsquare`、`ts_corr`、`ts_euclidean_dist`、`ts_cosine_similarity`、`ts_jensenshannon_dist`、`ts_cov`
- 横截面：`cs_rank`、`cs_zscore`、`cs_scale`、`cs_normalize`、`cs_neutralize`、`cs_winsorize`、`cs_corr`

### `contrib/factor/development/utils_long.py`

主要 API：

- `validate_input(df, expected_cols=None) -> DataFrame`
- `align_dataframes(df1, df2, how="inner", expected_cols=None) -> tuple`

### `contrib/factor/pool/residual_return.py`

用途：残差收益因子。

主要 API：

- `ResidualReturnFactor(base_factors, dependencies=None, update_mode=UpdateMode.DAILY, description="Residual Return Factor", version="1.0")`
- `compute(data) -> DataFrame`

注意：这里也使用了当前枚举中不存在的 `UpdateMode.DAILY`，直接实例化可能报错，需与 `factor/development/base.py` 的枚举保持一致。

### `contrib/__init__.py`、`contrib/factor/__init__.py`、`contrib/factor/development/__init__.py`

当前为空文件，没有额外导出。

## 7. `backtest/` 回测模块

### `backtest/data.py`

用途：回测数据对齐和按时间戳取数。

主要 API：

- `BaseDataHandler.get_data(timestamp) -> Dict[str, Any]`
- `VectorizedDataHandler(start_datetime, end_datetime, signal_mtx, exec_price_mtx, tradeable_mtx, standard_mtx=None, fill_method=np.nan, verbose=True)`
- `get_data(timestamp) -> Dict[str, Any]`
- `get_aligned_data(data_type) -> DataFrame`
- `get_original_data(data_type) -> DataFrame`

`data_type` 常用值：`signal`、`exec_price`、`tradeable`、`standard`。

### `backtest/position.py`

用途：由信号生成目标仓位。

主要 API：

- `BasePositionGenerator(allow_short=False, **kwargs)`
- `VectorPositionGenerator(signal_processing="raw", allow_short=False, **kwargs)`
- `VectorPositionGenerator.get_position(signal, tradeable, prev_position=None, **kwargs)`
- `FuturePositionGenerator(signal_processing="raw", allow_short=True, **kwargs)`
- `StockTopKPositionGenerator(k=10, max_turnover=None, equal_weight=True, signal_processing="raw", **kwargs)`

`signal_processing` 支持原始信号、rank/zscore 等处理逻辑，具体以源码实现为准。

### `backtest/pnl.py`

用途：根据前后仓位和成交价计算收益、换手和交易成本。

主要 API：

- `BasePnlCalculator(commission_rate=0.0003, slippage_rate=0.0002, **kwargs)`
- `VectorPnlCalculator.get_pnl(prev_position, cur_position, prev_price, exec_price, **kwargs) -> Dict[str, Any]`
- `VectorPnlCalculator.get_turnover_rate(prev_position, cur_position, **kwargs)`
- `VectorPnlCalculator.get_trading_cost(prev_position, cur_position, **kwargs)`
- `FuturePnlCalculator(commission_rate=0.0003, slippage_rate=0.0002, **kwargs)`
- `StockPnlCalculator(commission_rate=0.0003, slippage_rate=0.0002, stamp_tax_rate=0.001, **kwargs)`

### `backtest/engine.py`

用途：向量化回测编排。

主要 API：

- `BaseBacktestEngine(trade_freq=1)`
- `should_trade(step) -> bool`
- `VectorBacktestEngine(data_handler, position_generator, pnl_calculator, trade_freq=1, **kwargs)`
- `VectorBacktestEngine.run_backtest() -> Dict[str, DataFrame]`

调用示例：

```python
from alpha_factory.backtest.data import VectorizedDataHandler
from alpha_factory.backtest.position import StockTopKPositionGenerator
from alpha_factory.backtest.pnl import StockPnlCalculator
from alpha_factory.backtest.engine import VectorBacktestEngine

handler = VectorizedDataHandler(
    start_datetime="2024-01-01",
    end_datetime="2024-12-31",
    signal_mtx=signal,
    exec_price_mtx=price,
    tradeable_mtx=tradeable,
)
engine = VectorBacktestEngine(
    data_handler=handler,
    position_generator=StockTopKPositionGenerator(k=20, max_turnover=0.2),
    pnl_calculator=StockPnlCalculator(),
    trade_freq=1,
)
result = engine.run_backtest()
positions = result["positions"]
pnl_results = result["pnl_results"]
```

### `backtest/utils.py`、`backtest/__init__.py`

当前为空文件，没有可调用 API。

## 8. `strategy/` 与 `utils/`

### `strategy/base.py`、`strategy/__init__.py`

当前为空文件，没有可调用 API。策略模块在 README 中有规划，但源码尚未实现。

### `utils/__init__.py`

当前为空文件，没有可调用 API。

## 9. 依赖提示

源码中用到的主要三方库包括：

- 数据与科学计算：`pandas`、`numpy`、`scipy`、`statsmodels`
- 机器学习：`scikit-learn`、`lightgbm`
- 深度学习：`torch`
- 实验追踪：`mlflow`
- 并行/进度：`joblib`、`tqdm`
- 可视化：`plotly`
- 数据库：`pymysql`、`clickhouse_connect` 或源码环境中对应 ClickHouse 客户端

安装依赖后，建议在项目上层目录运行脚本，保证 `import alpha_factory...` 可以被 Python 正确解析。
