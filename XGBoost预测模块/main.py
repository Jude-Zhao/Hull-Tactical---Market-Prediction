import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

# 导入现有的预测模块以便复用核心功能
#from XGBoost_predict_submission import prepare_features, determine_investment_ratio


def load_and_preprocess_data(train_path, target_mode='returns'):
    """
    数据集切分与数据预处理的函数 - 支持三种目标变量模式
    
    参数:
        train_path (str): 训练数据的文件路径
        target_mode (str): 目标变量模式，可选：
            'returns' - 使用forward_returns作为目标值（默认）
            'volatility' - 使用当日波动率作为目标值
            'probability' - 使用上涨概率作为目标值（0/1）
        skip_rows (int): 需要跳过的前面几行数据（因为可能有缺失值）
    
    返回:
        tuple: (X_train, X_test, y_train, y_test, feature_columns)
            X_train: 训练集特征
            X_test: 测试集特征
            y_train: 训练集目标值
            y_test: 测试集目标值
            feature_columns: 特征列名称列表
    """
    print(f"正在加载数据: {train_path}")
    
    # 读取训练数据并跳过前N行有缺失的数据
    try:
        train_df = pd.read_csv(train_path)
        print(f"原始数据形状: {train_df.shape}")
        
        # 创建滞后一天的特征
        print("正在创建滞后特征...")
        train_df['lagged_forward_returns'] = train_df['forward_returns'].shift(1)
        train_df['lagged_risk_free_rate'] = train_df['risk_free_rate'].shift(1)
        train_df['lagged_market_forward_excess_returns'] = train_df['market_forward_excess_returns'].shift(1)
        print(f"创建滞后特征后的数据形状: {train_df.shape}")
        # 处理缺失值
        train_df = train_df.dropna()
        print(f"删除缺失值后的数据形状: {train_df.shape}")
        
        
        # 定义需要排除的列（原目标列不再作为特征）
        exclude_cols = ['date_id', 'forward_returns', 'risk_free_rate', 'market_forward_excess_returns']
        feature_columns = [col for col in train_df.columns if col not in exclude_cols]
        
        # 准备特征
        X = train_df[feature_columns].values
        
        # 根据模式选择目标变量
        if target_mode == 'returns':
            y = train_df['forward_returns'].values
            print("目标变量模式: 收益率预测 (forward_returns)")
        elif target_mode == 'volatility':
            # 计算当日波动率作为目标值
            y = np.abs(train_df['forward_returns'].values)
            print("目标变量模式: 波动率预测 (当日波动率)")
        elif target_mode == 'probability':
            # 计算上涨概率（0表示下跌，1表示上涨）
            y = (train_df['forward_returns'] > 0).astype(int).values
            print("目标变量模式: 上涨概率预测 (0=下跌, 1=上涨)")
        else:
            raise ValueError(f"不支持的target_mode: {target_mode}。可选值: 'returns', 'volatility', 'probability'")
        
        
        # 提取最后252天作为测试集，剩余作为训练集
        test_size = 252
        if len(X) < test_size:
            raise ValueError(f"数据量不足，需要至少{test_size}天数据，实际只有{len(X)}天")
        
        X_train, X_test = X[:-test_size], X[-test_size:]
        y_train, y_test = y[:-test_size], y[-test_size:]
        
        print(f"训练集大小: {len(X_train)}天")
        print(f"测试集大小: {len(X_test)}天（最后252天）")
        print(f"特征维度: {X.shape[1]}")
        print(f"目标变量模式: {target_mode}")
        print(f"滞后特征已添加: lagged_forward_returns, lagged_risk_free_rate, lagged_market_forward_excess_returns")
        
        return X_train, X_test, y_train, y_test, feature_columns
        
    except Exception as e:
        print(f"数据加载和预处理出错: {e}")
        raise


def feature_selection(X_train, X_test, y_train, feature_columns, top_n=40):
    """
    特征筛选的函数，使用XGBoost的特征重要性进行筛选
    
    参数:
        X_train: 训练集特征
        X_test: 测试集特征
        y_train: 训练集目标值
        feature_columns: 特征列名称列表
        top_n: 选择前N个重要的特征
    
    返回:
        tuple: (X_train_selected, X_test_selected, selected_features)
            X_train_selected: 筛选后的训练集特征
            X_test_selected: 筛选后的测试集特征
            selected_features: 筛选后的特征名称列表
    """
    print(f"正在进行特征筛选，选择前{top_n}个重要特征")
    
    # 训练一个基础XGBoost模型来获取特征重要性
    base_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=20,
        learning_rate=0.02,
        random_state=42
    )
    base_model.fit(X_train, y_train)
    
    # 获取特征重要性
    importances = base_model.feature_importances_
    
    # 获取前N个重要特征的索引
    indices = np.argsort(importances)[::-1][:top_n]
    
    # 获取筛选后的特征名称
    selected_features = [feature_columns[i] for i in indices]
    
    # 筛选特征
    X_train_selected = X_train[:, indices]
    X_test_selected = X_test[:, indices]
    
    print(f"特征筛选完成，选择的特征: {selected_features}")
    
    # 绘制特征重要性图
    plt.figure(figsize=(12, 8))
    plt.title('特征重要性排序')
    plt.bar(range(len(indices)), importances[indices])
    plt.xticks(range(len(indices)), selected_features, rotation=90)
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.show()  # 显示图形
    plt.close()  # 显示后再关闭
    
    return X_train_selected, X_test_selected, selected_features


def train_xgboost_model(X_train, y_train, X_val=None, y_val=None, params=None):
    """
    模型训练的函数
    
    参数:
        X_train: 训练集特征
        y_train: 训练集目标值
        X_val: 验证集特征（可选）
        y_val: 验证集目标值（可选）
        params: XGBoost模型参数（可选）
    
    返回:
        model: 训练好的XGBoost模型
    """
    print("正在训练XGBoost模型...")
    
    # 默认参数（与贝叶斯优化参数空间一致）
    default_params = {
        'n_estimators': 1000,           # 贝叶斯优化范围: 100-2000
        'max_depth': 6,                 # 贝叶斯优化范围: 3-12
        'learning_rate': 0.01,          # 贝叶斯优化范围: 0.005-0.3
        'subsample': 0.8,               # 贝叶斯优化范围: 0.6-1.0
        'colsample_bytree': 0.8,        # 贝叶斯优化范围: 0.6-1.0
        'reg_alpha': 0.1,               # 贝叶斯优化范围: 0-10
        'reg_lambda': 0.1,              # 贝叶斯优化范围: 0-10
        'gamma': 0.0,                   # 贝叶斯优化范围: 0.8-1.0（新添加）
        'random_state': 42,
        'objective': 'reg:squarederror',
        'tree_method': 'hist',
        'early_stopping_rounds': 50,
        'eval_metric': 'rmse'
    }
    
    # 如果提供了参数，则更新默认参数
    if params is not None:
        default_params.update(params)
        # 移除early_stopping_rounds以便在GridSearchCV中使用
        default_params.pop('early_stopping_rounds', None)
    
    # 创建模型
    model = xgb.XGBRegressor(**default_params)
    
    # 训练模型
    if X_val is not None and y_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=100
        )
    else:
        model.fit(X_train, y_train)
    
    print("XGBoost模型训练完成")
    return model


def hyperparameter_tuning(X_train, y_train, task_type='return'):
    """
    使用贝叶斯优化进行超参数调优
    
    参数:
        X_train: 训练集特征
        y_train: 训练集目标值
        task_type: 任务类型 ('volatility', 'direction', 'return')
    
    返回:
        best_params: 最佳超参数组合
    """
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args
    from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score
    import numpy as np
    
    print(f"正在进行贝叶斯优化超参数调优... 任务类型: {task_type}")
    
    # 定义参数搜索空间（扩展）
    dimensions = [
        Integer(100, 2000, name='n_estimators'),
        Integer(3, 12, name='max_depth'),
        Real(0.005, 0.3, name='learning_rate'),
        Real(0.6, 1.0, name='subsample'),
        Real(0.6, 1.0, name='colsample_bytree'),
        Real(0, 10, name='reg_alpha'),
        Real(0, 10, name='reg_lambda'),
        Real(0.8, 1.0, name='gamma')
    ]
    
    # 时间序列交叉验证
    tscv = TimeSeriesSplit(n_splits=5)
    
    @use_named_args(dimensions)
    def objective(**params):
        """目标函数，根据任务类型选择不同的评分标准"""
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            tree_method='hist',
            random_state=42,
            **params
        )
        
        scores = []
        for train_idx, val_idx in tscv.split(X_train):
            # 处理不同类型的数据（pandas DataFrame/Series 或 numpy array）
            if hasattr(X_train, 'iloc'):  # pandas DataFrame/Series
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            else:  # numpy array
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            
            if task_type == 'volatility':
                # 预测波动率：使用RMSE
                score = np.sqrt(mean_squared_error(y_val, y_pred))
            elif task_type == 'direction':
                # 预测涨跌方向：使用F1指标
                y_val_binary = (y_val > 0).astype(int)
                y_pred_binary = (y_pred > 0).astype(int)
                score = -f1_score(y_val_binary, y_pred_binary)  # 负号因为要最小化
            else:  # 'return'
                # 预测具体回报率：使用MAE
                score = mean_absolute_error(y_val, y_pred)
            
            scores.append(score)
        
        return np.mean(scores)
    
    # 执行贝叶斯优化
    print("开始贝叶斯优化搜索...")
    result = gp_minimize(
        func=objective,
        dimensions=dimensions,
        n_calls=50,  # 搜索次数
        random_state=42,
        verbose=True
    )
    
    # 提取最佳参数
    best_params = {
        'n_estimators': result.x[0],
        'max_depth': result.x[1],
        'learning_rate': result.x[2],
        'subsample': result.x[3],
        'colsample_bytree': result.x[4],
        'reg_alpha': result.x[5],
        'reg_lambda': result.x[6],
        'gamma': result.x[7]
    }
    
    print(f"贝叶斯优化完成，最佳参数: {best_params}")
    print(f"最佳得分: {result.fun:.6f}")
    
    return best_params

    

def determine_investment_ratio(return_prediction, volatility_prediction, up_probability, 
                             alpha=1.0, beta=0.01):
    """
    根据多维预测信息确定投资比例的分段非线性函数
    
    基于数学公式实现动态投资比例决策：
    w = 0, 当 r_pred ≤ 0
    w = α · p_up · ln(1 + β / r_pred) / σ, 当 r_pred > 0
    
    参数:
        return_prediction (float): 预测收益率 r_pred
        volatility_prediction (float): 波动率预测 σ（必须 > 0）
        up_probability (float): 上涨概率 p_up（0-1之间）
        alpha (float): 整体缩放系数，调整持仓绝对大小（默认1.0）
        beta (float): 非线性调节参数，控制"预测收益越高，持仓下降速度"的陡峭程度（默认0.01）
    
    返回:
        float: 投资比例 w，范围在[0, 1]之间
    
    数学原理:
    - 当预测收益率为负或零时，策略返回0（不投资）
    - 当预测收益率为正时，使用对数函数实现非线性增长
    - 对数函数确保"收益越高，持仓增长越平缓"的特性
    - 波动率作为分母，实现风险控制（波动率越高，持仓越谨慎）
    - 上涨概率强化正向信号（上涨概率越高，持仓越积极）
    """
    
    # ========== 输入参数有效性校验 ==========
    
    # 检查上涨概率是否在有效范围内
    if not (0 <= up_probability <= 1):
        raise ValueError(f"上涨概率 p_up 必须在[0, 1]范围内，当前值: {up_probability}")
    
    # 检查波动率是否大于零（避免除零错误）
    if volatility_prediction <= 0:
        raise ValueError(f"波动率 σ 必须大于0，当前值: {volatility_prediction}")
    
    # 检查缩放参数是否为正数
    if alpha <= 0:
        raise ValueError(f"缩放系数 α 必须大于0，当前值: {alpha}")
    
    # 检查非线性调节参数是否为正数
    if beta <= 0:
        raise ValueError(f"非线性调节参数 β 必须大于0，当前值: {beta}")
    
    # ========== 分段函数实现 ==========
    
    # 情况1: 预测收益率为负或零，不投资
    if return_prediction <= 0:
        return 0.0
    
    # 情况2: 预测收益率为正，使用非线性函数计算投资比例
    else:
        try:
            # 计算对数项: ln(1 + β / r_pred)
            # 确保 β / r_pred > 0 且避免数值溢出
            ratio = beta / return_prediction
            
            # 数值稳定性检查
            if ratio > 1e6:  # 避免数值过大
                ratio = 1e6
            elif ratio < 1e-10:  # 避免数值过小
                ratio = 1e-10
                
            log_term = np.log(1 + ratio)
            
            # 计算最终投资比例: α · p_up · log_term / σ
            investment_ratio = alpha * up_probability * log_term / volatility_prediction
            
            # ========== 结果边界约束 ==========
            
            # 确保投资比例在合理范围内 [0, 2]
            investment_ratio = max(0.0, min(2.0, investment_ratio))
            
            # 保留适当的小数位数（4位小数）
            return round(investment_ratio, 4)
            
        except (OverflowError, ValueError, ZeroDivisionError) as e:
            # 数值计算出错时的安全处理
            print(f"警告: 投资比例计算出现数值错误: {e}，返回保守值0.1")
            return 0.1

from scores_evaluation import score


def optimize_investment_params(return_prediction, volatility_prediction, up_probability, test_data):
    """
    确定最优持仓函数超参数 alpha 和 beta 的函数
    
    该函数通过网格搜索的方式，遍历不同的 alpha 和 beta 参数组合，
    对每个组合计算投资比例并使用 score 函数进行评分，最终选择
    评分最高的参数组合作为最优超参数。
    
    参数:
        return_prediction: 预测收益率数组 (numpy array)
        volatility_prediction: 波动率预测数组 (numpy array) 
        up_probability: 上涨概率数组 (numpy array)
        test_data: 测试集数据 (pd.DataFrame)，包含 forward_returns 和 risk_free_rate 列
    
    返回:
        dict: 包含最优参数和评分的字典
              {
                  'best_alpha': float,  # 最优 alpha 参数
                  'best_beta': float,   # 最优 beta 参数
                  'best_score': float,  # 最高评分
                  'alpha_grid': list,   # 搜索的 alpha 网格
                  'beta_grid': list,    # 搜索的 beta 网格
                  'results': list       # 所有组合的搜索结果
              }
    """
    
    print("开始优化投资比例函数超参数 alpha 和 beta...")
    
    # ========== 输入数据验证 ==========
    # 检查输入数组长度是否一致
    if len(test_data) != len(volatility_prediction) or len(return_prediction) != len(up_probability):
        raise ValueError("输入数组的长度不一致！")
    
    # ========== 定义参数搜索网格 ==========
    # alpha 控制整体持仓大小，beta 控制收益-持仓非线性关系
    alpha_range = np.arange(0.5, 2.5, 0.1)  # alpha 从 0.5 到 2.5，步长 0.1
    beta_range = np.arange(0.001, 0.05, 0.002)  # beta 从 0.001 到 0.05，步长 0.002
    
    print(f"alpha 搜索范围: {alpha_range[0]:.3f} 到 {alpha_range[-1]:.3f}，共 {len(alpha_range)} 个值")
    print(f"beta 搜索范围: {beta_range[0]:.3f} 到 {beta_range[-1]:.3f}，共 {len(beta_range)} 个值")
    print(f"总共需要测试 {len(alpha_range) * len(beta_range)} 种参数组合")
    
    best_score = -float('inf')  # 初始化最佳评分为负无穷
    best_alpha = None
    best_beta = None
    all_results = []  # 记录所有结果
    
    total_combinations = len(alpha_range) * len(beta_range)
    current_combination = 0
    
    # ========== 网格搜索优化 ==========
    for alpha in alpha_range:
        for beta in beta_range:
            current_combination += 1
            
            try:
                # 对当前参数组合计算投资比例
                investment_ratios = np.array([
                    determine_investment_ratio(
                        return_prediction=ret_pred,
                        volatility_prediction=vol_pred,
                        up_probability=up_prob,
                        alpha=alpha,
                        beta=beta
                    )
                    for ret_pred, vol_pred, up_prob in zip(
                        return_prediction, 
                        volatility_prediction, 
                        up_probability
                    )
                ])
                
                # 构建提交数据格式用于 score 函数
                submission_data = pd.DataFrame({
                    'row_id': test_data['row_id'].values if 'row_id' in test_data.columns else range(len(investment_ratios)),
                    'prediction': investment_ratios
                })
                
                # 调用 score 函数进行评分
                current_score = score(
                    solution=test_data,
                    submission=submission_data,
                    row_id_column_name='row_id'
                )
                
                # 记录当前结果
                result = {
                    'alpha': alpha,
                    'beta': beta,
                    'score': current_score,
                    'avg_investment_ratio': np.mean(investment_ratios),
                    'max_investment_ratio': np.max(investment_ratios),
                    'min_investment_ratio': np.min(investment_ratios)
                }
                all_results.append(result)
                
                # 更新最佳参数
                if current_score > best_score:
                    best_score = current_score
                    best_alpha = alpha
                    best_beta = beta
                
                # 进度显示
                if current_combination % 50 == 0 or current_combination == total_combinations:
                    progress = current_combination / total_combinations * 100
                    print(f"进度: {current_combination}/{total_combinations} ({progress:.1f}%) - 当前最佳评分: {best_score:.6f}")
                
            except Exception as e:
                print(f"警告: 参数组合 (alpha={alpha:.3f}, beta={beta:.3f}) 计算失败: {e}")
                continue
    
    # ========== 输出优化结果 ==========
    print("\n" + "="*60)
    print("超参数优化完成！")
    print("="*60)
    print(f"最优 alpha 参数: {best_alpha:.3f}")
    print(f"最优 beta 参数: {best_beta:.3f}")
    print(f"最高评分: {best_score:.6f}")
    
    # 统计信息
    valid_results = [r for r in all_results if not np.isnan(r['score'])]
    if valid_results:
        scores = [r['score'] for r in valid_results]
        print(f"总共测试了 {len(valid_results)} 个有效参数组合")
        print(f"评分统计 - 最高: {np.max(scores):.6f}, 最低: {np.min(scores):.6f}, 平均: {np.mean(scores):.6f}")
    
    # 返回结果
    optimization_result = {
        'best_alpha': best_alpha,
        'best_beta': best_beta,
        'best_score': best_score,
        'alpha_grid': alpha_range.tolist(),
        'beta_grid': beta_range.tolist(),
        'results': all_results
    }
    
    return optimization_result


# ========== 使用示例 ==========
# 
# # 示例 1: 基本使用
# optimization_result = optimize_investment_params(
#     return_prediction=y_pred,           # 模型预测的收益率
#     volatility_prediction=vol_pred,     # 波动率预测
#     up_probability=up_prob,             # 上涨概率预测
#     test_data=test_data_df              # 包含 forward_returns 和 risk_free_rate 的 DataFrame
# )
#
# # 示例 2: 获取最优参数
# best_alpha = optimization_result['best_alpha']
# best_beta = optimization_result['best_beta']
# best_score = optimization_result['best_score']
#
# # 示例 3: 查看详细结果
# results_df = pd.DataFrame(optimization_result['results'])
# results_df = results_df.sort_values('score', ascending=False)
# print("前10个最佳参数组合:")
# print(results_df.head(10))
#
# # 示例 4: 可视化参数搜索结果（需要 matplotlib）
# import matplotlib.pyplot as plt
# pivot_scores = results_df.pivot_table(
#     values='score', 
#     index='alpha', 
#     columns='beta', 
#     aggfunc='mean'
# )
# plt.figure(figsize=(10, 8))
# plt.imshow(pivot_scores.values, cmap='viridis', aspect='auto')
# plt.colorbar(label='Score')
# plt.title('超参数搜索热力图')
# plt.xlabel('Beta 参数索引')
# plt.ylabel('Alpha 参数索引')
# plt.show()


def evaluate_model(model, X_test, y_test):
    """
    针对测试集结果做预测得分的函数
    
    参数:
        model: 训练好的模型 
        X_test: 测试集特征
        y_test: 测试集目标值
        scaler: 用于标准化的StandardScaler对象（可选）
        selected_features: 选择的特征名称列表（可选）
    
    返回:
        dict: 包含各种评估指标的字典
    """
    print("正在评估模型性能...")
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 计算评估指标
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = np.mean(np.abs(y_test - y_pred))
    
    # ========== 计算投资策略的表现 ==========
    # 注意：由于新版本的 determine_investment_ratio 需要三个参数，
    # 我们需要为波动率预测和上涨概率提供合理的估计值
    
    # 模拟波动率预测（基于历史数据的简单估计）
    volatility_estimates = np.std(y_test) * np.ones_like(y_pred)  # 使用测试集的标准差作为波动率估计
    up_probability_estimates = np.clip((y_pred > 0).astype(float), 0.1, 0.9)  # 基于预测收益率计算上涨概率
    
    # 计算投资比例
    investment_ratios = np.array([
        determine_investment_ratio(
            return_prediction=pred,
            volatility_prediction=vol,
            up_probability=prob,
            alpha=1.0,
            beta=0.01
        ) 
        for pred, vol, prob in zip(y_pred, volatility_estimates, up_probability_estimates)
    ])
    
    # 计算策略收益
    strategy_returns = investment_ratios * y_test
    cumulative_returns = np.cumsum(strategy_returns)
    
    # 评估指标字典
    metrics = {
        'RMSE': rmse,
        'R2': r2,
        'MAE': mae,
        'Total_Return': np.sum(strategy_returns),
        'Win_Rate': np.mean(strategy_returns > 0)
    }
    
    print(f"模型评估结果:")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")
    
    # 绘制预测vs实际值图
    plt.figure(figsize=(12, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel('实际收益率')
    plt.ylabel('预测收益率')
    plt.title('预测vs实际收益率')
    plt.grid(True)
    plt.savefig('prediction_vs_actual.png')
    plt.close()
    
    # 绘制累积收益图
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_returns)
    plt.xlabel('时间步')
    plt.ylabel('累积收益')
    plt.title('策略累积收益')
    plt.grid(True)
    plt.savefig('cumulative_returns.png')
    plt.close()
    
    return metrics


def main():
    """
    主函数，整合所有功能进行模型训练和评估
    """
    print("=== XGBoost模型测试脚本 ===")
    
    # 数据路径 - 根据实际情况调整
    # 首先尝试读取本地数据
    local_train_path = '../../train.csv'
    # 如果本地数据不存在，则使用Kaggle路径（用于兼容性）
    train_path = local_train_path if os.path.exists(local_train_path) else '/kaggle/input/hull-tactical-market-prediction/train.csv'
    
    try:
        # 1. 数据加载和预处理
        X_train, X_test, y_train, y_test, feature_columns, _ = load_and_preprocess_data(train_path)
        
        # 2. 特征筛选
        X_train_selected, X_test_selected, selected_features = feature_selection(
            X_train, X_test, y_train, feature_columns, top_n=20
        )
        
        # 3. 超参数筛选（可选，如果数据集较大可以注释掉以节省时间）
        # best_params = hyperparameter_tuning(X_train_selected, y_train)
        
        # 4. 训练模型
        # 使用默认参数或最佳参数
        # model = train_xgboost_model(X_train_selected, y_train, X_test_selected, y_test, best_params)
        model = train_xgboost_model(X_train_selected, y_train, X_test_selected, y_test)
        
        # 5. 进行预测以获取超参数优化所需的数据
        y_pred = model.predict(X_test_selected)
        
        # 为超参数优化准备输入数据
        # 注意：这些数据需要根据实际项目结构调整
        volatility_prediction = np.std(y_pred) * np.ones_like(y_pred)  # 简单波动率估计
        up_probability = np.clip((y_pred > 0).astype(float), 0.1, 0.9)  # 基于预测的上涨概率
        
        # 构建测试数据格式（需要根据实际的 test_data 结构调整列名）
        test_data = pd.DataFrame({
            'row_id': range(len(y_test)),
            'forward_returns': y_test.values if hasattr(y_test, 'values') else y_test,
            'risk_free_rate': np.full(len(y_test), 0.02),  # 假设无风险利率为 2%
        })
        
        # 6. 优化投资比例函数超参数
        optimization_result = optimize_investment_params(
            return_prediction=y_pred,
            volatility_prediction=volatility_prediction,
            up_probability=up_probability,
            test_data=test_data
        )
        
        # 7. 使用最优参数评估模型
        print("\n使用最优超参数重新评估模型...")
        # 使用最优参数重新计算投资比例和评估指标
        optimal_investment_ratios = np.array([
            determine_investment_ratio(
                return_prediction=ret_pred,
                volatility_prediction=vol_pred,
                up_probability=up_prob,
                alpha=optimization_result['best_alpha'],
                beta=optimization_result['best_beta']
            )
            for ret_pred, vol_pred, up_prob in zip(y_pred, volatility_prediction, up_probability)
        ])
        
        # 构建提交数据并计算最终评分
        submission_data = pd.DataFrame({
            'row_id': test_data['row_id'],
            'prediction': optimal_investment_ratios
        })
        
        final_score = score(
            solution=test_data,
            submission=submission_data,
            row_id_column_name='row_id'
        )
        
        print(f"最终优化后评分: {final_score:.6f}")
        
        # 传统的模型评估（可选）
        metrics = evaluate_model(model, X_test_selected, y_test)
        
        # 保存模型
        model.save_model('xgboost_model.json')
        print("模型已保存为 'xgboost_model.json'")
        
        print("\n=== 测试完成 ===")
        
    except Exception as e:
        print(f"测试过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()