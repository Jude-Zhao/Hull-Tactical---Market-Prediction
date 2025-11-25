"""
Hull Tactical Competition - Solution B: Parametric Policy Optimization
XGBoost预测模块 + 自动参数调优
(Pure Pandas/Numpy Version)
"""

import pandas as pd
import numpy as np
import os
import sys
from scipy.optimize import minimize
import xgboost as xgb
import warnings

# 添加父目录到Python路径，以便能够导入scores_evaluation模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 忽略不必要的警告
warnings.filterwarnings('ignore')

# =============================================================================
# 1. 全局配置与参数
# =============================================================================

# 全局变量存储模型和优化参数
model_direction = None 
model_volatility = None 
model_return = None 
optimized_params = None # 存储优化后的 [alpha, beta]

# ----------------- 最优超参数 (保持不变) -----------------
BEST_PARAMS_DIRECTION = {
    'n_estimators': 1613, 'max_depth': 5, 'learning_rate': 0.235008845080467,
    'subsample': 0.8387400631785948, 'colsample_bytree': 0.7783331011414365,
    'reg_alpha': 0.9997491581800291, 'reg_lambda': 4.592488919658672,
    'gamma': 0.8667417222278044, 'random_state': 42,
    'objective': 'binary:logistic', 'tree_method': 'hist'
}

BEST_PARAMS_VOLATILITY = {
    'n_estimators': 371, 'max_depth': 9, 'learning_rate': 0.02164141581299458,
    'subsample': 0.88879950890673, 'colsample_bytree': 0.9754210836063002,
    'reg_alpha': 0.007787658410143285, 'reg_lambda': 9.922115592912178,
    'gamma': 0.9234963019255433, 'random_state': 42,
    'objective': 'reg:squarederror', 'tree_method': 'hist'
}

BEST_PARAMS_RETURN = {
    'n_estimators': 137, 'max_depth': 5, 'learning_rate': 0.19089320613723337,
    'subsample': 0.8942521689143974, 'colsample_bytree': 0.987551107426109,
    'reg_alpha': 0.0241272429288264, 'reg_lambda': 9.408535459483122,
    'gamma': 0.9037288822106474, 'random_state': 42,
    'objective': 'reg:squarederror', 'tree_method': 'hist'
}

# ----------------- 特征列表 (保持不变) -----------------
SELECTED_FEATURES_DIRECTION = ['V9', 'lagged_market_forward_excess_returns', 'V3', 'M10', 'P9', 'P5', 'E18', 'P10', 'M18', 'P7', 'V6', 'I4', 'P13', 'S6', 'M17', 'S7', 'P8', 'V5', 'M9', 'S1', 'P6', 'M12', 'V10', 'M8', 'S9', 'S5', 'M5', 'M4', 'I3', 'I6', 'S2', 'P12', 'M16', 'M13', 'P3', 'E4', 'I2', 'P4', 'S11', 'V7', 'V12', 'V13', 'V2', 'S12', 'V1', 'S10', 'P11', 'M3', 'E2', 'M15', 'I8', 'P1', 'M2', 'E9', 'E3', 'M11', 'I1', 'E7', 'E20', 'E15', 'E14', 'E6', 'E16', 'lagged_forward_returns', 'I9', 'V11', 'V8', 'M14', 'V4', 'M6', 'E5', 'E8', 'M1', 'S8', 'I7', 'S4', 'E19', 'E13', 'P2', 'lagged_risk_free_rate', 'S3', 'E12', 'I5', 'E11', 'D5', 'M7', 'E10', 'D7', 'E1', 'D9']
SELECTED_FEATURES_VOLATILITY = ['V7', 'E19', 'M11', 'I6', 'I5', 'P11', 'P6', 'I9', 'I7', 'V10', 'S4', 'M18', 'V1', 'V11', 'M1', 'I3', 'V3', 'lagged_market_forward_excess_returns', 'M10', 'E7', 'lagged_forward_returns', 'V13', 'M12', 'V5', 'M17', 'M2', 'E4', 'M3', 'V9', 'M13']
SELECTED_FEATURES_RETURN = ['I9', 'P5', 'P9', 'V9', 'V13', 'P2', 'lagged_forward_returns', 'V10', 'S1', 'V5', 'M14', 'E3', 'M9', 'M2', 'V2', 'P8', 'S2', 'S6', 'E2', 'V12', 'S10', 'I8', 'S7', 'V7', 'M4', 'P13', 'P6', 'E19', 'I4', 'S5', 'S9', 'M11', 'S3', 'I1', 'V1', 'E14', 'P4', 'S8', 'S4', 'M10', 'V6', 'M13', 'lagged_risk_free_rate', 'E17', 'I7', 'lagged_market_forward_excess_returns', 'M1', 'M12', 'I2', 'M3', 'M6', 'I3', 'S11', 'S12', 'P7', 'P12', 'P10', 'E7', 'M7', 'E16', 'P11', 'I6', 'D7', 'M8', 'M17', 'M18', 'M16', 'V3', 'V8', 'E8', 'E4', 'M15', 'P3', 'V4', 'E5', 'P1', 'E10', 'V11', 'E9', 'M5', 'I5', 'E18', 'E20', 'E12', 'E13']

# =============================================================================
# 2. 辅助功能类 (评价指标)
# =============================================================================

def calculate_sharpe_for_optimization(solution: pd.DataFrame, submission: pd.DataFrame) -> float:
    """
    针对优化器优化的比赛评价指标计算函数
    """
    MAX_INVESTMENT = 2
    MIN_INVESTMENT = 0
    
    solution = solution.reset_index(drop=True).copy()
    submission = submission.reset_index(drop=True)
    solution['position'] = submission['prediction'].values

    # 边界截断
    solution['position'] = solution['position'].clip(MIN_INVESTMENT, MAX_INVESTMENT)

    # 计算策略收益
    solution['strategy_returns'] = solution['risk_free_rate'] * (1 - solution['position']) + \
                                   solution['position'] * solution['forward_returns']

    # 计算指标
    trading_days_per_yr = 252
    strategy_excess = solution['strategy_returns'] - solution['risk_free_rate']
    strategy_mean_excess = strategy_excess.mean()
    strategy_std = solution['strategy_returns'].std()
    
    # 避免分母为0
    if strategy_std <= 1e-9: return -999.0
    
    sharpe = strategy_mean_excess / strategy_std * np.sqrt(trading_days_per_yr)
    strategy_vol = strategy_std * np.sqrt(trading_days_per_yr) * 100

    # 市场基准
    market_excess = solution['forward_returns'] - solution['risk_free_rate']
    market_vol = solution['forward_returns'].std() * np.sqrt(trading_days_per_yr) * 100
    market_mean_excess = market_excess.mean()
    
    # 惩罚项
    excess_vol = max(0, strategy_vol / market_vol - 1.2) if market_vol > 0 else 0
    vol_penalty = 1 + excess_vol
    
    return_gap = max(0, (market_mean_excess - strategy_mean_excess) * 100 * trading_days_per_yr)
    return_penalty = 1 + (return_gap**2) / 100

    adjusted_sharpe = sharpe / (vol_penalty * return_penalty)
    
    if np.isnan(adjusted_sharpe): return -999.0
    return adjusted_sharpe

# =============================================================================
# 3. 策略优化器 (核心改进模块)
# =============================================================================

class StrategyOptimizer:
    def __init__(self):
        # 初始参数猜测 (Alpha, Beta)
        self.params = np.array([0.003, 0.065]) 
        
    def get_position(self, preds_df, params):
        """
        基于非线性对数公式计算仓位
        Formula: pos = alpha * prob * ln(1 + beta/ret) / vol
        """
        alpha, beta = params
        
        ret_pred = preds_df['pred_ret'].values
        vol_pred = preds_df['pred_vol'].values
        prob_pred = preds_df['pred_prob'].values
        
        # 数值稳定性处理
        vol_safe = np.maximum(vol_pred, 1e-6)
        position = np.zeros_like(ret_pred)
        
        # 只计算预测收益为正的部分
        positive_mask = ret_pred > 1e-6
        
        if np.any(positive_mask):
            valid_rets = ret_pred[positive_mask]
            
            # 计算 Log 项
            ratio = beta / valid_rets
            ratio = np.clip(ratio, 1e-10, 1e6) # 防止溢出
            log_term = np.log1p(ratio)
            
            # 核心公式
            calc_pos = alpha * prob_pred[positive_mask] * log_term / vol_safe[positive_mask]
            position[positive_mask] = calc_pos

        return np.clip(position, 0, 2)

    def objective_function(self, params, preds_df, solution_df):
        """目标函数：最小化负的Score"""
        if params[0] <= 0 or params[1] <= 0: return 999.0
            
        current_pos = self.get_position(preds_df, params)
        submission = pd.DataFrame({'prediction': current_pos})
        score = calculate_sharpe_for_optimization(solution_df, submission)
        return -score

    def optimize(self, valid_df, preds_dict):
        """在验证集上优化 Alpha 和 Beta"""
        print("正在基于验证集优化策略参数 (Alpha, Beta)...")
        
        preds_df = pd.DataFrame({
            'pred_ret': preds_dict['return'],
            'pred_vol': preds_dict['volatility'],
            'pred_prob': preds_dict['direction']
        })
        
        solution_df = valid_df[['forward_returns', 'risk_free_rate']].copy()
        
        # 设定参数搜索范围
        # Alpha: [0.0001, 0.1], Beta: [0.001, 0.5]
        bounds = [(1e-5, 0.1), (1e-4, 0.5)]
        
        result = minimize(
            self.objective_function,
            self.params,
            args=(preds_df, solution_df),
            method='L-BFGS-B',
            bounds=bounds,
            tol=1e-5
        )
        
        if result.success:
            print(f"优化成功! Best Score: {-result.fun:.4f}")
            print(f"最优参数: Alpha={result.x[0]:.6f}, Beta={result.x[1]:.6f}")
            self.params = result.x
        else:
            print("优化未完全收敛，使用默认/最后参数")
            
        return self.params

# =============================================================================
# 4. 数据处理与特征工程
# =============================================================================

def preprocess_train_data(train_df):
    """训练数据的批量预处理"""
    # 确保是 DataFrame 副本
    df = train_df.copy()
    df['lagged_forward_returns'] = df['forward_returns'].shift(1)
    df['lagged_risk_free_rate'] = df['risk_free_rate'].shift(1)
    df['lagged_market_forward_excess_returns'] = df['market_forward_excess_returns'].shift(1)
    df = df.dropna().reset_index(drop=True)
    return df

def prepare_features_inference(test_df: pd.DataFrame, feature_list):
    """
    推理时的特征准备 (纯 Pandas 实现)
    假设输入数据中已经包含了所需的滞后特征列
    """
    # 筛选存在的列
    available = [col for col in feature_list if col in test_df.columns]
    
    # 提取数据并填充 NaN
    features = test_df[available].fillna(0.0).values
    return features

# =============================================================================
# 5. 训练主流程
# =============================================================================

def train_and_optimize():
    global model_direction, model_volatility, model_return, optimized_params
    
    print("开始训练流程...")
    train_path = 'D:/CodeingWorkPlace/VScodeWorkplace/Hull Tactical - Market Prediction/train.csv'
    if not os.path.exists(train_path): train_path = '../../train.csv'
    
    raw_df = pd.read_csv(train_path)
    
    # 剔除最后252天的数据用于最终测试 (Local Hold-out)
    if len(raw_df) > 252:
        raw_df = raw_df.iloc[:-252]
        
    full_df = preprocess_train_data(raw_df)
    
    # --- Step 1: 切分训练/验证集 (前80%训练, 后20%优化策略) ---
    split_idx = int(len(full_df) * 0.80)
    train_df = full_df.iloc[:split_idx]
    valid_df = full_df.iloc[split_idx:]
    
    print(f"训练集: {len(train_df)}, 验证集: {len(valid_df)}")

    # --- Step 2: 训练基础模型 ---
    print("训练基础 XGBoost 模型...")
    # Direction
    model_direction = xgb.XGBClassifier(**BEST_PARAMS_DIRECTION)
    model_direction.fit(train_df[SELECTED_FEATURES_DIRECTION], (train_df['forward_returns'] > 0).astype(int))
    # Volatility
    model_volatility = xgb.XGBRegressor(**BEST_PARAMS_VOLATILITY)
    model_volatility.fit(train_df[SELECTED_FEATURES_VOLATILITY], np.abs(train_df['forward_returns']))
    # Return
    model_return = xgb.XGBRegressor(**BEST_PARAMS_RETURN)
    model_return.fit(train_df[SELECTED_FEATURES_RETURN], train_df['forward_returns'])

    # --- Step 3: 验证集预测 ---
    val_preds = {
        'direction': model_direction.predict_proba(valid_df[SELECTED_FEATURES_DIRECTION])[:, 1],
        'volatility': model_volatility.predict(valid_df[SELECTED_FEATURES_VOLATILITY]),
        'return': model_return.predict(valid_df[SELECTED_FEATURES_RETURN])
    }
    
    # --- Step 4: 优化 Alpha 和 Beta ---
    optimizer = StrategyOptimizer()
    optimized_params = optimizer.optimize(valid_df, val_preds)
    
    # --- Step 5: 全量数据重训 ---
    print("使用全量数据重新训练模型...")
    model_direction.fit(full_df[SELECTED_FEATURES_DIRECTION], (full_df['forward_returns'] > 0).astype(int))
    model_volatility.fit(full_df[SELECTED_FEATURES_VOLATILITY], np.abs(full_df['forward_returns']))
    model_return.fit(full_df[SELECTED_FEATURES_RETURN], full_df['forward_returns'])
    
    print("训练流程完成。")

# =============================================================================
# 6. 在线推理接口
# =============================================================================

def predict(test: pd.DataFrame) -> float:
    """
    推理函数 (针对 API 调用)
    注意：输入 test 必须是 pandas DataFrame 且包含所需 lag 特征
    """
    global model_direction, model_volatility, model_return, optimized_params
    
    # 1. 初始化训练
    if optimized_params is None:
        train_and_optimize()
    
    # 2. 准备特征 (直接从 DataFrame 提取)
    feat_dir = prepare_features_inference(test, SELECTED_FEATURES_DIRECTION)[-1:]
    feat_vol = prepare_features_inference(test, SELECTED_FEATURES_VOLATILITY)[-1:]
    feat_ret = prepare_features_inference(test, SELECTED_FEATURES_RETURN)[-1:]
    
    # 3. 模型预测
    p_prob = model_direction.predict_proba(feat_dir)[0, 1]
    p_vol = model_volatility.predict(feat_vol)[0]
    p_ret = model_return.predict(feat_ret)[0]
    
    # 4. 计算最终仓位 (使用优化后的 Alpha/Beta)
    preds_df = pd.DataFrame({
        'pred_ret': [p_ret],
        'pred_vol': [p_vol],
        'pred_prob': [p_prob]
    })
    
    optimizer = StrategyOptimizer()
    position = optimizer.get_position(preds_df, optimized_params)[0]
    
    return float(position)

if __name__ == '__main__':
    # 首先运行训练和优化流程
    train_and_optimize()
    
    # 加载测试数据（最后252天）
    print("\n" + "="*60)
    print("开始测试流程 - 使用最后252天数据评估模型性能")
    print("="*60)
    
    # 读取完整数据
    train_path = 'D:/CodeingWorkPlace/VScodeWorkplace/Hull Tactical - Market Prediction/train.csv'
    
    full_data = pd.read_csv(train_path)
    
    # --- 重要：模拟真实测试场景 ---
    # 我们先在全集上计算好 lag 特征，然后直接切分出最后 252 天。
    # 这模拟了 API 提供包含正确 historical features 的数据的情况。
    full_data_with_lags = preprocess_train_data(full_data)
    
    # 找到最后252天的位置
    test_start_idx = max(0, len(full_data_with_lags) - 252)
    test_df_processed = full_data_with_lags.iloc[test_start_idx:].reset_index(drop=True)
    
    print(f"测试集大小: {len(test_df_processed)}天")
    
    # 使用训练好的模型进行预测
    print("使用训练好的模型进行预测...")
    test_preds = {
        'direction': model_direction.predict_proba(test_df_processed[SELECTED_FEATURES_DIRECTION])[:, 1],
        'volatility': model_volatility.predict(test_df_processed[SELECTED_FEATURES_VOLATILITY]),
        'return': model_return.predict(test_df_processed[SELECTED_FEATURES_RETURN])
    }
    
    # 计算投资比例
    print("计算投资比例...")
    preds_df = pd.DataFrame({
        'pred_ret': test_preds['return'],
        'pred_vol': test_preds['volatility'],
        'pred_prob': test_preds['direction']
    })
    
    optimizer = StrategyOptimizer()
    positions = optimizer.get_position(preds_df, optimized_params)
    
    # 构建提交数据格式
    submission_df = pd.DataFrame({
        'row_id': range(len(positions)),
        'prediction': positions
    })
    
    # 确保测试数据包含必要的列
    test_solution_df = test_df_processed.copy()
    if 'row_id' not in test_solution_df.columns:
        test_solution_df['row_id'] = range(len(test_solution_df))
    
    # 计算score得分
    print("计算最终score得分...")
    try:
        from scores_evaluation import score
        final_score = score(
            solution=test_solution_df,
            submission=submission_df,
            row_id_column_name='row_id'
        )
        print(f"\n最终Score得分: {final_score:.6f}")
    except ImportError:
        print("无法导入score函数，使用内部评价指标...")
        internal_score = calculate_sharpe_for_optimization(test_solution_df, submission_df)
        print(f"\n内部评价指标 (调整后夏普比率): {internal_score:.6f}")
    
    # 额外输出一些统计信息
    print("\n" + "="*60)
    print("投资策略统计信息:")
    print("="*60)
    print(f"平均持仓比例: {np.mean(positions):.4f}")
    print(f"最大持仓比例: {np.max(positions):.4f}")
    print(f"最小持仓比例: {np.min(positions):.4f}")
    print(f"零持仓天数: {(positions == 0).sum()}")
    print(f"满仓(≥1.9)天数: {(positions >= 1.9).sum()}")
    print(f"最优参数 - Alpha: {optimized_params[0]:.6f}, Beta: {optimized_params[1]:.6f}")
    print("="*60)