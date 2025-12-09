import os
import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
import kaggle_evaluation.default_inference_server

# ==================== 自定义损失函数 ====================

def asymmetric_return_loss(y_pred, dtrain, direction_penalty=3.0, huber_delta=0.02):
    """收益率预测：方向错误惩罚 + Huber平滑"""
    y_true = dtrain.get_label()
    residual = y_pred - y_true
    is_wrong_direction = (y_pred * y_true) < 0
    
    abs_res = np.abs(residual)
    is_small_error = abs_res <= huber_delta
    
    grad = np.where(is_small_error, residual, huber_delta * np.sign(residual))
    hess = np.where(is_small_error, 1.0, huber_delta / (abs_res + 1e-8))
    
    grad = np.where(is_wrong_direction, grad * direction_penalty, grad)
    hess = np.where(is_wrong_direction, hess * direction_penalty, hess)
    
    return grad, hess


def volatility_underestimate_loss(y_pred, dtrain, under_penalty=2.5):
    """波动率预测：低估风险时放大惩罚"""
    y_true = dtrain.get_label()
    residual = y_pred - y_true
    is_under_estimation = residual < 0
    
    grad = 2 * residual
    hess = np.full_like(y_true, 2.0)
    
    grad = np.where(is_under_estimation, grad * under_penalty, grad)
    hess = np.where(is_under_estimation, hess * under_penalty, hess)
    
    return grad, hess


def direction_cost_sensitive_loss(y_pred, dtrain, false_negative_cost=5.0):
    """方向预测：假阴性（漏涨）代价更高"""
    y_true = dtrain.get_label()
    proba = 1.0 / (1.0 + np.exp(-y_pred))
    
    grad = proba - y_true
    hess = proba * (1.0 - proba) + 1e-8
    
    is_false_negative = (y_true == 1) & (proba < 0.5)
    
    grad = np.where(is_false_negative, grad * false_negative_cost, grad)
    hess = np.where(is_false_negative, hess * false_negative_cost, hess)
    
    return grad, hess


# ==================== 模型配置（底层Booster对象） ====================

# 改用xgb.Booster对象存储模型
model_direction = None  # xgb.Booster
model_volatility = None  # xgb.Booster
model_return = None  # xgb.Booster

BEST_PARAMS_DIRECTION = {
    'max_depth': 5, 'learning_rate': 0.235, 'subsample': 0.8387,
    'colsample_bytree': 0.7783, 'reg_alpha': 0.9997, 'reg_lambda': 4.5925,
    'gamma': 0.8667, 'random_state': 42, 'tree_method': 'hist'
}

BEST_PARAMS_VOLATILITY = {
    'max_depth': 9, 'learning_rate': 0.0216, 'subsample': 0.8888,
    'colsample_bytree': 0.9754, 'reg_alpha': 0.0078, 'reg_lambda': 9.9221,
    'gamma': 0.9235, 'random_state': 42, 'tree_method': 'hist'
}

BEST_PARAMS_RETURN = {
    'max_depth': 5, 'learning_rate': 0.1909, 'subsample': 0.8943,
    'colsample_bytree': 0.9876, 'reg_alpha': 0.0241, 'reg_lambda': 9.4085,
    'gamma': 0.9037, 'random_state': 42, 'tree_method': 'hist'
}

# ===== 特征列表（从原始文件完整复制） =====

SELECTED_FEATURES_DIRECTION = ['V9', 'lagged_market_forward_excess_returns', 'V3', 'M10', 'P9', 'P5', 'E18', 
                              'P10', 'M18', 'P7', 'V6', 'I4', 'P13', 'S6', 'M17', 'S7', 'P8', 'V5', 'M9', 
                              'S1', 'P6', 'M12', 'V10', 'M8', 'S9', 'S5', 'M5', 'M4', 'I3', 'I6', 'S2', 
                              'P12', 'M16', 'M13', 'P3', 'E4', 'I2', 'P4', 'S11', 'V7', 'V12', 'V13', 
                              'V2', 'S12', 'V1', 'S10', 'P11', 'M3', 'E2', 'M15', 'I8', 'P1', 'M2', 
                              'E9', 'E3', 'M11', 'I1', 'E7', 'E20', 'E15', 'E14', 'E6', 'E16', 
                              'lagged_forward_returns', 'I9', 'V11', 'V8', 'M14', 'V4', 'M6', 'E5', 
                              'E8', 'M1', 'S8', 'I7', 'S4', 'E19', 'E13', 'P2', 'lagged_risk_free_rate', 
                              'S3', 'E12', 'I5', 'E11', 'D5', 'M7', 'E10', 'D7', 'E1', 'D9']

SELECTED_FEATURES_VOLATILITY = ['V7', 'E19', 'M11', 'I6', 'I5', 'P11', 'P6', 'I9', 'I7', 'V10', 'S4', 
                              'M18', 'V1', 'V11', 'M1', 'I3', 'V3', 'lagged_market_forward_excess_returns', 
                              'M10', 'E7', 'lagged_forward_returns', 'V13', 'M12', 'V5', 'M17', 'M2', 
                              'E4', 'M3', 'V9', 'M13']

SELECTED_FEATURES_RETURN = ['I9', 'P5', 'P9', 'V9', 'V13', 'P2', 'lagged_forward_returns', 'V10', 'S1', 
                          'V5', 'M14', 'E3', 'M9', 'M2', 'V2', 'P8', 'S2', 'S6', 'E2', 'V12', 'S10', 
                          'I8', 'S7', 'V7', 'M4', 'P13', 'P6', 'E19', 'I4', 'S5', 'S9', 'M11', 'S3', 
                          'I1', 'V1', 'E14', 'P4', 'S8', 'S4', 'M10', 'V6', 'M13', 'lagged_risk_free_rate', 
                          'E17', 'I7', 'lagged_market_forward_excess_returns', 'M1', 'M12', 'I2', 'M3', 
                          'M6', 'I3', 'S11', 'S12', 'P7', 'P12', 'P10', 'E7', 'M7', 'E16', 'P11', 'I6', 
                          'D7', 'M8', 'M17', 'M18', 'M16', 'V3', 'V8', 'E8', 'E4', 'M15', 'P3', 'V4', 
                          'E5', 'P1', 'E10', 'V11', 'E9', 'M5', 'I5', 'E18', 'E20', 'E12', 'E13']

OPTIMAL_ALPHA = 0.003001
OPTIMAL_BETA = 0.065005

# ==================== 数据预处理 ====================

def preprocess_data(train_df):
    """添加滞后特征"""
    train_df['lagged_forward_returns'] = train_df['forward_returns'].shift(1)
    train_df['lagged_risk_free_rate'] = train_df['risk_free_rate'].shift(1)
    train_df['lagged_market_forward_excess_returns'] = train_df['market_forward_excess_returns'].shift(1)
    train_df = train_df.dropna().reset_index(drop=True)
    return train_df

def prepare_features(test_df, feature_list):
    """准备特征矩阵"""
    if 'lagged_forward_returns' not in test_df.columns:
        test_df = test_df.with_columns([
            pl.lit(0.0).alias('lagged_forward_returns'),
            pl.lit(0.0).alias('lagged_risk_free_rate'),
            pl.lit(0.0).alias('lagged_market_forward_excess_returns')
        ])
    
    available_features = [col for col in feature_list if col in test_df.columns]
    features = test_df.select(available_features).to_numpy()
    features = np.nan_to_num(features, nan=0.0)
    return features

# ==================== 模型训练（核心修复） ====================

def train_models():
    """使用xgb.train()训练，支持自定义损失"""
    global model_direction, model_volatility, model_return
    
    print("正在使用自定义损失函数训练XGBoost模型...")
    
    # 读取训练数据
    train_path = '/kaggle/input/hull-tactical-market-prediction/train.csv'
    if not os.path.exists(train_path):
        train_path = '../../train.csv'
    
    train_df = pd.read_csv(train_path)
    train_df = preprocess_data(train_df)
    
    # ============= 方向预测模型 =============
    print("训练方向预测模型...")
    X_direction = train_df[SELECTED_FEATURES_DIRECTION].values
    y_direction = (train_df['forward_returns'] > 0).astype(int).values
    
    dtrain_direction = xgb.DMatrix(X_direction, label=y_direction)
    model_direction = xgb.train(
        BEST_PARAMS_DIRECTION,
        dtrain_direction,
        num_boost_round=1613,
        obj=lambda y_pred, dtrain: direction_cost_sensitive_loss(y_pred, dtrain, false_negative_cost=0.2)
    )
    
    # ============= 波动率预测模型 =============
    print("训练波动率预测模型...")
    X_volatility = train_df[SELECTED_FEATURES_VOLATILITY].values
    y_volatility = np.abs(train_df['forward_returns'].values)
    
    dtrain_volatility = xgb.DMatrix(X_volatility, label=y_volatility)
    model_volatility = xgb.train(
        BEST_PARAMS_VOLATILITY,
        dtrain_volatility,
        num_boost_round=371,
        obj=lambda y_pred, dtrain: volatility_underestimate_loss(y_pred, dtrain, under_penalty=0.2)
    )
    
    # ============= 收益率预测模型 =============
    print("训练收益率预测模型...")
    X_return = train_df[SELECTED_FEATURES_RETURN].values
    y_return = train_df['forward_returns'].values
    
    dtrain_return = xgb.DMatrix(X_return, label=y_return)
    model_return = xgb.train(
        BEST_PARAMS_RETURN,
        dtrain_return,
        num_boost_round=137,
        obj=lambda y_pred, dtrain: asymmetric_return_loss(y_pred, dtrain, direction_penalty=0.1, huber_delta=0.02)
    )
    
    print("所有模型训练完成")

# ==================== 预测函数 ====================

def determine_investment_ratio(return_prediction, volatility_prediction, up_probability):
    """投资比例计算"""
    if not (0 <= up_probability <= 1):
        up_probability = max(0.0, min(1.0, up_probability))
    
    if volatility_prediction <= 0:
        volatility_prediction = 1e-6
    
    if return_prediction <= 0 and up_probability <= 0.5:
        return 0.0

    elif up_probability > 0.5 and return_prediction < 0:
        return 0.08
    
    try:
        ratio = OPTIMAL_BETA / return_prediction
        ratio = max(1e-10, min(1e6, ratio))
        
        log_term = np.log(1 + ratio)
        investment_ratio = OPTIMAL_ALPHA * up_probability * log_term / volatility_prediction
        investment_ratio = max(0.0, min(2.0, investment_ratio))
        
        return round(investment_ratio, 4)
    except Exception as e:
        print(f"警告: 投资比例计算错误: {e}，返回保守值0.1")
        return 0.1

def predict(test: pl.DataFrame) -> float:
    """主预测函数 - 使用Booster.predict()"""
    global model_direction, model_volatility, model_return
    
    if any(m is None for m in [model_direction, model_volatility, model_return]):
        train_models()
    
    features_direction = prepare_features(test, SELECTED_FEATURES_DIRECTION)
    features_volatility = prepare_features(test, SELECTED_FEATURES_VOLATILITY)
    features_return = prepare_features(test, SELECTED_FEATURES_RETURN)
    
    # 取最新样本
    last_features_direction = features_direction[-1:]
    last_features_volatility = features_volatility[-1:]
    last_features_return = features_return[-1:]
    
    
    # 方向模型：Booster返回logit，需手动转概率
    dtest_direction = xgb.DMatrix(last_features_direction)
    up_logit = model_direction.predict(dtest_direction)[0]
    up_probability = 1.0 / (1.0 + np.exp(-up_logit))
    
    # 波动率模型：直接预测
    dtest_volatility = xgb.DMatrix(last_features_volatility)
    volatility_prediction = model_volatility.predict(dtest_volatility)[0]
    
    # 收益率模型：直接预测
    dtest_return = xgb.DMatrix(last_features_return)
    return_prediction = model_return.predict(dtest_return)[0]
    
    # ============= 计算投资比例 =============
    investment_ratio = determine_investment_ratio(
        return_prediction=return_prediction,
        volatility_prediction=volatility_prediction,
        up_probability=up_probability
    )
    
    return investment_ratio

# ==================== 推理服务器 ====================

inference_server = kaggle_evaluation.default_inference_server.DefaultInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    data_dir = '/kaggle/input/hull-tactical-market-prediction/'
    if not os.path.exists(data_dir):
        data_dir = '../../'
    inference_server.run_local_gateway((data_dir,))