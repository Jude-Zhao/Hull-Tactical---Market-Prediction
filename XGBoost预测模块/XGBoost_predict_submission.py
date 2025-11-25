"""
XGBoost预测模块 - 提交脚本
根据实验结果优化的多模型预测系统
"""

import os
import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
import kaggle_evaluation.default_inference_server

# 全局变量存储三个模型和相关参数
model_direction = None  # 预测涨跌方向
model_volatility = None  # 预测波动率
model_return = None     # 预测具体收益率

# 最优超参数配置（来自实验结果）
BEST_PARAMS_DIRECTION = {
    'n_estimators': 1613,
    'max_depth': 5,
    'learning_rate': 0.235008845080467,
    'subsample': 0.8387400631785948,
    'colsample_bytree': 0.7783331011414365,
    'reg_alpha': 0.9997491581800291,
    'reg_lambda': 4.592488919658672,
    'gamma': 0.8667417222278044,
    'random_state': 42,
    'objective': 'binary:logistic',
    'tree_method': 'hist'
}

BEST_PARAMS_VOLATILITY = {
    'n_estimators': 371,
    'max_depth': 9,
    'learning_rate': 0.02164141581299458,
    'subsample': 0.88879950890673,
    'colsample_bytree': 0.9754210836063002,
    'reg_alpha': 0.007787658410143285,
    'reg_lambda': 9.922115592912178,
    'gamma': 0.9234963019255433,
    'random_state': 42,
    'objective': 'reg:squarederror',
    'tree_method': 'hist'
}

BEST_PARAMS_RETURN = {
    'n_estimators': 137,
    'max_depth': 5,
    'learning_rate': 0.19089320613723337,
    'subsample': 0.8942521689143974,
    'colsample_bytree': 0.987551107426109,
    'reg_alpha': 0.0241272429288264,
    'reg_lambda': 9.408535459483122,
    'gamma': 0.9037288822106474,
    'random_state': 42,
    'objective': 'reg:squarederror',
    'tree_method': 'hist'
}

# 最优特征选择（来自实验结果）
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

# 最优持仓超参数
OPTIMAL_ALPHA = 0.003001
OPTIMAL_BETA = 0.065005

def preprocess_data(train_df):
    """数据预处理函数 - 按照实验要求添加滞后特征"""
    # 创建滞后一天的特征
    train_df['lagged_forward_returns'] = train_df['forward_returns'].shift(1)
    train_df['lagged_risk_free_rate'] = train_df['risk_free_rate'].shift(1)
    train_df['lagged_market_forward_excess_returns'] = train_df['market_forward_excess_returns'].shift(1)
    
    # 处理缺失值
    train_df = train_df.dropna().reset_index(drop=True)
    
    return train_df

def prepare_features(test_df, feature_list):
    """准备特征数据，只使用指定的特征列"""
    # 确保所有必需的滞后特征存在
    if 'lagged_forward_returns' not in test_df.columns:
        # 对于初始测试数据，设置默认值
        test_df = test_df.with_columns([
            pl.lit(0.0).alias('lagged_forward_returns'),
            pl.lit(0.0).alias('lagged_risk_free_rate'),
            pl.lit(0.0).alias('lagged_market_forward_excess_returns')
        ])
    
    # 获取实际存在的特征列
    available_features = [col for col in feature_list if col in test_df.columns]
    
    # 提取特征
    features = test_df.select(available_features).to_numpy()
    
    # 处理缺失值
    features = np.nan_to_num(features, nan=0.0)
    
    return features

def train_models():
    """训练三个XGBoost模型"""
    global model_direction, model_volatility, model_return
    
    print("正在训练三个XGBoost模型...")
    
    # 读取训练数据
    train_path = '/kaggle/input/hull-tactical-market-prediction/train.csv'
    if not os.path.exists(train_path):
        # 本地测试路径
        train_path = '../../train.csv'
    
    train_df = pd.read_csv(train_path)
    
    # 数据预处理
    train_df = preprocess_data(train_df)
    
    # 训练方向预测模型（上涨概率）
    print("训练方向预测模型...")
    X_direction = train_df[SELECTED_FEATURES_DIRECTION].values
    y_direction = (train_df['forward_returns'] > 0).astype(int).values
    model_direction = xgb.XGBClassifier(**BEST_PARAMS_DIRECTION)
    model_direction.fit(X_direction, y_direction)
    
    # 训练波动率预测模型
    print("训练波动率预测模型...")
    X_volatility = train_df[SELECTED_FEATURES_VOLATILITY].values
    y_volatility = np.abs(train_df['forward_returns'].values)
    model_volatility = xgb.XGBRegressor(**BEST_PARAMS_VOLATILITY)
    model_volatility.fit(X_volatility, y_volatility)
    
    # 训练收益率预测模型
    print("训练收益率预测模型...")
    X_return = train_df[SELECTED_FEATURES_RETURN].values
    y_return = train_df['forward_returns'].values
    model_return = xgb.XGBRegressor(**BEST_PARAMS_RETURN)
    model_return.fit(X_return, y_return)
    
    print("所有模型训练完成")

def determine_investment_ratio(return_prediction, volatility_prediction, up_probability):
    """根据多维预测信息确定投资比例的分段非线性函数"""
    # 输入参数有效性校验
    if not (0 <= up_probability <= 1):
        up_probability = max(0.0, min(1.0, up_probability))
    
    if volatility_prediction <= 0:
        volatility_prediction = 1e-6
    
    # 分段函数实现
    if return_prediction <= 0:
        return 0.0
    else:
        try:
            # 计算对数项: ln(1 + β / r_pred)
            ratio = OPTIMAL_BETA / return_prediction
            
            # 数值稳定性检查
            if ratio > 1e6:
                ratio = 1e6
            elif ratio < 1e-10:
                ratio = 1e-10
                
            log_term = np.log(1 + ratio)
            
            # 计算最终投资比例: α · p_up · log_term / σ
            investment_ratio = OPTIMAL_ALPHA * up_probability * log_term / volatility_prediction
            
            # 确保投资比例在合理范围内 [0, 2]
            investment_ratio = max(0.0, min(2.0, investment_ratio))
            
            # 保留4位小数
            return round(investment_ratio, 4)
            
        except Exception as e:
            print(f"警告: 投资比例计算出现数值错误: {e}，返回保守值0.1")
            return 0.1

def predict(test: pl.DataFrame) -> float:
    """预测函数 - 使用三个模型进行综合预测"""
    global model_direction, model_volatility, model_return
    
    # 第一次调用时训练模型
    if model_direction is None or model_volatility is None or model_return is None:
        train_models()
    
    # 准备各模型的特征
    features_direction = prepare_features(test, SELECTED_FEATURES_DIRECTION)
    features_volatility = prepare_features(test, SELECTED_FEATURES_VOLATILITY)
    features_return = prepare_features(test, SELECTED_FEATURES_RETURN)
    
    # 使用最后一个样本进行预测
    last_features_direction = features_direction[-1:]
    last_features_volatility = features_volatility[-1:]
    last_features_return = features_return[-1:]
    
    # 三个模型的预测
    up_probability = model_direction.predict_proba(last_features_direction)[0, 1]
    volatility_prediction = model_volatility.predict(last_features_volatility)[0]
    return_prediction = model_return.predict(last_features_return)[0]
    
    # 计算投资比例
    investment_ratio = determine_investment_ratio(
        return_prediction=return_prediction,
        volatility_prediction=volatility_prediction,
        up_probability=up_probability
    )
    
    return investment_ratio

# 初始化推理服务器
inference_server = kaggle_evaluation.default_inference_server.DefaultInferenceServer(predict)

# 根据运行环境决定服务模式
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    # 本地测试时使用的路径
    data_dir = '/kaggle/input/hull-tactical-market-prediction/'
    if not os.path.exists(data_dir):
        data_dir = '../../'
    inference_server.run_local_gateway((data_dir,))