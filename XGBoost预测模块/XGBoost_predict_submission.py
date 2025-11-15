# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
"""
评估API要求您设置一个服务器来响应推理请求。
我们已经定义了服务器；您只需要编写predict函数。
当我们在隐藏测试集上评估您的提交时，`default_gateway`中定义的客户端将在不同的容器中运行，
直接访问隐藏测试集并逐个时间步传递数据。

您的代码将始终可以访问竞赛文件的已发布副本。
"""

import os
import numpy as np
import pandas as pd
import polars as pl
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

import kaggle_evaluation.default_inference_server

# 全局变量存储模型和预处理器
model = None
scaler = None
feature_columns = None

# XGBoost超参数配置 - 可调节接口
XGBOOST_PARAMS = {
    'n_estimators': 1000,      # 树的数量
    'max_depth': 6,             # 最大深度
    'learning_rate': 0.01,     # 学习率
    'subsample': 0.8,           # 子样本比例
    'colsample_bytree': 0.8,    # 特征采样比例
    'reg_alpha': 0.1,           # L1正则化
    'reg_lambda': 0.1,          # L2正则化
    'random_state': 42,         # 随机种子
    'objective': 'reg:squarederror',  # 回归目标函数
    'tree_method': 'hist',      # 快速训练方法
    'early_stopping_rounds': 50, # 早停轮数
    'eval_metric': 'rmse'       # 评估指标
}

def prepare_features(test_df):
    """准备特征数据"""
    global feature_columns
    
    # 获取特征列（排除date_id和目标相关列）
    if feature_columns is None:
        # 排除不需要的列
        exclude_cols = ['date_id', 'is_scored', 'lagged_forward_returns', 
                       'lagged_risk_free_rate', 'lagged_market_forward_excess_returns']
        feature_columns = [col for col in test_df.columns if col not in exclude_cols]
    
    # 提取特征
    features = test_df.select(feature_columns).to_numpy()
    
    # 处理缺失值
    features = np.nan_to_num(features, nan=0.0)
    
    return features

def train_model():
    """训练XGBoost模型 - 使用可调节超参数"""
    global model, scaler
    
    # 读取训练数据 - 跳过前1007行有缺失的数据，与LSTM保持一致
    train_df = pd.read_csv('/kaggle/input/hull-tactical-market-prediction/train.csv')
    train_df = train_df.iloc[1007:].reset_index(drop=True)  # 从第1008行开始
    
    # 准备特征和目标 - 与LSTM保持一致的特征选择
    exclude_cols = ['date_id', 'forward_returns', 'risk_free_rate', 'market_forward_excess_returns']
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    X = train_df[feature_cols].values
    y = train_df['forward_returns'].values
    
    # 处理缺失值
    X = np.nan_to_num(X, nan=0.0)
    
    # 标准化特征（可选，XGBoost对特征缩放不敏感，但可以提升稳定性）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分训练集和验证集（时间序列划分）
    train_size = int(0.8 * len(X_scaled))
    X_train, X_val = X_scaled[:train_size], X_scaled[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    # 创建XGBoost模型 - 使用可调节的超参数
    model = xgb.XGBRegressor(**XGBOOST_PARAMS)
    
    # 训练模型 - 使用验证集进行早停
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # 输出训练信息
    print(f"XGBoost模型训练完成")
    print(f"训练样本数: {len(X_train)}, 验证样本数: {len(X_val)}")
    print(f"特征维度: {X_scaled.shape[1]}")
    
    # 验证集性能评估
    val_pred = model.predict(X_val)
    val_rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))
    print(f"验证集RMSE: {val_rmse:.6f}")

def determine_investment_ratio(predicted_return):
    """根据预测的收益率决定投资比例 - 反函数策略"""
    # 反函数策略：预测收益越高，投资比例越小
    # 使用 ln(1 + β/r_pred) 的形式，其中β是调节参数
    
    # 设置调节参数β，控制反函数的敏感度
    beta = 0.01  # 可以根据实际效果调整
    
    # 防止除零错误，设置最小预测收益阈值
    min_return = 1e-6
    abs_predicted_return = max(abs(predicted_return), min_return)
    
    # 使用反函数关系：ln(1 + β/|r_pred|)
    # 对于正收益：投资比例随收益增加而减小
    if predicted_return > 0:
        # 正收益：投资比例 = ln(1 + β/r_pred) ，但反向缩放
        ratio = np.log(1 + beta / predicted_return)
        # 将比例映射到 [0, 2] 范围，高正收益对应低投资比例
        ratio = 2.0 - min(ratio * 1, 2.0)  # 乘以100增强敏感度
    else:
        ratio = 0
    
    return max(min(ratio, 2.0), 0.0)  # 确保在 [0, 2] 范围内
    

def predict(test: pl.DataFrame) -> float:
    """用您的推理代码替换此函数。
    您可以返回Pandas或Polars数据框，但推荐使用Polars以获得更好的性能。
    除了第一批外，每批预测必须在提供特征后的5分钟内返回。
    """
    global model, scaler
    
    # 第一次调用时训练模型
    if model is None:
        print("正在训练XGBoost模型...")
        train_model()
    
    # 准备特征
    features = prepare_features(test)
    
    # 标准化特征（如果训练时使用了标准化）
    if scaler is not None:
        features_scaled = scaler.transform(features)
    else:
        features_scaled = features
    
    # 使用最后一个样本进行预测（与LSTM保持一致）
    last_features = features_scaled[-1:]
    
    # 预测
    predicted_return = model.predict(last_features)[0]
    
    # 根据预测收益决定投资比例
    investment_ratio = determine_investment_ratio(predicted_return)
    
    return investment_ratio

# 当您的笔记本在隐藏测试集上运行时，必须在笔记本启动后的15分钟内调用inference_server.serve
# 否则网关将抛出错误。如果您需要超过15分钟来加载模型，可以在第一次
# `predict`调用期间执行，该调用没有通常的1分钟响应期限。
inference_server = kaggle_evaluation.default_inference_server.DefaultInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(('/kaggle/input/hull-tactical-market-prediction/',))