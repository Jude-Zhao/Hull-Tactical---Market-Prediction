import numpy as np
import pandas as pd
import polars as pl
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

# 导入现有的预测模块以便复用核心功能
from XGBoost_predict_submission import prepare_features, determine_investment_ratio


def load_and_preprocess_data(train_path, skip_rows=1007):
    """
    数据集切分与数据预处理的函数
    
    参数:
        train_path (str): 训练数据的文件路径
        skip_rows (int): 需要跳过的前面几行数据（因为可能有缺失值）
    
    返回:
        tuple: (X_train, X_test, y_train, y_test, feature_columns, scaler)
            X_train: 训练集特征
            X_test: 测试集特征
            y_train: 训练集目标值
            y_test: 测试集目标值
            feature_columns: 特征列名称列表
            scaler: 用于标准化的StandardScaler对象
    """
    print(f"正在加载数据: {train_path}")
    
    # 读取训练数据并跳过前N行有缺失的数据
    try:
        train_df = pd.read_csv(train_path)
        print(f"原始数据形状: {train_df.shape}")
        
        # 跳过前N行
        train_df = train_df.iloc[skip_rows:].reset_index(drop=True)
        print(f"跳过前{skip_rows}行后的数据形状: {train_df.shape}")
        
        # 定义需要排除的列
        exclude_cols = ['date_id', 'forward_returns', 'risk_free_rate', 'market_forward_excess_returns']
        feature_columns = [col for col in train_df.columns if col not in exclude_cols]
        
        # 准备特征和目标
        X = train_df[feature_columns].values
        y = train_df['forward_returns'].values
        
        # 处理缺失值
        X = np.nan_to_num(X, nan=0.0)
        
        # 标准化特征
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 时间序列划分：80%训练，20%测试
        train_size = int(0.8 * len(X_scaled))
        X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        print(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
        print(f"特征维度: {X_scaled.shape[1]}")
        
        return X_train, X_test, y_train, y_test, feature_columns, scaler
        
    except Exception as e:
        print(f"数据加载和预处理出错: {e}")
        raise


def feature_selection(X_train, X_test, y_train, feature_columns, top_n=20):
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
        max_depth=6,
        learning_rate=0.1,
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
    plt.close()
    
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
    
    # 默认参数
    default_params = {
        'n_estimators': 1000,
        'max_depth': 6,
        'learning_rate': 0.01,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
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


def hyperparameter_tuning(X_train, y_train):
    """
    超参数筛选的函数，使用GridSearchCV进行参数调优
    
    参数:
        X_train: 训练集特征
        y_train: 训练集目标值
    
    返回:
        best_params: 最佳超参数组合
    """
    print("正在进行超参数筛选...")
    
    # 定义参数网格
    param_grid = {
        'n_estimators': [500, 1000],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'reg_alpha': [0.01, 0.1, 1.0],
        'reg_lambda': [0.01, 0.1, 1.0]
    }
    
    # 创建基础模型
    base_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        tree_method='hist',
        random_state=42
    )
    
    # 时间序列交叉验证
    tscv = TimeSeriesSplit(n_splits=3)
    
    # GridSearchCV
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_squared_error',
        verbose=2,
        n_jobs=-1
    )
    
    # 执行网格搜索
    grid_search.fit(X_train, y_train)
    
    # 获取最佳参数
    best_params = grid_search.best_params_
    print(f"超参数筛选完成，最佳参数: {best_params}")
    
    return best_params


def evaluate_model(model, X_test, y_test, scaler=None, selected_features=None):
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
    
    # 计算投资策略的表现
    investment_ratios = np.array([determine_investment_ratio(pred) for pred in y_pred])
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
        X_train, X_test, y_train, y_test, feature_columns, scaler = load_and_preprocess_data(train_path)
        
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
        
        # 5. 评估模型
        metrics = evaluate_model(model, X_test_selected, y_test, scaler, selected_features)
        
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