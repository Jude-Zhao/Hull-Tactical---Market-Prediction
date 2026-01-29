# Kaggle Competition: Hull Tactical - Market Prediction —— Using XGBoost for Market Prediction

## Project Overview
Using XGBoost model to predict market data and generate investment decision recommendations.

## How to Determine Investment Proportions?

According to the definition of the score function, a penalty is applied when the strategy volatility exceeds 1.2 times the index volatility and the strategy return is less than the index return.

After obtaining the predicted returns for the previous 180 days and maximizing the score function as the objective, we obtained the following image. The horizontal axis represents the position ratio, and the vertical axis represents the daily return rate.

![image.png](image.png)

From the figure, we can see that we need to fit a curve similar to an inverse function, and when the return rate is zero, the position ratio is also 0.

Single models have relatively low accuracy in directly predicting daily return rates. We combined predictions from multiple models to enhance accuracy and improve generalization capability.

The final position ratio function is:

$$
w = \begin{cases}
0, & r_{\text{pred}} \le 0 \text{ and } p_{\text{up}} \le 0.5 \\
0.08, & r_{\text{pred}} < 0 \text{ and } p_{\text{up}} > 0.5 \\
\displaystyle
\alpha \cdot p_{\text{up}} \cdot \frac{\ln\left(1 + \dfrac{\beta}{r_{\text{pred}}}\right)}{\sigma}, & r_{\text{pred}} > 0
\end{cases}
$$

Where:
- $r_{\text{pred}}$: Model's predicted value for daily return rate
- $p_{\text{up}}$: Model's predicted probability for "upward" direction (0–1)
- $\sigma$: Predicted volatility of the corresponding asset, used for risk adjustment
- $\alpha$: Global leverage coefficient, controlling overall position limits
- $\beta$: Sensitivity parameter, determining the steepness of the curve when return rate approaches zero

Position control is implemented for three scenarios:
1. When predicted return rate is negative and upward probability is low, maintain no positions
2. When predicted return rate is negative but upward probability is high, maintain 0.08 positions as a conservative strategy
3. When predicted return rate is positive, calculate optimal position ratio based on return rate, upward probability, and volatility

Through joint adjustment of $\alpha$ and $\beta$, more refined position control can be achieved in the "return-risk" tradeoff.

## Core Files
- `main.py`: Main program, including data loading, feature engineering, model training and evaluation
- `XGBoost_predict_submission.py`: Generate final prediction result file (official submission)
- `stacking.py`: Model fusion strategy script
- `script.ipynb`: Experimental script

## Project Approach
1. Load training data, perform preprocessing and missing value handling
2. Construct lag features and technical indicators as model inputs
3. Use XGBoost to train multiple sub-models (return rate, volatility, direction)
4. Combine prediction results to calculate optimal investment proportions
5. Generate submission files that meet requirements

## Usage
1. Run `main.py` for model training and validation
2. Execute `XGBoost_predict_submission.py` to generate prediction results
3. View visualized analysis through `script.ipynb`

## Dependencies
- Python 3.x
- XGBoost
- Pandas, NumPy
- Scikit-learn

---

# Kaggle比赛： Hull Tactical - Market Prediction —— 使用XGBoost市场预测

## 项目概述
使用XGBoost模型对市场数据进行预测，生成投资决策建议。

## 如何决定投资比例？

根据score函数的定义，当策略波动率大于指数波动率的1.2倍以及策略收益小于指数收益的时候，会施加分数的惩罚。

当我们获取前180天的预测收益，以最大化score函数为目标后，我们得到了下面的图像。横坐标是持仓比例、纵坐标是当日收益率。

![image.png](image.png)

从图中我们可以看到，需要拟合一个类似反函数的曲线，并且当收益率为零时，持仓比率为0。

单个模型直接预测当日收益率的准确性较低，我们结合了多个模型的预测结果，来强化模型的准确性，提高其泛化能力。

最终的持仓比例函数为：

$$
w = \begin{cases}
0, & r_{\text{pred}} \le 0 \text{ 且 } p_{\text{up}} \le 0.5 \\
0.08, & r_{\text{pred}} < 0 \text{ 且 } p_{\text{up}} > 0.5 \\
\displaystyle
\alpha \cdot p_{\text{up}} \cdot \frac{\ln\left(1 + \dfrac{\beta}{r_{\text{pred}}}\right)}{\sigma}, & r_{\text{pred}} > 0
\end{cases}
$$

其中  
- $r_{\text{pred}}$：模型对当日收益率的预测值  
- $p_{\text{up}}$：模型预测"上涨"方向的概率（0–1）  
- $\sigma$：对应资产的预测波动率，用于风险调整  
- $\alpha$：全局杠杆系数，控制整体仓位上限
- $\beta$：灵敏度参数，决定曲线在收益率接近零时的陡峭程度

实现了三种情况的仓位控制：
1. 当预测收益率为负且上涨概率较低时，完全不持仓
2. 当预测收益率为负但上涨概率较高时，持仓0.08作为保守策略
3. 当预测收益率为正时，根据收益率、上涨概率和波动率计算最优持仓比例

通过联合调节 $\alpha$ 与 $\beta$，可在"收益–风险"权衡中实现更精细的仓位控制。

## 核心文件
- `main.py`: 主程序，包含数据加载、特征工程、模型训练和评估
- `XGBoost_predict_submission.py`: 生成最终预测结果文件（正式提交）
- `stacking.py`: 模型融合策略脚本
- `script.ipynb`: 实验脚本

## 项目思路
1. 加载训练数据，进行预处理和缺失值处理
2. 构建滞后特征和技术指标作为模型输入
3. 使用XGBoost训练多个子模型（收益率、波动率、方向）
4. 结合预测结果计算最优投资比例
5. 生成符合要求的提交文件

## 使用方法
1. 运行`main.py`进行模型训练和验证
2. 执行`XGBoost_predict_submission.py`生成预测结果
3. 通过`script.ipynb`查看可视化分析

## 依赖
- Python 3.x
- XGBoost
- Pandas, NumPy
- Scikit-learn