
import warnings
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score

matplotlib.rcParams['font.sans-serif']=['SimHei']
matplotlib.rcParams['axes.unicode_minus']=False
warnings.filterwarnings('ignore')

# 设置随机种子以确保结果可重现
np.random.seed(42)

# 读取数据
df = pd.read_excel('addr.xlsx', sheet_name='男胎检测数据')

# 数据预处理 - 转换孕周为数值型
def convert_gestational_week(week_str):
    if isinstance(week_str, str):
        if 'w' in week_str:
            parts = week_str.split('w')
            weeks = int(parts[0])
            days = 0
            if '+' in parts[1]:
                days = int(parts[1].split('+')[1])
            return weeks + days/7.0
    return np.nan

df['检测孕周数值'] = df['检测孕周'].apply(convert_gestational_week)

# 随机选择15个样本
sample_size = 20
random_sample = df.sample(n=sample_size, random_state=42)



# 读取数据
df = pd.read_excel('addr.xlsx', sheet_name='男胎检测数据')

# 数据预处理
# 转换孕周为数值型
def convert_gestational_week(week_str):
    if isinstance(week_str, str):
        if 'w' in week_str:
            parts = week_str.split('w')
            weeks = int(parts[0])
            days = 0
            if '+' in parts[1]:
                days = int(parts[1].split('+')[1])
            return weeks + days/7.0
    return np.nan

df['检测孕周数值'] = df['检测孕周'].apply(convert_gestational_week)

# 选择特征和目标变量
features = ['检测孕周数值', '孕妇BMI', '体重', '年龄']
X = random_sample[features]
y = random_sample['Y染色体浓度']

# 处理缺失值
X.fillna(X.mean(), inplace=True)
y.fillna(y.mean(), inplace=True)

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 准备不同回归模型
models = {
    '多元线性回归': LinearRegression(),
    '二次多项式回归': LinearRegression(),
    '岭回归 (L2正则化)': Ridge(alpha=1.0),
    'Lasso回归 (L1正则化)': Lasso(alpha=0.001),
    '弹性网络回归': ElasticNet(alpha=0.001, l1_ratio=0.05),
    '随机森林回归': RandomForestRegressor(n_estimators=100, random_state=42),
    '神经网络回归': MLPRegressor(hidden_layer_sizes=(23, 23), activation='relu', 
                               solver='adam', alpha=0.01, batch_size='auto', 
                               learning_rate='constant', learning_rate_init=0.01, 
                               max_iter=500, random_state=42)
}

# 为多项式回归准备特征
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_scaled)

# 训练模型并评估
results = {}
for name, model in models.items():
    if name == '二次多项式回归':
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)
    else:
        model.fit(X_scaled, y)
        y_pred = model.predict(X_scaled)
    
    # 计算评估指标
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    
    results[name] = {
        'model': model,
        'R²': r2,
        'MSE': mse,
        'predictions': y_pred
    }

# 打印所有模型性能
print("模型性能比较:")
print("=" * 50)
for name, metrics in results.items():
    print(f"{name}: R² = {metrics['R²']:.4f}, MSE = {metrics['MSE']:.4f}")
print()

# 创建子图展示不同模型的拟合效果
fig, axes = plt.subplots(2, 4, figsize=(15, 12))
axes = axes.ravel()

# 为每个模型创建散点图
for i, (name, metrics) in enumerate(results.items()):
    # 按检测孕周排序以便绘制平滑曲线
    sorted_idx = np.argsort(random_sample['检测孕周数值'].values)
    x_sorted = range(len(sorted_idx))
    y_true_sorted = y.values[sorted_idx]
    y_pred_sorted = metrics['predictions'][sorted_idx]
    
    axes[i].scatter(x_sorted, y_true_sorted, color='blue', alpha=0.7, label='实际值')
    axes[i].plot(x_sorted, y_pred_sorted, color='red', linewidth=2, label='预测值')
    axes[i].set_title(f'{name} (R$^{2}$ = {metrics["R²"]:.4f})')
    axes[i].set_xlabel('样本索引（按检测孕周排序）')
    axes[i].set_ylabel('Y染色体浓度')
    axes[i].legend()
    axes[i].grid(True, linestyle='--', alpha=0.7)
    axes[7].set_visible(False)

plt.tight_layout()
plt.savefig('regression/model_comparison.png', dpi=300)
plt.show()

# 创建实际值与预测值对比图
plt.figure(figsize=(10, 8))
colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'cyan', 'magenta']
markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']

for i, (name, metrics) in enumerate(results.items()):
    plt.scatter(y, metrics['predictions'], color=colors[i], marker=markers[i], 
                alpha=0.7, label=name)

# 绘制理想拟合线
min_val = min(min(y), min([min(metrics['predictions']) for metrics in results.values()]))
max_val = max(max(y), max([max(metrics['predictions']) for metrics in results.values()]))
plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='理想拟合')

plt.xlabel('实际Y染色体浓度')
plt.ylabel('预测Y染色体浓度')
plt.title('不同模型预测效果对比')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('regression/prediction_comparison.png', dpi=300)
plt.show()

# 绘制残差图
plt.figure(figsize=(10, 8))
for i, (name, metrics) in enumerate(results.items()):
    residuals = y - metrics['predictions']
    plt.scatter(metrics['predictions'], residuals, color=colors[i], 
                marker=markers[i], alpha=0.7, label=name)

plt.axhline(y=0, color='k', linestyle='--')
plt.xlabel('预测值')
plt.ylabel('残差')
plt.title('不同模型的残差分析')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('regression/residual_comparison.png', dpi=300)
plt.show()