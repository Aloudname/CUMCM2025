
import warnings
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score


matplotlib.rcParams['font.sans-serif']=['SimHei']
matplotlib.rcParams['axes.unicode_minus']=False
warnings.filterwarnings('ignore')

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

# 选择可能相关的数值型自变量
potential_features = [
    '年龄', '身高', '体重', '检测孕周数值', '孕妇BMI', '原始读段数', 
    '在参考基因组上比对的比例', '重复读段的比例', '唯一比对的读段数', 
    'GC含量', '13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值', 
    'X染色体的Z值', 'Y染色体的Z值', 'X染色体浓度', '13号染色体的GC含量', 
    '18号染色体的GC含量', '21号染色体的GC含量', '被过滤掉读段数的比例', 
    '生产次数'
]

# 创建特征数据集和目标变量
X = df[potential_features].copy()
y = df['Y染色体浓度'].copy()

# 处理缺失值
X.fillna(X.mean(), inplace=True)
y.fillna(y.mean(), inplace=True)

# 计算相关系数
correlation_matrix = X.corrwith(y)
correlation_matrix = correlation_matrix.sort_values(ascending=False)

print("变量与Y染色体浓度的相关系数:")
print(correlation_matrix)

# 选择相关性最高的几个特征
top_features = correlation_matrix.abs().sort_values(ascending=False).head(8).index.tolist()
print(f"\n选择的相关性最高的特征: {top_features}")

# 准备建模数据
X_selected = X[top_features]
X_scaled = StandardScaler().fit_transform(X_selected)
y_log = np.log1p(y)  # 对目标变量进行对数变换，使其更接近正态分布

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_log, test_size=0.2, random_state=42
)

# 初始化多种回归模型
models = {
    '线性回归': LinearRegression(),
    '岭回归': Ridge(alpha=1.0),
    'Lasso回归': Lasso(alpha=0.1),
    '随机森林': RandomForestRegressor(n_estimators=100, random_state=42),
    '支持向量机': SVR(kernel='rbf', C=1.0, gamma='scale')
}

# 训练和评估模型
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # 计算评估指标
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    # 交叉验证
    cv_scores = cross_val_score(model, X_scaled, y_log, cv=5, scoring='r2')
    
    results[name] = {
        'R²': r2,
        'MSE': mse,
        'CV R²均值': cv_scores.mean(),
        'CV R²标准差': cv_scores.std()
    }

# 打印模型性能比较
print("\n模型性能比较:")
print("=" * 50)
for name, metrics in results.items():
    print(f"{name}:")
    print(f"  R² = {metrics['R²']:.4f}, MSE = {metrics['MSE']:.4f}")
    print(f"  CV R² = {metrics['CV R²均值']:.4f} (±{metrics['CV R²标准差']:.4f})")
    print()

# 绘制相关性热图
plt.figure(figsize=(12, 10))
corr_matrix = pd.concat([X[top_features], y], axis=1).corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('特征与Y染色体浓度的相关性热图')
plt.tight_layout()
plt.savefig('./correlation/correlation_heatmap.png', dpi=300)
plt.show()

# 绘制特征重要性（针对随机森林）
rf_model = models['随机森林']
feature_importance = pd.DataFrame({
    'feature': top_features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('特征重要性')
plt.title('随机森林特征重要性排序')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('./correlation/feature_importance.png', dpi=300)
plt.show()

# 输出最相关的几个特征及其相关系数
print("最相关的特征及其与Y染色体浓度的相关系数:")
for feature in top_features:
    corr = correlation_matrix[feature]
    print(f"{feature}: {corr:.4f}")

# 进一步分析这些特征的统计显著性
features = ['检测孕周数值', '孕妇BMI', '体重', '年龄']

# 使用statsmodels进行多元线性回归并获取p值
X_with_const = sm.add_constant(X_scaled)
model_sm = sm.OLS(y, X_with_const).fit()
p_values = model_sm.pvalues[1:]

print("多元线性回归系数和p值:")
for i, feature in enumerate(features):
    print(f"{feature}: coeff={model_sm.params[i+1]:.4f}, p-value={p_values[i]:.4f}")
print(f"模型整体p值: {model_sm.f_pvalue:.4f}")
print()