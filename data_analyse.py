import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class InterestCircleAnalysis:
    def __init__(self, data_path):
        """
        初始化分析类
        """
        self.df = pd.read_csv(data_path)
        self.preprocess_data()

    def preprocess_data(self):
        """
        数据预处理
        """
        print("数据基本信息：")
        print(f"样本量：{len(self.df)}")
        print(f"变量数：{len(self.df.columns)}")

        # 创建兴趣圈层虚拟变量
        interest_columns = ['二次元', '游戏', '追星', '体育', '文艺',
                            '科技', '户外', '宠物', '美食', '桌游']

        for interest in interest_columns:
            self.df[f'兴趣_{interest}'] = self.df['主要兴趣圈层'].apply(
                lambda x: 1 if interest in str(x) else 0
            )

        relationship_map = {
            '单身且无恋爱意愿': 0,
            '单身但有恋爱意愿': 1,
            '暧昧/接触中': 2,
            '恋爱中': 3,
            '已婚/长期稳定关系': 4
        }
        self.df['恋爱状态编码'] = self.df['恋爱状态'].map(relationship_map)

        # 创建是否单身变量
        self.df['是否单身'] = self.df['恋爱状态'].apply(
            lambda x: 1 if '单身' in str(x) else 0
        )

        # 创建是否有恋爱意愿变量
        self.df['有恋爱意愿'] = self.df['恋爱状态'].apply(
            lambda x: 1 if '有恋爱意愿' in str(x) or '恋爱' in str(x) or '已婚' in str(x) else 0
        )

    def descriptive_analysis(self):
        """
        描述性统计分析
        """
        print("\n=== 描述性统计分析 ===")

        # 1. 兴趣圈层分布
        interest_cols = [col for col in self.df.columns if col.startswith('兴趣_')]
        interest_counts = self.df[interest_cols].sum().sort_values(ascending=False)

        plt.figure(figsize=(12, 6))
        interest_counts.plot(kind='bar')
        plt.title('各兴趣圈层参与人数分布')
        plt.xlabel('兴趣圈层')
        plt.ylabel('人数')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('interest_distribution.png', dpi=300)
        plt.show()

        print("\n兴趣圈层参与人数TOP5：")
        print(interest_counts.head())

        # 2. 恋爱状态分布
        plt.figure(figsize=(10, 6))
        relationship_dist = self.df['恋爱状态'].value_counts()
        relationship_dist.plot(kind='pie', autopct='%1.1f%%')
        plt.title('恋爱状态分布')
        plt.ylabel('')
        plt.tight_layout()
        plt.savefig('relationship_status.png', dpi=300)
        plt.show()

        # 3. 交叉分析：兴趣圈层 vs 恋爱状态
        print("\n=== 各兴趣圈层恋爱率分析 ===")
        relationship_by_interest = {}

        for interest in interest_cols:
            interest_name = interest.replace('兴趣_', '')
            group_df = self.df[self.df[interest] == 1]
            if len(group_df) > 10:  # 确保有足够样本
                single_rate = group_df['是否单身'].mean()
                in_relationship_rate = 1 - single_rate
                relationship_by_interest[interest_name] = {
                    '样本量': len(group_df),
                    '单身率': single_rate,
                    '恋爱率': in_relationship_rate,
                    '有恋爱意愿比例': group_df['有恋爱意愿'].mean()
                }

        relationship_df = pd.DataFrame(relationship_by_interest).T
        relationship_df = relationship_df.sort_values('恋爱率', ascending=False)

        print(relationship_df)

        # 可视化
        plt.figure(figsize=(14, 8))
        plt.subplot(2, 1, 1)
        relationship_df['恋爱率'].plot(kind='bar')
        plt.title('各兴趣圈层恋爱率对比')
        plt.ylabel('恋爱率')
        plt.xticks(rotation=45)

        plt.subplot(2, 1, 2)
        relationship_df['有恋爱意愿比例'].plot(kind='bar', color='orange')
        plt.title('各兴趣圈层恋爱意愿比例对比')
        plt.ylabel('有恋爱意愿比例')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig('relationship_by_interest.png', dpi=300)
        plt.show()

    def correlation_analysis(self):
        """
        相关性分析
        """
        print("\n=== 相关性分析 ===")

        # 选择相关变量
        analysis_vars = ['年龄', '每周投入时间编码', '社交活跃度编码',
                         '恋爱状态编码', '是否单身', '有恋爱意愿']

        # 编码时间投入
        time_map = {'少于5小时': 1, '5-10小时': 2, '10-20小时': 3, '20小时以上': 4}
        self.df['每周投入时间编码'] = self.df['每周投入时间'].map(time_map)

        # 编码社交活跃度
        activity_map = {'不活跃': 1, '一般': 2, '比较活跃': 3, '非常活跃': 4}
        self.df['社交活跃度编码'] = self.df['社交活跃度'].map(activity_map)

        # 计算相关系数矩阵
        corr_matrix = self.df[analysis_vars].corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                    square=True, linewidths=0.5)
        plt.title('变量相关性热图')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=300)
        plt.show()

        # 兴趣圈层与恋爱状态的卡方检验
        print("\n=== 兴趣圈层与恋爱状态的卡方检验 ===")

        interest_cols = [col for col in self.df.columns if col.startswith('兴趣_')]
        results = []

        for interest in interest_cols:
            if self.df[interest].nunique() > 1:
                contingency_table = pd.crosstab(self.df[interest], self.df['是否单身'])
                chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

                interest_name = interest.replace('兴趣_', '')
                results.append({
                    '兴趣圈层': interest_name,
                    '卡方值': chi2,
                    'P值': p,
                    '是否显著': '是' if p < 0.05 else '否'
                })

        chi2_results = pd.DataFrame(results)
        print(chi2_results.sort_values('P值'))

    def regression_analysis(self):
        """
        回归分析
        """
        print("\n=== 回归分析：兴趣圈层对恋爱状态的影响 ===")

        # 准备回归变量
        interest_cols = [col for col in self.df.columns if col.startswith('兴趣_')]

        # 控制变量
        control_vars = ['年龄', '性别编码', '每周投入时间编码', '社交活跃度编码']

        # 编码性别
        gender_map = {'男': 0, '女': 1, '其他/不愿透露': 2}
        self.df['性别编码'] = self.df['性别'].map(gender_map)

        # 1. 逻辑回归：预测是否单身
        print("\n1. 逻辑回归：预测是否单身")

        # 准备数据
        X_vars = control_vars + interest_cols
        X = self.df[X_vars].fillna(0)
        y = self.df['是否单身']

        # 添加常数项
        X = sm.add_constant(X)

        # 逻辑回归
        logit_model = sm.Logit(y, X)
        logit_result = logit_model.fit(disp=0)

        print(logit_result.summary())

        # 提取显著变量
        coefficients = pd.DataFrame({
            '变量': logit_result.params.index,
            '系数': logit_result.params.values,
            'P值': logit_result.pvalues
        })

        significant_vars = coefficients[coefficients['P值'] < 0.05].sort_values('系数', ascending=False)
        print("\n显著的预测变量（P值<0.05）：")
        print(significant_vars[['变量', '系数', 'P值']])

        # 2. OLS回归：预测恋爱意愿强度
        print("\n2. OLS回归：预测恋爱意愿强度")

        X_vars_ols = control_vars + interest_cols
        X_ols = self.df[X_vars_ols].fillna(0)
        y_ols = self.df['有恋爱意愿']

        X_ols = sm.add_constant(X_ols)
        ols_model = sm.OLS(y_ols, X_ols)
        ols_result = ols_model.fit()

        print(ols_result.summary())

    def advanced_analysis(self):
        """
        进阶分析：聚类分析和交互效应
        """
        print("\n=== 进阶分析 ===")

        # 1. 基于兴趣圈层的聚类分析
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        interest_cols = [col for col in self.df.columns if col.startswith('兴趣_')]
        X_cluster = self.df[interest_cols]

        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_cluster)

        # 使用肘部法则确定最佳K值
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            wcss.append(kmeans.inertia_)

        # 选择K=4进行聚类
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        self.df['兴趣聚类'] = kmeans.fit_predict(X_scaled)

        # 分析不同聚类的恋爱情况
        cluster_analysis = self.df.groupby('兴趣聚类').agg({
            '是否单身': 'mean',
            '有恋爱意愿': 'mean',
            '年龄': 'mean',
            '社交活跃度编码': 'mean'
        }).round(3)

        print("\n不同兴趣聚类组的特征：")
        print(cluster_analysis)

        # 2. 交互效应分析：社交活跃度×兴趣圈层
        print("\n=== 社交活跃度与兴趣圈层的交互效应 ===")

        # 选择几个主要的兴趣圈层
        main_interests = ['兴趣_二次元', '兴趣_游戏', '兴趣_体育', '兴趣_文艺']

        for interest in main_interests:
            interest_name = interest.replace('兴趣_', '')

            # 创建交互项
            self.df[f'{interest}_交互'] = self.df[interest] * self.df['社交活跃度编码']

            # 分组分析
            high_activity = self.df[self.df['社交活跃度编码'] >= 3]
            low_activity = self.df[self.df['社交活跃度编码'] <= 2]

            high_rate = high_activity[high_activity[interest] == 1]['是否单身'].mean()
            low_rate = low_activity[low_activity[interest] == 1]['是否单身'].mean()

            print(f"{interest_name}圈层：")
            print(f"  高活跃度组单身率：{high_rate:.3f}")
            print(f"  低活跃度组单身率：{low_rate:.3f}")
            print(f"  差异：{abs(high_rate - low_rate):.3f}")

    def generate_report(self):
        """
        生成分析报告
        """
        print("\n=== 分析总结报告 ===")

        # 主要发现
        print("\n主要发现：")

        # 计算总体恋爱率
        overall_single_rate = self.df['是否单身'].mean()
        overall_relationship_rate = 1 - overall_single_rate

        print(f"1. 总体恋爱率：{overall_relationship_rate:.2%}")
        print(f"2. 总体单身率：{overall_single_rate:.2%}")

        # 找出恋爱率最高和最低的兴趣圈层
        interest_cols = [col for col in self.df.columns if col.startswith('兴趣_')]
        relationship_rates = []

        for interest in interest_cols:
            interest_name = interest.replace('兴趣_', '')
            group = self.df[self.df[interest] == 1]
            if len(group) > 10:
                rate = 1 - group['是否单身'].mean()
                relationship_rates.append((interest_name, rate, len(group)))

        relationship_rates.sort(key=lambda x: x[1], reverse=True)

        print(f"\n3. 恋爱率最高的兴趣圈层：{relationship_rates[0][0]} ({relationship_rates[0][1]:.2%})")
        print(f"4. 恋爱率最低的兴趣圈层：{relationship_rates[-1][0]} ({relationship_rates[-1][1]:.2%})")

        # 社交活跃度的影响
        high_active = self.df[self.df['社交活跃度编码'] >= 3]
        low_active = self.df[self.df['社交活跃度编码'] <= 2]

        print(f"\n5. 高社交活跃度群体恋爱率：{1 - high_active['是否单身'].mean():.2%}")
        print(f"6. 低社交活跃度群体恋爱率：{1 - low_active['是否单身'].mean():.2%}")


def main():
    """
    主函数
    """
    print("兴趣圈层对恋爱情况影响分析系统")
    print("=" * 50)

    analyzer = InterestCircleAnalysis("survey_data_for_analysis.csv")

    # 执行各项分析
    analyzer.descriptive_analysis()
    analyzer.correlation_analysis()
    analyzer.regression_analysis()
    analyzer.advanced_analysis()
    analyzer.generate_report()


if __name__ == "__main__":
    main()
