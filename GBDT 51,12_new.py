#导入依赖库
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, ShuffleSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import math
import numpy as np
# from joblib import dump, load
from sklearn.model_selection import cross_validate, KFold
# from scipy.interpolate import interp1d
# import scienceplots
#使用SciencePlots
# with plt.style.context(['science','no-latex']):

# # 导入pylustrator
# import pylustrator
# # 开启pylustrator
# pylustrator.start()

#使用sns绘图风格
# import seaborn as sns
# plt.style.use('seaborn-whitegrid')
# sns.set_style("white")
# sns.set()

import warnings
warnings.filterwarnings('ignore')


#读取数据表格
# data = pd.read_csv('data_51,12_new.csv')
data = pd.read_csv('data_51,12_chuyi_n.csv')


# print(data)
#输入特征X(从0计，第二列至倒数第二列)
X = data.iloc[:,1:-1]
# print(X.shape)
#自动拼接成numpy描述符
X = X.to_numpy()
#数据标准化(可能提高模型性能，注释该部分代码以跳过)
# X = StandardScaler().fit_transform(X)
# print(X)

#标签Y Std(最后一列)
Y = data.iloc[:,-1]
# print(Y)
Y = Y.to_numpy()
# print(Y.shape)

# print('Min:',min(Y))
# print('Max:',max(Y))

range=[-6,56]



# #可视化交叉验证得到的最优模型在整个数据集上的性能
#
# #获取最优模型
# best_model = clf.best_estimator_
# #预测
# Y_pred = best_model.predict(X)
# #绘制
# fig, ax = plt.subplots()
# #绘制预测-真值对散点
# ax.scatter(Y,Y_pred)
# #绘制对角线
# ax.plot([range], [range], "--k")
# ax.set_ylabel("Target predicted")
# ax.set_xlabel("True Target")
# ax.set_title("Regression performance of the best model")
# #计算决定系数R2,MAE以及RMSE作为图注
# ax.text(
#     -5,
#     40,
#     r"$R^2$=%.2f, MAE=%.2f, RMSE=%.2f"
#     % (r2_score(Y, Y_pred), mean_absolute_error(Y, Y_pred), math.sqrt(mean_squared_error(Y, Y_pred))),
# )
# #x y轴上下限
# ax.set_xlim(range)
# ax.set_ylim(range)
#
# plt.show()
#
# print(clf.best_params_)

#这一行不需要注释，后面两块代码都要用
# params = {'learning_rate': 0.01, 'loss': 'huber', 'subsample': 1,'n_estimators':300, 'max_depth':3, 'max_features':'auto'}
# params={'criterion': 'squared_error', 'learning_rate': 0.1, 'loss': "absolute_error", 'max_depth': 6, 'max_features': 'sqrt', 'min_impurity_decrease': 4.0, 'n_estimators': 200, 'subsample': 0.6}
params = {  'criterion': 'friedman_mse'
            , 'learning_rate': 0.17
            , 'loss': "huber"
            , 'max_depth': 5
            # , 'max_features': 'auto'
            , 'min_impurity_decrease': 4
            , 'n_estimators': 160
            , 'subsample': 0.7
            }

#基于最新模优参数训练型
model = GradientBoostingRegressor(**params
                                  ,random_state=1445
                                  #1445
                                  )

# # 基于最优参数再单独训练一个新模型，进行一次留一验证，测试集比例0.1(和文献一样)
# # 分配训练集，验证集
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15
                                                    , random_state=1443
                                                    )


model.fit(X_train, y_train)


# score = 'neg_mean_absolute_error'
# # score = 'r2'
# a1 = cross_val_score(model,X,Y,cv=20,scoring=score)
# score = np.mean(a1)
#
# print('十折交叉验证的得分为：{}'.format(score))


#保存模型
# dump(model,'GBDT.joblib')

#分别预测
Y_pred = model.predict(X)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

#绘制

# with plt.style.context(['science','no-latex']):


fig, ax = plt.subplots(figsize=(6,5.5))

ax.plot([-6,56], [-6,56], "--k",alpha=0.7)
#绘制预测-真值对散点
#训练集数据点

scatter1 = ax.scatter(y_train,y_train_pred,s=110
                      ,color = '#317CB7'
                      ,marker = 'o',alpha=0.9,edgecolors='black'
           )
#测试集数据点(红色)
scatter2 = ax.scatter(y_test,y_test_pred,s=110
                      ,color='#E73847'
                      ,marker = 'o',alpha=0.9,edgecolors='black')

legend_x = 0.02  # 自定义图例的水平位置
legend_y = 0.78  # 自定义图例的垂直位置
# ax.legend(loc='upper right', bbox_to_anchor=(legend_x, legend_y))


ax.legend([scatter1,scatter2],['Training set','Test set'],loc='upper left',bbox_to_anchor=(legend_x, legend_y))
#绘制对角线

ax.set_ylabel("Predicted S$_{i}$ (mV/K)",fontsize=17,fontname='Arial')

ax.set_xlabel("True Target (mV/K)",fontsize=17,fontname='Arial')

ax.set_title("GBDT-12",fontsize=19)
#计算整个数据集的决定系数R2和MAE作为图注
ax.text(
    -3.5,
    50.5,
    r"Train: R$^{2}$=%.2f, MAE=%.2f, RMSE=%.2f"
    % (r2_score(y_train, y_train_pred), mean_absolute_error(y_train, y_train_pred)
       ,mean_squared_error(y_train, y_train_pred,squared=True))
    # ,color = 'tomato'
    , fontsize=13
    ,fontname='Arial'
)
ax.text(
    -3.5,
    45,
    r"Test: R$^{2}$=%.2f, MAE=%.2f, RMSE=%.2f"
    % (r2_score(y_test, y_test_pred), mean_absolute_error(y_test, y_test_pred)
       , mean_squared_error(y_test, y_test_pred,squared=True))
    # , color='skyblue'
    ,fontsize=13
    ,fontname='Arial'
)

#x y轴上下限
ax.set_xlim(range)
ax.set_ylim(range)

plt.xticks(fontsize=13,fontname='Arial')
plt.yticks(fontsize=13,fontname='Arial')


plt.show()



#十折交叉验证,计算模型整体的性能
cv = KFold(n_splits=10,shuffle=True,random_state=1443)

score1 = 'neg_mean_absolute_error'
score2 = 'neg_root_mean_squared_error'
score3 = 'r2'
score4 = 'neg_median_absolute_error'

loss1 = cross_validate(model, X, Y    #注意：这里放入的模型不需要进行训练，数据也直接放入X,y
                               ,cv=cv
                               ,scoring=score1
                               ,return_train_score=True
                               # ,verbose=True
                               )
print('MAE:', -np.mean(loss1['train_score']), -np.mean(loss1['test_score']))

loss2 = cross_validate(model, X, Y    #注意：这里放入的模型不需要进行训练，数据也直接放入X,y
                               ,cv=cv
                               ,scoring=score2
                               ,return_train_score=True
                               # ,verbose=True
                               )
print('RMSE:', -np.mean(loss2['train_score']), -np.mean(loss2['test_score']))

loss3 = cross_validate(model, X, Y    #注意：这里放入的模型不需要进行训练，数据也直接放入X,y
                               ,cv=cv
                               ,scoring=score3 #负根均方误差
                               ,return_train_score=True
                               # ,verbose=True
                               )
print('R2:', np.mean(loss3['train_score']), np.mean(loss3['test_score']))



# # 用多种可视化曲线更全面地分析我们的最优模型



# # 自定义绘制曲线的方法
# def plot_learning_curve(
#         estimator,
#         title,
#         X,
#         y,
#         score="neg_root_mean_squared_error",
#         axes=None,
#         ylim=None,
#         cv=None,
#         train_sizes=np.linspace(0.1, 1.0, 5),
# ):
#
#     # axes.set_title(title,fontname='Arial',fontsize=16)
#     if ylim is not None:
#         axes.set_ylim(*ylim)
#     axes.set_xlabel("Training samples",fontname='Arial',fontsize=15)
#     axes.set_ylabel("RMSE (mV/K)",fontname='Arial',fontsize=15)
#
#     # 使用learning_curve()方法，可自动交叉验证并返回各种统计信息
#     # 训练使用的训练集数目逐次增加
#     train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
#         estimator,
#         X,
#         y,
#         cv=cv,
#         scoring=score,
#         train_sizes=train_sizes,
#         return_times=True,
#     )
#     # 计算性能指标平均值和标准差
#     train_scores_mean = np.mean(train_scores, axis=1)
#     train_scores_std = np.std(train_scores, axis=1)
#     test_scores_mean = np.mean(test_scores, axis=1)
#     test_scores_std = np.std(test_scores, axis=1)
#     fit_times_mean = np.mean(fit_times, axis=1)
#     fit_times_std = np.std(fit_times, axis=1)
#
#     # 绘制学习曲线
#     # 绘制网格
#     # axes.grid()
#     # 指标波动范围(指标标准差，反映模型稳定性，或说泛化能力)
#     # axes[0].fill_between(
#     #     train_sizes,
#     #     train_scores_mean - train_scores_std,
#     #     train_scores_mean + train_scores_std,
#     #     alpha=0.1,
#     #     color="r",
#     # )
#     # axes[0].fill_between(
#     #     train_sizes,
#     #     test_scores_mean - test_scores_std,
#     #     test_scores_mean + test_scores_std,
#     #     alpha=0.1,
#     #     color="g",
#     # )
#     # 以指标均值绘制曲线
#     score_train = (-1)*train_scores_mean
#     score_test = (-1)*test_scores_mean
#     axes.plot(
#         train_sizes, score_train, "o-", color="#317CB7", label="Training"
#     )
#     axes.plot(
#         train_sizes, score_test, "o-", color="#E73847", label="Test"
#     )
#     # 图例
#     axes.legend(loc="best")
#
#     return plt
#
#
# # 3行1列
# # 如果各图有重叠figsize的宽或靠的太近 自行调大高
# fig, axes = plt.subplots(1, 1, figsize=(4, 3.3),constrained_layout=True)
#
# # 自定义样本分配
# cv = ShuffleSplit(n_splits=15, test_size=0.2, random_state=1445)
#
# title = "Learning Curve"
#
# # 使用我们自己搜索到的最佳超参
# # estimator = GradientBoostingRegressor(**params,random_state=1445)
# estimator = model
# # 绘制3种曲线 如果纵轴范围不合理 则自行调整ylim
# # 建议使用较大的交叉验证次数(cv)，以减小每次测试平均指标的不确定性，有利于曲线平滑
# plot_learning_curve(
#     estimator, title, X, Y, axes=axes,ylim=(-2, 18), cv=cv,
# )
# plt.xticks(fontname='Arial', fontsize=14)
# plt.yticks(fontname='Arial', fontsize=14)
# plt.show()



# # feature_importance
# model = GradientBoostingRegressor(**params,)
# #基于最优参数再单独训练一个新模型，进行一次留一验证，测试集比例0.1(和文献一样)
#
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15,random_state=1443)
# model.fit(X_train, y_train)
# feature_importance = model.feature_importances_
# sorted_idx = np.argsort(feature_importance) #将一维数组从小到大排序并返回索引值
# print(feature_importance)
# pos = np.arange(sorted_idx.shape[0]) + 0.5
# #
# figure = plt.figure(figsize=(12, 12))
# # plt.plot()
# plt.barh(pos, feature_importance[sorted_idx], align="center")
# feature_names = ['MW1','qed1','BJ1','HA1','MLP1','MR1','MW2','VE2','TS2','HD2','RB2','MLP2']
# plt.yticks(pos, np.array(feature_names)[sorted_idx])
# #
# plt.title("Feature Importance",fontsize=15 )
# plt.show()