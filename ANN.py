from sklearn.neural_network import MLPRegressor
#导入依赖库
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, ShuffleSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import math
import numpy as np
from joblib import dump, load
from sklearn.model_selection import cross_validate, KFold
#超参数设置可能不合理，训练中会出现警告信息，可以忽略掉(如不忽略请注释掉代码)
import warnings
warnings.filterwarnings("ignore")

#读取数据表格
# data = pd.read_csv('data_51,12_new.csv')
data = pd.read_csv('data_51,12_chuyi_n.csv')


X = data.iloc[:,1:-1]
X = X.to_numpy()
y = data.iloc[:, -1]
y = y.to_numpy()
X = StandardScaler().fit_transform(X)

range=[-6,56]



# model = MLPRegressor({'activation':'tanh'})
#
# tuned_parameters = [
#     {"hidden_layer_sizes": [(6,3),(6,3,2),(5,),(5,2),(4,2),(4,)]
#         # , "activation":["logistic","tanh"]
#      ,"learning_rate_init":[0.01,0.1,0.15,0.5],"max_iter":[100,200,300],"alpha":[0.0001,0.01,0.1]
#     },]
#
# #构建相应的交叉验证超参搜索器(这里使用网格搜索) 使用平均误差MAE的负数作为评分指标
# clf = GridSearchCV(model, tuned_parameters, scoring="neg_root_mean_squared_error",cv=5)
#
# #反复训练并搜索最佳超参
# clf.fit(X,y)
#
# print("当前数据集上的最佳模型超参数设置:")
# print(clf.best_params_)
# print("相应评分：")
# print("%0.3f eV" %(-clf.best_score_)) #这里的%0.3为设置保留三位小数
# print("各超参数设置及相应模型评分(评分/±标准差):")
# means = -clf.cv_results_["mean_test_score"]
# stds = clf.cv_results_["std_test_score"]
# for mean, std, params in zip(means, stds, clf.cv_results_["params"]):
#     print("%0.3f  (+/-%0.03f ) for %r" % (mean, std * 2, params))


# params = {'activation': 'tanh'
#     , 'alpha': 0.1
#     , 'hidden_layer_sizes': (5, )
#     , 'learning_rate_init': 0.01
#     , 'max_iter': 5000
#
# }
params = {
    'hidden_layer_sizes': (6,),
    'alpha':0.05456326364630905,
    'activation':'tanh',
    # 'activation':'logistic',
    # 'solver':'lbfgs',
    # 'learning_rate':'invscaling',
    'learning_rate':'constant',
    'learning_rate_init':0.1698036839261919,
    'max_iter':677,
    'momentum': 0.16236316773363227,

}



model = MLPRegressor(**params
                     ,random_state=1473

                     )
#实例化模型 1447 1473

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.15
                                                 , random_state=1443
                                                 )

model.fit(X_train,y_train)

Y_pred = model.predict(X)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

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

ax.legend([scatter1,scatter2],['Training set','Test set'],loc='lower right')
#绘制对角线

ax.set_ylabel("Predicted S$_{i}$ (mV/K)",fontsize=17,fontname='Arial')
ax.set_xlabel("True Target (mV/K)",fontsize=17,fontname='Arial')

ax.set_title("ANN-12",fontsize=19)
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
#
# #十折交叉验证,计算模型整体的性能
# cv = KFold(n_splits=10,shuffle=True,random_state=1443)
#
# score1 = 'neg_mean_absolute_error'
# score2 = 'neg_root_mean_squared_error'
# score3 = 'r2'
# score4 = 'neg_median_absolute_error'
#
# loss1 = cross_validate(model, X, y    #注意：这里放入的模型不需要进行训练，数据也直接放入X,y
#                                ,cv=cv
#                                ,scoring=score1
#                                ,return_train_score=True
#                                # ,verbose=True
#                                )
# print('MAE:', -np.mean(loss1['train_score']), -np.mean(loss1['test_score']))
#
# loss2 = cross_validate(model, X, y    #注意：这里放入的模型不需要进行训练，数据也直接放入X,y
#                                ,cv=cv
#                                ,scoring=score2
#                                ,return_train_score=True
#                                # ,verbose=True
#                                )
# print('RMSE:', -np.mean(loss2['train_score']), -np.mean(loss2['test_score']))
#
# loss3 = cross_validate(model, X, y    #注意：这里放入的模型不需要进行训练，数据也直接放入X,y
#                                ,cv=cv
#                                ,scoring=score3
#                                ,return_train_score=True
#                                # ,verbose=True
#                                )
# print('R2:', np.mean(loss3['train_score']), np.mean(loss3['test_score']))
