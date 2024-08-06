#使用sklearn接口使用XGBoost

from xgboost import XGBRegressor
from sklearn.model_selection import cross_validate, KFold
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import warnings
warnings.filterwarnings("ignore")


data = pd.read_csv('data_51,12_new.csv')
# data = pd.read_csv('data_51,12_chuyi_n.csv')


X = data.iloc[:,1:-1]
y = data.iloc[:,-1]

#sklearn普通训练代码三步走：实例化，fit，score

range = [-6,56]

params = {          'n_estimators':180
                    ,'learning_rate':0.15
                    ,'booster':'gbtree'
                    ,'colsample_bytree':1
                    ,'colsample_bynode':0
                    ,'gamma':5
                    ,'reg_lambda':0.8
                    ,'min_child_weight':0
                    ,'max_depth':6
                    ,'subsample':0.8
                    # ,'max_features':'auto'
                    # ,"objective":params["objective"]
                    # ,'rate_drop':0.1
                    # ,'random_state':1412
                    }

model = XGBRegressor(**params
                     ,random_state=1446
                     )
#实例化模型

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.15
                                                 ,random_state=1443
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

ax.set_title("XGBoost-12",fontsize=19)
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
#                                ,scoring=score3 #负根均方误差
#                                ,return_train_score=True
#                                # ,verbose=True
#                                )
# print('R2:', np.mean(loss3['train_score']), np.mean(loss3['test_score']))

# # feature_importance
# model = GradientBoostingRegressor(**params,)
# #基于最优参数再单独训练一个新模型，进行一次留一验证，测试集比例0.1(和文献一样)
#



# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15
#                                                     # ,random_state=1443
#                                                     )
# model.fit(X, y)
# feature_importance = model.feature_importances_
# # print(feature_importance)
# sorted_idx = np.argsort(feature_importance) #将一维数组从小到大排序并返回索引值
# # print(sorted_idx.shape[0])
# pos = np.arange(sorted_idx.shape[0]) + 0.5
# #
# figure = plt.figure(figsize=(12, 12))
# # plt.plot()
# plt.barh(pos, feature_importance[sorted_idx], align="center")
# feature_names = ['MW1','qed1','BJ1','HA1','MLP1','MR1','MW2','VE2','TS2','HD2','RB2','MLP2']
# plt.yticks(pos, np.array(feature_names)[sorted_idx])
# plt.title("Feature Importance",fontsize=15 )
# plt.show()