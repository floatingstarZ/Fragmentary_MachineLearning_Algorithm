# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from datetime import date
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor,plot_importance
import xgboost as xgb
from sklearn import cross_validation, metrics
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split,cross_val_predict,cross_val_score, ShuffleSplit
from sklearn.model_selection import GridSearchCV,KFold
from sklearn import cross_validation, metrics
import matplotlib.pylab as plt
from matplotlib.pyplot import savefig
from matplotlib.pylab import rcParams

def full_birthDate(x):
    if(x[-2:] == '00'):
        return x[:-2]+'20'+x[-2:]
    else:
        return x[:-2]+'19'+x[-2:]

def trans(x):
    if(x == 'Medium'):
        return 1
    elif(x == 'High'):
        return 2
    else:        
        return 0

def judge(x):
    if(x > 0):
        return 1
    else:        
        return 0

# 读取数据
def getData():
    train = pd.read_csv('./data/train.csv')
    test = pd.read_csv('./data/test.csv')    

    train['birth_date_'] = train['birth_date'].apply(lambda x: full_birthDate(x))
    test['birth_date_'] = test['birth_date'].apply(lambda x: full_birthDate(x))    

    train['birth_date'] = pd.to_datetime(train['birth_date_'])
    train['age'] = ((pd.Timestamp.now() - train['birth_date']).apply(lambda x: x.days) / 365).apply(lambda t: int(t))

    test['birth_date'] = pd.to_datetime(test['birth_date_'],format='%m/%d/%Y', errors='coerce')
    test['age'] =  ((pd.Timestamp.now() - test['birth_date']).apply(lambda x: x.days) / 365).apply(lambda t: int(t))

    train['work_rate_att_'] = train['work_rate_att'].apply(lambda x: trans(x)).apply(lambda t: int(t))
    train['work_rate_def_'] = train['work_rate_def'].apply(lambda x: trans(x)).apply(lambda t: int(t))

    test['work_rate_att_'] = test['work_rate_att'].apply(lambda x: trans(x)).apply(lambda t: int(t))
    test['work_rate_def_'] = test['work_rate_def'].apply(lambda x: trans(x)).apply(lambda t: int(t))

    train = train.drop('id',axis=1)
    train = train.drop('birth_date',axis=1)
    train = train.drop('birth_date_',axis=1)
    train = train.drop('work_rate_att',axis=1)
    train = train.drop('work_rate_def',axis=1)

    test = test.drop('id',axis=1)
    test = test.drop('birth_date',axis=1)
    test = test.drop('birth_date_',axis=1)
    test = test.drop('work_rate_att',axis=1)
    test = test.drop('work_rate_def',axis=1)

    return train,test

def result_(test_res):
    submit = pd.read_csv('./data/sample_submit.csv')    
    submit['y'] = np.array(test_res)
    submit.to_csv('my_RF_prediction.csv', index=False)

def data_ana(train, test):

    # 获得球员最擅长位置上的评分
    positions = ['rw', 'rb', 'st', 'lw', 'cf', 'cam', 'cm', 'cdm', 'cb', 'lb', 'gk']

    train['best_pos'] = train[positions].max(axis=1)
    test['best_pos'] = test[positions].max(axis=1)

    # 计算球员的身体质量指数(BMI)
    train['BMI'] = 10000. * train['weight_kg'] / (train['height_cm'] ** 2)
    test['BMI'] = 10000. * test['weight_kg'] / (test['height_cm'] ** 2)

    # 判断一个球员是否是守门员
    train['is_gk'] = train['gk'].apply(lambda x: judge(x))
    test['is_gk'] = test['gk'].apply(lambda x: judge(x))    

    return train,test

def view_filter(train):
    # 可视化盒图    
    # # 统计输出信息
    percentile_result = np.percentile(train['y'], [25, 50, 75])
    num = 0
    for i in list(train['y']):
        if(i > percentile_result[2] * 1.5):
            num+=1
            print(i)
    # print('离群点个数：',num,'\n四分位数Q3：',percentile_result[2])
    # print(num/len(list(train['y'])))
    # 显示图例
    plt.boxplot(x=train['y'],showmeans=True,meanline=True,whis=1.5)
    plt.legend()
    savefig('盒图.jpg')
    # 显示图形
    plt.show()
    plt.close()


def modelfit(alg, data, labels_, cols, useTrainCV=True, cv_folds=7, early_stopping_rounds=50):
    # 可以返回n_estimates的最佳数目，为什么呢, 哪里返回？
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(data, label=labels_)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='mae', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
    #Fit the algorithm on the data
    seed = 20
    # seed=20从0.97升为了0.98
    # Model Report
    # r2_score : 0.9845
    # MAE:  0.4723899992310908 %
    test_size = 0.3
    x_train,x_test,y_train,y_test = train_test_split(data, labels_, test_size=test_size,random_state=seed)    
    print(x_train.shape[1],y_train.shape[1])    
    eval_set = [(x_test,y_test)]
    alg.fit(x_train, y_train, early_stopping_rounds=early_stopping_rounds, eval_metric='mae',eval_set=eval_set,verbose=True)        
    #Predict training set:
    dtrain_predictions = alg.predict(x_test)

    # print(type(dtrain_predictions),type(labels_))
    y_true = list(y_test)
    y_pred = list(dtrain_predictions)
    print(type(y_pred),type(y_true))
    
    #Print model report:
    print("\nModel Report")
    print("r2_score : %.4g" % metrics.r2_score(y_true, y_pred))
    mae_y = 0.00
    for i in range(len(y_true)):
        mae_y += np.abs(np.float(y_true[i])-y_pred[i])
    print("MAE: ", (mae_y*4799+6)/len(y_true))    
    # Model Report
    # r2_score : 0.9673
    # MAE:  0.636517748270864 %
    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)   

    # feat_imp.plot(kind='bar', title='Feature Importances')
    # plt.ylabel('Feature Importance Score')
    fig, ax = plt.subplots(1, 1, figsize=(8, 13))    
    plot_importance(alg, max_num_features=20, height=0.5, ax=ax)
    plt.show()
    # 重要性排序
    feat_sel = list(feat_imp[feat_imp.values > 18].index)
    featur = []
    for i in range(len(feat_sel)):
        featur.append(cols[int(feat_sel[i][1:])])
    return featur
    
def MAE_(xgb1,train_x,train_y):
    y_pre = xgb1.predict(train_x)
    num = 0
    for i in range(len(y_pre)):
        num += np.abs(y_pre[i] - train_y[i])
    print((num*4799+6))

def xgboost_train(data_, labels_, cols):  
    # print(data_.hea说d(),labels_.head())  
    # # 特征选择------------------------------------------------------------------
    # xgb1 = XGBRegressor(learning_rate =0.1,max_depth=5,min_child_weight=1,n_estimators=1000,
    #                 gamma=0,subsample=0.8,colsample_bytree=0.8,objective= 'reg:logistic',
    #                     nthread=4,scale_pos_weight=1,seed=27)   
    # modelfit(xgb1, data_.values,labels_.values,cols)
    # feature_ = modelfit(xgb1, data_.values,labels_.values,cols) # 特征选择
    # print(feature_)
    # # 结果------------------------------------------------------------------
    # Model Report
    # r2_score : 0.986
    # MAE:  20.587096283760808
    # # 特征选择------------------------------------------------------------------
    feature_ = ['potential', 'age', 'best_pos', 'reactions', 'club', 'long_passing', 'BMI', 'vision', 'heading_accuracy', 'st', 'nationality', 'stamina', 'cf', 'aggression', 'free_kick_accuracy', 'pas', 'finishing', 'crossing', 'phy', 'marking']
    train_x = data_[feature_]    
    train_y = labels_.values    
    # # 半手动调参----------------------------------------------------------------------------------------------
    # param_test1 = {
    #           'max_depth':[18,19,20]
    # }
    # gsearch1 = GridSearchCV(estimator = XGBRegressor(learning_rate=0.1, n_estimators=366,min_child_weight=1,subsample=0.85,colsample_bytree=0.8,
    #                    gamma=0,objective= 'reg:logistic', nthread=4, scale_pos_weight=1, seed=27), 
    #                 param_grid = param_test1,scoring='neg_median_absolute_error',n_jobs=4, iid=False, cv=5)
    # gsearch1.fit(train_x,train_y)
    # print(gsearch1.best_params_,gsearch1.best_score_)
    # # 半手动调参---结果太差了-------------------------------------------------------------------------------------------
    # {'max_depth': 19, 'min_child_weight': 1} -0.000962098801735801
    # 虽然是负数，最后的mae还是可以的23.55
    train_x = train_x.as_matrix()
    xgb1 = XGBRegressor(learning_rate=0.1, n_estimators=366, max_depth=19, min_child_weight=1,subsample=0.85,colsample_bytree=0.8,
                       gamma=0,objective= 'reg:logistic', nthread=4, scale_pos_weight=1, seed=27)
    xgb1.fit(train_x,train_y)
    MAE_(xgb1,train_x,train_y)
    return xgb1

if __name__ == '__main__':
    train_,test_ = getData()
    train,test = data_ana(train_,test_)
    # print(train.shape[1])        
    # p = view_filter(train) # 离群点

    # reg:logistic要求对label归一化
    minn = train['y'].min(axis=0)
    maxx = train['y'].max(axis=0)
    # print(minn,maxx)
    train['yy'] = train['y'].apply(lambda x: (x - minn)/(maxx - minn))
    train = train.drop('y',axis=1)

    # 为了方便分片，调整列的顺序 
    cols = train.columns.values.tolist()
    cols.insert(train.shape[0]-1,cols.pop(cols.index('yy'))) # 将标签列放在末尾
    print(train.shape[1])
    print(test.shape[1])

    # xgboost_train(train.iloc[:,:train.shape[1]-1], train.iloc[:,train.shape[1]-1:],cols)

    xgb_ = XGBRegressor()
    xgb_ = xgboost_train(train.iloc[:,:train.shape[1]-1], train.iloc[:,train.shape[1]-1:],cols)
    
    feature_ = ['potential', 'age', 'best_pos', 'reactions', 'club', 'long_passing', 'BMI', 'vision', 'heading_accuracy', 'st', 'nationality', 'stamina', 'cf', 'aggression', 'free_kick_accuracy', 'pas', 'finishing', 'crossing', 'phy', 'marking']
    test = test[feature_]
    test = test.as_matrix()    
    # print(test.head())
    test_pre = xgb_.predict(test)
    test_pre_ = pd.DataFrame(test_pre).apply(lambda x: (maxx - minn)*x + minn)
    result_(test_pre_)
