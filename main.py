import numpy as np
import scipy as sp
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import xgboost as xgb
import os
from sklearn.externals import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt

def startjob(isxgb):
    pd.set_option('display.width', 400)
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('display.max_columns', 70)
    column_names = 'age', 'workclass', 'fnlwgt', 'education', 'education-num', \
                   'marital-status', 'occupation', 'relationship', 'race', \
                   'sex', 'capital-gain', 'capital-loss', \
                   'hours-per-week', 'native-country', 'incometype'
    if isxgb:
        model_file = './data/adult_xgb.pkl'
    else:
        model_file = './data/adult.pkl'

    if os.path.exists(model_file):
        model = joblib.load(model_file)
    else:
        print('读入数据')
        data = pd.read_csv('./data/adult.data', header=None, names = column_names)
        x, y = codeforenumcolumns(data)
        print(x.head())
        print(y.head())
        # print(x.describe())
        # print(y.describe())
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size=0.7, random_state=0)
        if isxgb:
            model = getxgboostmodel(model_file, x_train, x_valid, y_train, y_valid)
        else:
            model = getsklearnmodel(model_file, x_train, x_valid, y_train, y_valid)
    data_test = pd.read_csv('./data/adult.test', header=None, skiprows=1, names = column_names)
    x_test, y_test = codeforenumcolumns(data_test)
    if isxgb:
        datafortest = xgb.DMatrix(x_test, label=y_test)
        y_test_pred = model.predict(datafortest)
        printxgbscore(y_test_pred, datafortest.get_label())
    else:
        testbysklearn(model, x_test, y_test)
        y_test_proba = model.predict_proba(x_test)
        print(y_test_proba)
        y_test_proba = y_test_proba[:, 1]
        #Compute Receiver operating characteristic(ROC), 接受者操作特征
        #横轴：负正类率(false postive rate, FPR) 特异度，划分实例中所有负例占所有负例的比例
        #纵轴：真正类率(true postive rate, TPR) 灵敏度，Sensitivity(正类覆盖率)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_test_proba)
        #AUC (Area under Curve): Roc曲线下的面积，介于0.1和1之间。Auc作为数值可以直观的评价分类器的好坏，值越大越好
        auc = metrics.auc(fpr, tpr)
        print('AUC = ', auc)
        #或直接调用roc_auc_score
        #print('AUC = ', metrics.roc_auc_score(y_test, y_test_proba)

        mpl.rcParams['font.sans-serif'] = u'SimHei'
        mpl.rcParams['axes.unicode_minus'] = False
        plt.figure(facecolor='w')
        plt.plot(fpr,
                 tpr,
                 'r-',
                 lw=2,
                 alpha=0.8,
                 label='AUC=%3.f' % auc)
        plt.plot((0, 1), (0, 1), c='b', lw=1.5, ls='--', alpha=0.7)
        plt.xlim((-0.01, 1.02))
        plt.ylim((-0.01, 1.02))
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.grid(b=True)
        plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=14)
        plt.title(u'预测人群收入数据的ROC曲线和AUC值 ', fontsize=17)
        plt.show()

def testbysklearn(model, x_test, y_test):
    y_test_pred = model.predict(x_test)
    print('测试集准确率: ', accuracy_score(y_test, y_test_pred))
    print('测试集查准率: ', precision_score(y_test, y_test_pred))
    print('测试集召回率: ', recall_score(y_test, y_test_pred))
    print('测试集F1: ', f1_score(y_test, y_test_pred))


def codeforenumcolumns(data):
    for name in data.columns:
        if name in ('workclass', 'education', 'marital-status',
                    'occupation', 'relationship', 'race', 'sex', 'native-country', 'incometype'):
            data[name] = pd.Categorical(data[name]).codes
    x = data[data.columns[:-1]]
    y = data[data.columns[-1]]
    return x, y

def getsklearnmodel(model_file, x_train, x_valid, y_train, y_valid):
    show_result = True
    model = RandomForestClassifier(n_estimators=50,
                                   criterion='gini',
                                   max_depth=14,
                                   min_samples_split=7,
                                   oob_score=True,
                                   class_weight={0: 1, 1: 1 / y_train.mean()})
    # model = GradientBoostingClassifier(n_estimators=20)
    model.fit(x_train, y_train)
    joblib.dump(model, model_file)
    if show_result:
        print('OOB准确率: ', model.oob_score_)
        y_train_pred = model.predict(x_train)
        print('训练集准确率: ', accuracy_score(y_train, y_train_pred))
        print('训练集查准率: ', precision_score(y_train, y_train_pred))
        print('训练集召回率: ', recall_score(y_train, y_train_pred))
        print('训练集F1: ', f1_score(y_train, y_train_pred))

        y_valid_pred = model.predict(x_valid)
        print('验证集准确率: ', accuracy_score(y_valid, y_valid_pred))
        print('验证集查准率: ', precision_score(y_valid, y_valid_pred))
        print('验证集召回率: ', recall_score(y_valid, y_valid_pred))
        print('验证集F1: ', f1_score(y_valid, y_valid_pred))
    return model

def getxgboostmodel(model_file, x_train, x_valid, y_train, y_valid):
    data_train = xgb.DMatrix(x_train, label=y_train)
    data_test = xgb.DMatrix(x_valid, label=y_valid)
    watch_list = [(data_test, 'eval'), (data_train, 'train')]
    param = {'max_depth': 6,
             'eta': 0.3,
             'silent': 1,
             'objective': 'binary:logistic'}
    bst = xgb.train(param, data_train, num_boost_round=30, evals=watch_list)
    y_hat = bst.predict(data_test)
    y = data_test.get_label()
    printxgbscore(y_hat, y)
    joblib.dump(bst, model_file)
    return bst

def printxgbscore(y_hat, y):
    error = sum(y != (y_hat > 0.5))
    errorrate = float(error) / len(y_hat)
    print('样本总数: \t', len(y_hat))
    print('错误数目: \t%4d' % error)
    print('准确率: \t%.5f%%' % (100 - 100 * errorrate))

if __name__ == '__main__':
    startjob(False)
