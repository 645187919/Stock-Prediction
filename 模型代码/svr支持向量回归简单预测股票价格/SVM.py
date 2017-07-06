
#coding=utf-8
import talib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
import csv
from sklearn import preprocessing
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import datetime,time

# 第一步：获取数据。
# 参数：filename:数据文件名称
def get_data(filename):
    sample = []
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)  # skipping column names
        for row in csvFileReader:
            sample.append(row)
    print "samples:", sample
    return sample
# 第二步：处理数据。
# @parameters  data:原始数据 k：金融指标用于影响sma的一个值。当k=1时代表数据为原始数据。
def deal_data(date,k):
    close=[ ]
    open=[ ]
    high=[ ]
    low=[ ]
    time=[ ]
    volume=[ ]
    # 提取出需要的特征
    for row in date:
        time.append(row[1])
        open.append(float(row[2]))
        high.append(float(row[3]))
        low.append(float(row[4]))
        close.append(float(row[5]))
        volume.append(float(row[7]))

    deal_time = np.reshape(time, (len(time), 1))
    deal_open = np.reshape(open, (len(open), 1))
    # print "sma_open:",type(deal_open[2])
    deal_high = np.reshape(high, (len(high), 1))
    deal_low = np.reshape(low, (len(low), 1))
    deal_close = np.reshape(close, (len(close), 1))
    deal_volume=np.reshape(volume, (len(close), 1))
    # --------------------------------------------------------------------------------
    # 将提取出的特征重新合成一个新的矩阵
    origal_data = np.hstack((deal_open, deal_high, deal_low,deal_close))
    # origal_data = np.hstack((deal_open, deal_high, deal_low,deal_close,deal_volume))
    copy_data=origal_data
    while k - 1:
        copy_data = np.delete(copy_data, 0, 0)
        k -= 1
    print type(origal_data[0][1])
    print "copy_data:", copy_data
    print "copy_data的长度", len(copy_data)

    print "origal_data:", origal_data
    print "origal_data的长度", len(origal_data)
    print "-----------------------------------"
    return origal_data,copy_data

# --------------------------------------------------------------------------------------
# MA技术指标处理数据
def MA_deal_data(date,k):

    close = []
    open = []
    high = []
    low = []
    time = []
    # 提取出需要的特征
    for row in date:

        open.append(float(row[0]))
        high.append(float(row[1]))
        low.append(float(row[2]))
        close.append(float(row[3]))

# 将其转化为数组,用于SMA方法的输入.
    open_arr = np.array(open)
    high_arr = np.array(high)
    low_arr = np.array(low)
    close_arr = np.array(close)
    time_arr = np.array(time)

    # print "数组的数据是：", close_arr
    days = int(k)  # k天均线

    sma_open = talib.SMA(open_arr, days)
    sma_high = talib.SMA(high_arr, days)
    sma_low = talib.SMA(low_arr, days)
    sma_close = talib.SMA(close_arr, days)
    # print len(sma_close)

    # print "day日均线的数组:", sma_close

# 将处理后的数据合并成一个新的矩阵.
    sma_time = np.reshape(time_arr, (len(time_arr), 1))
    sma_open = np.reshape(sma_open, (len(sma_open), 1))
    # print "sma_open:",sma_open
    sma_high = np.reshape(sma_high, (len(sma_high), 1))
    sma_low = np.reshape(sma_low, (len(sma_high), 1))
    sma_close = np.reshape(sma_close, (len(sma_high), 1))
    sma_data = np.hstack((sma_open, sma_high, sma_low, sma_close))
    # print "sma_data",sma_data

# 剔除矩阵中的无效值
    while days - 1:
        sma_data = np.delete(sma_data, 0, 0)
        days -= 1

    sma_deal_data = sma_data
    # print type(sma_deal_data[0][1])
    print len(sma_deal_data)
    print "sma_deal_data:", sma_deal_data


    return sma_deal_data
# def anto_chose_stocks():







# KDJ指标对应的策略
def KDJstrategy(data,period):
    days=period
    for i in range(data.shape[0] - days - 1):  # data.shape[0]代表data的行数.
        X = data[i:i + days, :]
        all_high=X[:,1]
        # print type(all_high[0])
        # print all_high
        all_low=X[:,2]
        all_close=X[:,3]

        slowk, slowd = talib.STOCH(all_high, all_low, all_close,
                                   fastk_period=9,
                                   slowk_period=3,
                                   slowk_matype=0,
                                   slowd_period=3,
                                   slowd_matype=0
                                   )
        # print "slowk:", slowk
        # print "leng of slowk:", len(slowk)
        slowd_copy = slowd
        slowk_copy = slowk
        # print "slowd:", slowd_copy
        close_real = all_close
        #J线图
        slowj = 3 * slowk_copy - 2 * slowd_copy
        #画图表示
        # normal = []
        # high_normal = []
        # print "j线：", slowj
        # long = len(slowj)
        # while long:
        #     normal.append(int(10))
        #     high_normal.append(int(90))
        #     long -= 1
        # print "常数线;", normal
        # x = np.linspace(0, len(slowk_copy), len(slowk_copy))
        # plt.subplot(2, 1, 1)  # （行，列，活跃区）
        # plt.title("red is slowd And orange is J")
        # plt.plot(x, slowk_copy)
        # plt.plot(x, slowd_copy, 'r')
        # plt.plot(x, slowj)
        # plt.plot(x, normal)
        # plt.plot(x, high_normal, 'k')
        #
        # # plt.plot(x,int(20))
        # # plt.plot(x,80)
        # plt.subplot(2, 1, 2)
        # plt.plot(x, close_real)
        # # plt.scatter(slowd,'g--')
        # plt.show()

        # 获得最近的kd
        slowk = slowk[-1]
        slowd = slowd[-1]
        if slowk > 90 or slowd > 80:
            print "超买区，股价有可能下跌，建议卖出股票"
        if slowk < 10 or slowd < 20:
            print "超卖区，股价有可能上涨，建议买股"


# 用SVM方法建模
# @parameters  data:经过MA处理后的数据（当k=1时也就代表该数据是原始数据）；origal_data:原始数据；period:输入的数据量
def getpre_SVM(data,origal_data,period):
    print "data:",data
    # 对数据做归一化处理
    data = preprocessing.scale(data)
    or_date=preprocessing.scale(origal_data)
    print "标注化处理后的结果：", data[0, :]
    i = 0
    t = 0.0  # t : 预测成功次数。
    m = 0.0  # m : 预测上涨，且真实情况上涨的次数。
    e = 0.0  # e : 预测上涨，但真实情况下跌的次数。
    days = period-1# days:period

    predictvalue =0#   0对应的是open属性   1：high   2：low   3：close
    # 利用滚动预测的方法进行建模
    for i in range(data.shape[0] - days - 1):  # data.shape[0]代表data的行数.
        X = data[i:i + days, :]
        y = data[i + 1:i + days + 1, predictvalue]
        svr = SVR(kernel='rbf', C=1e3, gamma=0.01)
        svr.fit(X, y)
        X2 = data[i+1:i + days + 1, :]
        y_pre = svr.predict(X2)
        y_real = or_date[i + 2:i + days + 2, predictvalue]
        #chang量代表他们的涨跌趋势
        y_real_change = y_real[-1] - y_real[-2]
        y_pre_change = y_pre[-1] - y_pre[-2]
        if y_real_change * y_pre_change > 0:
            t = t + 1

        if y_pre_change > 0:
            if y_real_change > 0:
                m = m + 1
            else:
                e = e + 1
    print "days:",period
    print "预测的正确率：",t / (len(data) - period-1) * 100
    print"预测涨的正确率：",m / (m + e) * 100

# 随机森林方法建模
def getpre_RFR(data,origal_data,period):
    '''parameters = {'n_estimators':[100,200,300],'min_samples_leaf':[1,2],\
                   'max_features':('auto','log2'),'max_depth':[3,6,9]}'''
    # parameters = {'n_estimators': [10, 20], 'max_features': ('auto', 'log2'), \
    #               'max_depth': [10, 20],'min_samples_leaf':[1,2]}
    # parameters = {'n_estimators': [10], 'max_features':['log2'], \
    #               'max_depth': [60], 'min_samples_leaf': [1]}
    data = preprocessing.scale(data)
    or_date=preprocessing.scale(origal_data)

    print "标注化处理后的结果：", data[0, :]

    i = 0
    t = 0.0  # t : 预测成功次数。
    m = 0.0  # m : 预测上涨，且真实情况上涨的次数。
    e = 0.0  # e : 预测上涨，但真实情况下跌的次数。
    days =period-1

    predictvalue = 0  # open  close   high    low
    for i in range(data.shape[0] - days - 1):  # data.shape[0]代表data的行数.
        X = data[i:i + days, :]
        y = data[i + 1:i + days + 1, predictvalue]


        clf=RandomForestRegressor(max_features=4 ,n_estimators=100,n_jobs=4)
        clf.fit(X, y)

        X2 = data[i+1:i + days + 1, :]
        y_pre = clf.predict(X2)
        y_real = or_date[i + 2:i + days + 2, predictvalue]
        y_real_change = y_real[-1] - y_real[-2]
        y_pre_change = y_pre[-1] - y_pre[-2]
        if y_real_change * y_pre_change > 0:
            t = t + 1
        if y_pre_change > 0:

            if y_real_change > 0:
                m = m + 1
            else:
                e = e + 1
    acuracy=t / (len(data) - days) * 100

    print "预测的正确率：",acuracy

    print"预测涨的正确率：", m / (m + e) * 100
    return acuracy


 # GBDT方法建模
def getpre_GBDT(data,period):
    # 参数
    parameters = {'n_estimators': [500], 'max_features': ('auto', 'log2'), \
                  'max_depth': [6],'min_samples_leaf':[1]}

    data = preprocessing.scale(data)

    print "标注化处理后的结果：", data[0, :]
    i = 0
    t = 0.0  # t : 预测成功次数。
    m = 0.0  # m : 预测上涨，且真实情况上涨的次数。
    e = 0.0  # e : 预测上涨，但真实情况下跌的次数。
    days = period

    predictvalue = 0  # open  close   high    low
    for i in range(data.shape[0] - days - 1):  # data.shape[0]代表data的行数.
        X = data[i:i + days, :]
        y = data[i + 1:i + days + 1, predictvalue]
        # 网格搜索寻找最优参数
        clf = GridSearchCV(GradientBoostingRegressor() , parameters)

        clf.fit(X, y)

        X2 = data[i+1:i + days + 1, :]
        y_pre = clf.predict(X2)
        y_real = data[i + 2:i + days + 2, predictvalue]
        y_real_change = y_real[-1] - y_real[-2]
        y_pre_change = y_pre[-1] - y_pre[-2]
        if y_real_change * y_pre_change > 0:
            t = t + 1
        if y_pre_change > 0:

            if y_real_change > 0:
                m = m + 1
            else:
                e = e + 1

    print "预测的正确率：", t / (len(data) - days) * 100

    print"预测涨的正确率：", m / (m + e) * 100
    print clf.best_params_



data = get_data(r'dates\sh600000.csv')
data = data[::-1]

print "data的类型：",type(data)
print "倒序后的数据：", data
# k日均线k的值
k=1
# 一份数据origal_data用于做SMA处理，另一份原始数据copy_data保留用于后期和预测值作比较
origal_data,copy_data=deal_data(data,k)
# sma_deal_data=MA_deal_data(origal_data,k)

# getpre_SVM(sma_deal_data,copy_data,120)
# time=1
day=150
# sum=0
# 用于后期的自动调参测试
# while day<=200:
#     print "k:",k
#     print "day:",day
#     getpre_SVM(origal_data, copy_data, day)
#     day+=20
# acuracy=getpre_RFR(origal_data,origal_data, day)
# print"Random Forest：",acuracy
getpre_SVM(copy_data, copy_data, day)
# KDJstrategy(origal_data,70)
#
# while day<=200:
#     print "sum:",sum
#     print "day:",day
#     starttime = datetime.datetime.now()
#     print "计时开始！", starttime
#
#
#     while(time<=5):
#         acuracy=getpre_SVM(sma_deal_data,copy_data, day)
#         sum+=acuracy
#         time+=1
#
#     acuracy = sum / 5
#
#     print "5次平均后的准确率是;", acuracy
#     sum=0
#     time=1
#     endtime=datetime.datetime.now()
#     times=endtime-starttime
#
#
#     print "计时结束，花费时间是：",times
#     day+=20



    #     while(time<=5):
#         acuracy=getpre_RFR(origal_data,origal_data, day)
#         sum+=acuracy
#         time+=1
#
#     acuracy = sum / 5
#
#     print "5次平均后的准确率是;", acuracy
#

#
#






# getpre_RFR(sma_deal_data,100)
# getpre_GBDT(sma_deal_data,200)




# 画股票数据图
    # _, ax = plt.subplots(figsize = [16,8])
# plt.scatter(range(days), y, c='k', label='y')
# plt.hold('on')
#
# plt.scatter(range(days, days + 1), y_real[-1], c='r', label='y_real')
# plt.hold('on')
#
# plt.plot(range(days + 1), y_pre, c='b', label='y_pre')
# plt.xlabel('day')
# plt.ylabel('price')
# plt.title('Support Vector Regression')
# plt.legend()
# plt.show()