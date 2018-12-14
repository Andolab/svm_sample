import numpy as np
from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
from tqdm import tqdm
import csv

def make_array(arr, nums):
    array = []
    for i in nums:
        array.append(arr[i])
    return array

def cross_val(keys, data, label, method):
    print(method)
    n_fold = 10 # 何分割の交差検証にするか
    k_fold = cross_validation.KFold(n=len(data), n_folds=n_fold, shuffle=True, random_state=42)# 配列の添字を指定個数のグループに分割
    params = {'C':[10, 100, 1000], 'gamma':[0.001, 0.0001], 'kernel':['rbf']} # gridsearchの変化範囲
    svc = GridSearchCV(SVC(), params, cv=n_fold, n_jobs=-1)
    accuracy  = []
    precision = []
    recall    = []
    f_score   = []
    misslist  = ['correct' for i in range(len(label))]
    csvlist   = []
    pbar = tqdm(total=n_fold)
    for train, test in k_fold:
        train_data  = make_array(data, train)
        train_label = make_array(label, train)
        test_data   = make_array(data, test)
        test_label  = make_array(label, test)
        svc.fit(train_data, train_label)             # 学習
        pred_label = svc.predict(test_data)        # 予測
        for i in range(0,len(test_label)):
            if test_label[i] != pred_label[i]:
                misslist[test[i]] = 'miss'
        #print('正解    : {}'.format(np.array(test_label)))
        #print('予測    : {}'.format(np.array(pred_label)))
        #print('正解率: {}'.format(accuracy_score(test_label, pred_label)))
        #print(classification_report(test_label, pred_label))
        accuracy.append(accuracy_score(test_label, pred_label))
        pre, rec, fsc, sup = precision_recall_fscore_support(test_label, pred_label)
        precision.append(pre)
        recall.append(rec)
        f_score.append(fsc)
        pbar.update(1)
    pbar.close()
    csvlist.append(calc_result(accuracy, precision, recall, f_score))
    write_result(misslist,csvlist,method,keys)

# 結果計算
def calc_result(acc, pre, rec, fsc):
    result = []
    for n in [1,0]:
        accuracy = []
        precision = []
        recall = []
        f_score = []
        for i in range(len(pre)):
            accuracy.append(acc[i])
            precision.append(pre[i][n])
            recall.append(rec[i][n])
            f_score.append(fsc[i][n])
        result.append(np.mean(precision))
        result.append(np.mean(recall))
        result.append(np.mean(f_score))
    result.append(np.mean(accuracy))
    return result

# CSVファイルに結果書き込み
def write_result(miss,score,method,keys):
    with open('miss_result.csv', 'a', encoding='UTF-8', newline='') as f:
        writer = csv.writer(f)
        miss.insert(0,method)
        writer.writerow(miss)

    with open('result.csv', 'a', encoding='UTF-8', newline='') as f:
        writer = csv.writer(f)
        for data in score:
            data.insert(0,method)
            writer.writerow(data)
