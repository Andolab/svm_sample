#coding=utf-8
from sklearn import datasets
from cross_validation import cross_val
import csv

def exec_svm(data, labels, method, keys):
    cross_val(keys, data,labels, method)

if __name__ == '__main__':
    # データ作成
    digits  = datasets.load_digits()                            # sklearnのsample dataset
    vectors = digits.data                                       # データのベクトル表現の配列
    keys    = ["data-"+str(i) for i in range(0,len(vectors))]   # データ番号の配列(実際は必要ないが、データごとに正誤を見たいから)
    labels  = digits.target                                     # データの正解ラベルの配列

    # 実行結果保存ファイルの初期化
    with open('miss_result.csv','w',encoding='UTF-8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['method']+keys)

    with open('result.csv', 'w', encoding='UTF-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['method','p-pre','p-rec','p-f', 'n-pre','n-rec','n-f','acc'])

    #print(vectors[0])
    #print(keys[0])
    #print(labels[0])

    # SVM実行
    exec_svm(vectors, labels, 'sample', keys)
