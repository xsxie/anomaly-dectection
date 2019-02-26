#coding = UTF-8
#This python file uses the following encoding: utf-8

"""
	Case study
	Function: (1)计算p_value
	          (2)计算credibility 和 confidence
	          计算新日志与每一个模板的得分
	          计算该得分在每一个模板的排名
	          计算p_value
	          计算Credibility和Confidence
	          多进程计算
	Author: Xie xueshuo; Xiao xuhang
	Date: 2019.02.21
"""

import Levenshtein
import csv
import pandas as pd
import numpy as np
import time
import multiprocessing
import heapq


"""计算两个字符串的相似性，Non-conformal measure，score function"""
def string_similar(s1, s2):
    sim = Levenshtein.distance(s1, s2)
    return sim

"""将一个多维列表按指定长度分成多个列表"""
def list_of_groups(init_list, children_list_len):
    list_of_groups = zip(*(iter(init_list),) *children_list_len)
    end_list = [list(i) for i in list_of_groups]
    count = len(init_list) % children_list_len
    end_list.append(init_list[-count:]) if count !=0 else end_list
    return end_list


"""计算训练日志消息的content部分与对应模板的score"""
def training_score(Input_filepath, Input_filepath2, Output_filepath):

    with open(Input_filepath, 'rb') as csvfile:
        reader = csv.DictReader(csvfile)
        column1 = [row['Content'] for row in reader]
    with open(Input_filepath, 'rb') as csvfile:
        reader = csv.DictReader(csvfile)
        column2 = [row['EventTemplate'] for row in reader]
    with open(Input_filepath, 'rb') as csvfile:
        reader = csv.DictReader(csvfile)
        column3 = [row['EventId'] for row in reader]
    distList = []
    for i in range(len(column1)):
        sim = string_similar(column1[i], column2[i])
        distList.append(sim)
    with open(Input_filepath2, 'rb') as csvfile:
        reader = csv.DictReader(csvfile)
        column4 = [row['EventId'] for row in reader]
    distMat = np.zeros((len(column3), len(column4)))
    for j in range(len(column4)):
        for k in range(len(column3)):
            if column4[j] ==column3[k]:
                distMat[k][j] = distList[k]
    np.savetxt(Output_filepath, distMat, delimiter=',')
    return Output_filepath

"""计算detection日志消息的content部分与每一个模板的score"""
def detection_score(Input_filepath1,Input_filepath2,Output_filepath):
    with open(Input_filepath1, 'rb') as csvfile:
        reader = csv.DictReader(csvfile)
        column1 = [row['Content'] for row in reader]
    with open(Input_filepath2, 'rb') as csvfile:
        reader = csv.DictReader(csvfile)
        column2 = [row['EventTemplate'] for row in reader]
    L1 = len(column1)
    L2 = len(column2)
    distList=[]
    distMat=np.zeros((L1,L2))
    for i in range(len(column1)):
        for j in range(len(column2)):
            sim = string_similar(column1[i], column2[j])
            distMat[i][j] = sim
            distList.append(sim)
    distArray=np.array(distList)
    np.savetxt(Output_filepath, distMat, delimiter=',')
    return Output_filepath

"""将score按照模板个数进行排序"""
def score_sort(Intput_filepath, Output_filepath):

    df = pd.DataFrame(pd.read_csv(Intput_filepath, header=None, index_col=False))
    row_length = df.iloc[:, 0].size
    column_length = df.columns.size
    distList = []
    distMat = np.zeros((row_length, column_length))
    for j in range(column_length):
        column = df[j]
        column_sort = column.sort_values(ascending=False)
        score_array = np.array(column_sort)
        for k in range(len(score_array)):
            distMat[k][j] = score_array[k]
    distList.append(score_array)
    distArray = np.array(distList)
    np.savetxt(Output_filepath, distMat, delimiter=',')
    return Output_filepath

"""p_value 计算函数"""
def p_value(value,list, number):
    count = 0.0
    for i in range(len(list)):
        if list[i] >= value:
            count = count + 1.0
    p_value = float(count/number)
    return p_value

"""计算每条日志与每个模板的p_value"""
def p_value_all(Input_filepath, Input_filepath1, Input_filepath2, Output_filepath, name):
    df = pd.DataFrame(pd.read_csv(Input_filepath, header=None, index_col=False))
    df1 = pd.DataFrame(pd.read_csv(Input_filepath1, header=None, index_col=False))
    row_length = df.iloc[:, 0].size
    column_length = df.columns.size
    distList = []
    distMat = np.zeros((row_length, column_length))
    number = column_length * row_length
    count = 0.0
    for j in range(column_length):
        for i in range(row_length):
            column = df1[j]
            start = time.time()

            with open(Input_filepath2, 'rb') as csvfile:
                reader = csv.DictReader(csvfile)
                column1 = [row['Occurrences'] for row in reader]
            number_occurr = int(column1[j])

            Pvalue = p_value(df.loc[i][j], column, number_occurr)
            end = time.time()
            count = count + 1.0
            print "%s Time：%f" % (name, (end-start))
            print "%s, %s/%s, Process: %f" % (name, count, number, (count/number))
            distMat[i][j] = Pvalue
        distList.append(Pvalue)
    distArray = np.array(distList)
    np.savetxt(Output_filepath, distMat, delimiter=',')
    return Output_filepath

"""计算每条日志对应的confidence和credibility"""
def confidence(Input_filepath, Output_filepath):
    df = pd.DataFrame(pd.read_csv(Input_filepath, header=None, index_col=False))
    row_length = df.iloc[:, 0].size
    distMat = np.zeros((row_length, 2))
    for i in range(row_length):
        row_list = df.loc[i].tolist()
        row_sen = heapq.nlargest(2, row_list)
        cre = row_sen[0]
        con = 1.0 - row_sen[1]
        distMat[i][0] = cre
        distMat[i][1] = con
    np.savetxt(Output_filepath, distMat, delimiter=',')
    return Output_filepath

def main(name_algorithm):

    Input_filepath_structured = './logparser/%s_result/training.log_structured.csv' % name_algorithm
    Input_filepath_template = './logparser/%s_result/training.log_templates.csv' % name_algorithm
    Input_filepath_structured_detection = './logparser/dectection_small.log_structured.csv'

    Output_filepath_score = './score/training/%s_score.csv' % name_algorithm
    Output_filepath_score_snort = './score/snort/%s_score_snort.csv' % name_algorithm
    Output_filepath_dectection_score = './score/dectection/%s_dectection_score.csv' % name_algorithm

    Output_filepath_p_value = './p_value/%s_p_value.csv' % name_algorithm
    Output_filepath_cre_con = './p_value/%s_cre_con.csv' % name_algorithm

    print name_algorithm

    """training"""
    print "training"
    training_score(Input_filepath_structured, Input_filepath_template, Output_filepath_score) #计算训练日志集中的日志与算法模板集中的模板的不一致得分
    score_sort(Output_filepath_score, Output_filepath_score_snort) #将得分按照列从小到大排序，提升查找效率

    """dectection"""
    print "dectection"
    detection_score(Input_filepath_structured_detection, Input_filepath_template, Output_filepath_dectection_score) #计算检测日志集中的日志与算法模板集中的模板的不一致得分

    """conformal prediction"""
    print "conformal prediction"
    p_value_all(Output_filepath_dectection_score, Output_filepath_score_snort, Input_filepath_template, Output_filepath_p_value, name_algorithm) #计算检测日志集中的日志与算法模板集中的模板的p_value
    confidence(Output_filepath_p_value, Output_filepath_cre_con) #计算算法对每一条日志的Credibility和Confidence

    print name_algorithm

if __name__ == '__main__':

    namelist_algorithm = ['LogCluster', 'SHISO', 'LogSig', 'LFA',
                          'Drain', 'Spell', 'AEL', 'IPLoM', 'MoLFI',
                          'Lenma', 'SLCT', 'LogMine']

    """多进程并行计算"""
    pool = multiprocessing.Pool(processes=6)
    for i in xrange(7):
        for name_algorithm in namelist_algorithm:
            pool.apply_async(main, (name_algorithm,))
    pool.close()
    pool.join()