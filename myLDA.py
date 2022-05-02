import os
import jieba
import glob
import re
import numpy as np
import math
from pylab import *
import  matplotlib as mpl
import matplotlib.pyplot as plt
#支持中文
mpl.rcParams['font.sans-serif'] = ['SimHei']
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

def read_data():
    catalog = "inf.txt"
    with open(catalog, "r") as f:
        all_files = f.readline().split(",")
        print(all_files)

    train_files_dict = dict()
    test_files_dict = dict()
    # test 200行
    # train 50000行
    train_num = 500
    min_len = 20
    test_num = 60
    # test_num = 10
    test_length = 20
    # c=0
    for name in all_files:
        # c+=1
        with open(os.path.join(name + ".txt"), "r", encoding='ansi') as f:
            file_read = f.readlines()
            para_num = len(file_read)
            choice_index = []
            train_text = ""
            i = 0
            train_choice = list(range(para_num))
            while True:
                j=0
                tmp = np.random.randint(0, len(train_choice))
                train = train_choice[tmp]
                line = file_read[train]
                line = re.sub('\s', '', line)
                line = re.sub('[\u0000-\u4DFF]', '', line)
                line = re.sub('[\u9FA6-\uFFFF]', '', line)
                if len(line) <= min_len:
                    choice_index.append(train)
                    train_choice.pop(tmp)
                    if len(train_choice) <= 10:
                        break
                    continue
                train_text += line
                choice_index.append(train)
                train_choice.pop(tmp)
                i += 1
                if i >= train_num or len(train_choice)<=10:
                    break

            train_files_dict[name] = train_text
            i = 0
            test_choice = train_choice
            while True:
                tmp = np.random.randint(0, len(test_choice))
                test = test_choice[tmp]
                if test in choice_index:
                    test_choice.pop(tmp)
                    if test_choice == []:
                        break
                    continue
                test_line = ""
                line = file_read[test]
                line = re.sub('\s', '', line)
                line = re.sub('[\u0000-\u4DFF]', '', line)
                line = re.sub('[\u9FA6-\uFFFF]', '', line)
                if len(line) <= min_len:
                    choice_index.append(test)
                    test_choice.pop(tmp)
                    continue
                test_line += line
                if not name in test_files_dict.keys():
                    test_files_dict[name] = [test_line]
                else:
                    test_files_dict[name].append(test_line)
                i += 1
                if i >= test_num or test_choice == []:
                    break
        # if c>=2:
        #    break
    return train_files_dict, test_files_dict


def main():
    train_texts_dict, test_texts_dict = read_data()

    train_terms_list = []
    train_terms_dict = dict()
    name_list = []
    for name in train_texts_dict.keys():
        text = train_texts_dict[name]
        seg_list = list(jieba.cut(text, cut_all=False))  # 使用精确模式

        terms_string = ""
        for term in seg_list:
            terms_string += term + " "
        train_terms_dict[name] = terms_string
        train_terms_list.append(terms_string)
        name_list.append(name)
        print("finished to calculate the train " + name + " text")

    test_terms_dict = dict()
    for name in test_texts_dict.keys():
        text_list = test_texts_dict[name]
        for text in text_list:
            seg_list = list(jieba.cut(text, cut_all=False))  # 使用精确模式
            terms_string = ""
            for term in seg_list:
                terms_string += term + " "
            if not name in test_terms_dict.keys():
                test_terms_dict[name] = [terms_string]
            else:
                test_terms_dict[name].append(terms_string)
        print("finished to calculate the test " + name + " text")

    # calculate terms vector
    cnt_vector = CountVectorizer(max_features=500)
    cnt_tf_train = cnt_vector.fit_transform(train_terms_list)

    lda = LatentDirichletAllocation(n_components=35,  # 主题个数
                                    max_iter=3000,  # EM算法的最大迭代次数
                                    # learning_method='online',
                                    # learning_offset=50.,  # 仅仅在算法使用online时有意义，取值要大于1。用来减小前面训练样本批次对最终模型的影响
                                    random_state=0)
    target = lda.fit_transform(cnt_tf_train)
    print("target: ", target.shape)
    test_correct_number = 0
    test_wrong_number = 0
    for name in test_terms_dict.keys():
        terms_list = test_terms_dict[name]
        for terms in terms_list:
            cnt_tf_file = cnt_vector.transform([terms])

            res = lda.transform(cnt_tf_file)
            min_index = ((target - res.repeat(target.shape[0], axis=0)) ** 2).sum(axis=1).argmin()

            if name == name_list[min_index]:
                test_correct_number += 1
            else:
                test_wrong_number += 1

    print("test accuracy: ", test_correct_number / (test_correct_number + test_wrong_number))
    bar_width = 1
    xtick = np.arange(7) + 1
    f,ax = plt.subplots(4, 4)
    plt.subplots_adjust(wspace=1,hspace=0.8)
    for i in range(16):
        if i<4:
            k=0
        elif i<8:
            k=1
        elif i<12:
            k=2
        else:
            k=3
        ax[k,i%4].bar(np.arange(target.shape[1]) + bar_width / 2, target[i][:], bar_width, alpha=1)
        ax[k,i%4].set_xlim(0, 35)
        ax[k,i%4].set_ylim(0, 1)
        ax[k,i%4].set_title(name_list[i]+'主题分布',{'family' : 'SimHei','weight' : 'normal','size' : 8})
        ax[k,i%4].set_xticks([0,35])
    plt.show()


if __name__ == "__main__":
    # execute only if run as a script
    main()
