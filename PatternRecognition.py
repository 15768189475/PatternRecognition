import cv2
import os
import time
import random
from numpy import *


def train_number(): return 30  # 用于训练的样本个数，（0， 55）


def load_image(route):  # 使用opencv来加载图片，并进行前期处理
    print("load_image...")
    for i in range(10):
        if i != 9:
            for filename in os.listdir(route + '0' + str(i + 1)):  # 'D:/img/Sample009'
                img = cv2.imread(route + '0' + str(i + 1) + "/" + filename, 0)
                img = cv2.resize(img, (120, 90))
                img = array(img)
                # print(img)
                # time.sleep(100)
                img = reshape(img, (10800, 1))
                # img = img / 255  # 特殊方式的二值化，针对此数据集有效
                # for i in range(10800):  # 反色
                #     if img[i] == 0: img[i] = 1
                #     if img[i] == 1: img[i] = 0
                img = row_stack((img, matrix([1])))  # 增广
                array_img.append(img)
        else:
            for filename in os.listdir(route + str(i + 1)):  # 'D:/img/Sample010'
                img = cv2.imread(route + str(i + 1) + "/" + filename, 0)
                img = cv2.resize(img, (120, 90))
                img = array(img)
                img = reshape(img, (10800, 1))
                # img = img / 255  # 特殊方式的二值化，针对此数据集有效
                # for i in range(10800):  # 反色
                #     if img[i] == 0: img[i] = 1
                #     if img[i] == 1: img[i] = 0
                img = row_stack((img, matrix([1])))  # 增广
                array_img.append(img)
    print("load image successful")


def init():
    random_w = []
    for i in range(10):
        for j in range(10801):
            if i == 0:
                random_w.append(random.random())  # 随机初始化w
            else:
                random_w[j] = random.random()
        random_w = array(random_w)
        random_w = reshape(random_w, (10801, 1))
        w.append(random_w)


def train():
    print("start train")
    right_number = 0
    success_flag = 0
    cycle = 0
    max_right = 0
    while success_flag == 0:
        cycle += 1
        print(cycle, end=" ")  # 输出显示迭代次数
        print(max_right)
        max_right = 0
        for i in range(train_number()):
            if success_flag == 1: break
            for j in range(10):
                img_now = array_img[i + j * 55]  # 取样本
                flag = 0
                for k in range(10):
                    if j != k:
                        if dot(transpose(w[j]), img_now) <= dot(transpose(w[k]), img_now):  # 分类错误了
                            w[j] += img_now  # 修正权向量
                            for m in range(10):
                                if m != j:
                                    w[m] -= img_now  # 修正权向量
                            break
                        else:
                            flag += 1
                if flag == 9:  # 此样本分类正确
                    right_number += 1
                else:
                    if right_number > max_right:
                        max_right = right_number
                    right_number = 0
                if right_number == train_number() * 12:  # 所有的测试样本都分类正确了
                    success_flag = 1
                    break


def test():
    print("start test")
    fail = 0
    for i in range(train_number(), 55):  # 将剩下的样本代入进去测试
        for j in range(10):
            img_now = array_img[i + j * 55]
            for k in range(10):
                if j != k:
                    if dot(transpose(w[j]), img_now) <= dot(transpose(w[k]), img_now):  # 错分类了
                        fail += 1
                        break
    all_number = (55 - train_number()) * 10
    success_rate = (all_number - fail) / all_number
    print("success rate:", end=" ")
    print(success_rate)


if __name__ == '__main__':
    array_img = []  # 加载图片的数组
    w = []  # 权值矩阵
    load_image('D:/img/Sample0')  # 加载图片
    init()  # 初始化
    train()  # 训练到收敛
    test()  # 测试准确率
