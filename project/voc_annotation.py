import os
import random
import xml.etree.ElementTree as ET
import config

import numpy as np

from utils.utils import get_classes

# 指定annocatoin_mode用于指定文件运行时的计算内容
# annotation_mode 为0代表整个标签处理过程,包括获得VOCdevkit/VOC2007/Imagesets里面的txt以及训练用的2e07_train.txt 2ee7_val.tx
# annotation mode 为1代表获得VOCdevkit/VOC2007/ImageSets里面的txt
# annotation mode 为2代表获得训练用的2007_train.txt
annotation_mode = 0

# 用于生成2007_train.txt，2007_val.txt的日标信息
# 与训练集预测所用的classes_path一致即可
# 如果生成的2007 train.txt里面没有目标信息
# 那么是因为classes没有设定正确
# 仅在annotation model为0和2的时候有效
classes_path = config.vocClassesPath

# trainval_percent用于定义指定(训练集+验证集)与测试集的比例，默认情况(训练集+验证集): 测试机 = 9:1
# train_percent用于定义指定(训练集+验证集)与测试集的比例，默认情况(训练集+验证集): 测试机 = 9:1
# 仅在annotation_model为0和1的时候有效
trainval_percnet = 0.9
train_percnet = 0.9

# 指向VOC数据集的文件夹
# 默认指向根目录下的VOC数据集
VOCdevkit_path = config.vocDevkitPath

VOCdevkit_sets = [("2007", "train"), ("2007", "val")]
classes, _ = get_classes(classes_path)

# 统计目标数量
photo_nums = np.zeros(len(VOCdevkit_sets))
nums = np.zeros(len(classes))


def convert_annotation(year, image_id, list_file):
    in_file = open(
        os.path.join(VOCdevkit_path, "VOC%s/Annotations/%s.xml" % (year, image_id)),
        encoding="utf-8",
    )
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter("object"):
        difficult = 0
        if obj.find("difficult") != None:
            difficult = obj.find("difficult").text
        cls = obj.find("name").text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find("bndbox")
        b = (
            int(float(xmlbox.find("xmin").text)),
            int(float(xmlbox.find("ymin").text)),
            int(float(xmlbox.find("xmax").text)),
            int(float(xmlbox.find("ymax").text)),
        )
        list_file.write(" " + ",".join([str(a) for a in b]) + "," + str(cls_id))
        nums[classes.index(cls)] = nums[classes.index(cls)] + 1
    # in_file = open(os.path.join(VOCdevkit_path,'VOC%s/Annotations/%s.xml'%(year,image_id)),encoding='utf-8')
    # tree = ET.parse(in_file)
    # root = tree.getroot()
    #
    #
    # for obj in root.iter('object'):
    #     difficult = 0
    #     if obj.find('difficult')!=None:
    #         difficult = obj.find('difficult').text
    #     cls = obj.find('name').text
    #     if cls not in classes or int(difficult)==1:
    #         continue
    #     cls_id = classes.index(cls)
    #     xmlbox = obj.find('bndbox')
    #     b = (int(float(xmlbox.find('xmin').text)),
    #          int(float(xmlbox.find('ymin').text)),
    #          int(float(xmlbox.find('xmax').text)),
    #          int(float(xmlbox.find('ymax').text)))
    #     list_file.write(" "+",".join([str(a) for a in b])+','+str(cls_id))
    #     nums[classes.index(cls)] = nums[classes.index(cls)]+1


if __name__ == "__main__":
    random.seed(0)
    if " " in os.path.abspath(VOCdevkit_path):
        raise ValueError("数据集存放的文件夹路径与图片名称中不能有空格")
    if annotation_mode == 0 or annotation_mode == 1:
        print("Genertate txt in ImageSets")
        xmlfilePath = os.path.join(VOCdevkit_path, "VOC2007/Annotations")
        saveBasePath = os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Main")
        temp_xml = os.listdir(xmlfilePath)
        total_xml = []
        for xml in temp_xml:
            if xml.endswith(".xml"):
                total_xml.append(xml)
        num = len(total_xml)
        list = range(num)
        tv = int(num * trainval_percnet)
        tr = int(tv * train_percnet)
        trainval = random.sample(list, tv)
        train = random.sample(trainval, tr)

        print("train and val size", tv)
        print("train size", tr)
        ftrainval = open(os.path.join(saveBasePath, "trainval.txt"), "w")
        ftest = open(os.path.join(saveBasePath, "test.txt"), "w")
        ftrain = open(os.path.join(saveBasePath, "train.txt"), "w")
        fval = open(os.path.join(saveBasePath, "val.txt"), "w")

        for i in list:
            name = total_xml[i][:-4] + "\n"
            if i in trainval:
                ftrainval.write(name)
                if i in train:
                    ftrain.write(name)
                else:
                    fval.write(name)
            else:
                ftest.write(name)

        ftrainval.close()  # 流执行完成后，需要进行关闭作，就像水龙头一样，避免资源浪费
        ftrain.close()
        fval.close()
        ftest.close()
        print("Generate txt in IMageSets done.")

if annotation_mode == 0 or annotation_mode == 2:
    print("Generate 2007_train.txt and 2007_val.txt for train.")
    type_index = 0
    for year, image_set in VOCdevkit_sets:
        image_ids = (
            open(
                os.path.join(
                    VOCdevkit_path, "VOC%s/ImageSets/Main/%s.txt" % (year, image_set)
                ),
                encoding="utf-8",
            )
            .read()
            .strip()
            .split()
        )
        list_file = open("%s_%s.txt" % (year, image_set), "w", encoding="utf-8")
        for image_id in image_ids:
            list_file.write(
                "%s/VOC%s/JPEGImages/%s.jpg"
                % (os.path.abspath(VOCdevkit_path), year, image_id)
            )  # 移除 \n 确保路径格式正确
            convert_annotation(year, image_id, list_file)
            list_file.write("\n")  # 在每行结束后添加换行符，确保格式正确
        photo_nums[type_index] = len(image_ids)
        type_index += 1
        list_file.close()
    print("Generate 2007_train.txt and 2007_val.txt for train done.")

    # if annotation_mode == 0 or annotation_mode == 2:
    #     print("Generate 2007_train.txt and 2007_val.txt for train.")
    #     type_index = 0
    #     for year, image_set in VOCdevkit_sets:
    #         image_ids = open(os.path.join(VOCdevkit_path, 'VOC%s\ImageSets\Main\%s.txt' % (year, image_set)), encoding='utf-8').read().strip().split()
    #         list_file = open('%s_%s.txt' % (year, image_set), 'w', encoding='utf-8')
    #         for image_id in image_ids:
    #             list_file.write('%s\VOC%s\JPEGImages\%s.jpg\n' % (os.path.abspath(VOCdevkit_path), year, image_id))
    #             convert_annotation(year, image_id, list_file)
    #         photo_nums[type_index] = len(image_ids)
    #         type_index += 1
    #         list_file.close()
    #     print("Generate 2007_train.txt and 2007_val.txt for train done.")

    def printTable(List1, List2):
        for i in range(len(List1[0])):
            print("|", end=" ")
            for j in range(len(List1)):
                print(List1[j][i].rjust(int(List2[j])), end=" ")
                print("|", end=" ")
            print()

    str_nums = [str(int(x)) for x in nums]
    tableData = [classes, str_nums]
    colWidths = [0] * len(tableData)
    len1 = 0
    for i in range(len(tableData)):
        if len(tableData[i]) > colWidths[i]:
            colWidths[i] = len(tableData[i])
    printTable(tableData, colWidths)

    if photo_nums[0] <= 500:
        print(
            "训练集数量小于500，属于较小的数据量，注意设置较大的训练 Epoch 以满足足够的梯度下降次数 Step "
        )
    if np.sum(nums) == 0:
        print(
            "在数据集中并未获得任何目标，注意修改 classes_path 对应的自己的数据集，并且保证标签名字正确，否则训练集会没有任何效果"
        )
