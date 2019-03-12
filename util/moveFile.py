# -*- coding: utf-8 -*-
# 用于将不同类别的文件夹下的图片搬到某个文件夹，并在文件前面加上目录名字:
# 比如caltech256，里面有256个文件夹，每个文件夹放的都是一类，用movefiles.py可以生成一个database的文件夹，
# 这个文件夹把databaseClassified的图片都搬到里面来，并且生成待查询的图片queryImgs.txt和databaseClasses.txt
# python movefiles.py

import os
import shutil

# query_number_percent = 0.5 # 设置每类拿百分之多少出来作为查询

directory = "database"  # 设置新路径
databaseClasses = 'MEN'

if not os.path.exists(directory):
    os.makedirs(directory)

newImgDBPath = os.path.abspath(directory)

cnt = 0
for root, dirs, files in os.walk(databaseClasses):
    for i, str_each_folder in enumerate(dirs):
        # we get the directory path
        str_the_path = '/'.join([root, str_each_folder])
        files_number = len((os.listdir(str_the_path)))  # 子目录下文件数目
        # list all the files using directory path
        for ind, str_each_file in enumerate(os.listdir(str_the_path)):
            # now add the new one
            str_img_path = '/'.join([str_the_path, str_each_file])
            for j, str_img_name in enumerate(os.listdir(str_img_path)):
                str_new_name = 'input_{03d}_groundtruth'.format(cnt)
                str_old_name = '/'.join([str_img_path, str_img_name])
                str_new_name = '/'.join([newImgDBPath, str_new_name])
                shutil.copy2(str_old_name, str_new_name)  # 拷贝原文件到设置的新目录下
                cnt = cnt + 1
            # if ind in index:
            #     g.writelines('%s\n' % str_new_name)
            # full path for both files
            # now rename using the two above strings and the full path to the files
            # os.rename(str_old_name, str_new_name) # 搬运原文件到设置的新目录下

        #  we can print the folder name so we know that all files in the folder are done
        print('%s, %d images' % (str_each_folder, files_number))
        # f.writelines('%s %d\n' % ('{0:03}'.format(i+1)+'_'+str_each_folder, files_number))
# g.close
# f.close
