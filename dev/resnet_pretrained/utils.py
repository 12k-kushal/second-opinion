#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 22:20:38 2021

@author: spathak
"""

from torch.utils.data import Dataset, DataLoader, Sampler
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torchvision import transforms
import pandas as pd
import os
import torch
import math
from skimage import io
from PIL import Image
from PIL import ImageOps
import numpy as np
import sys
import glob
import random
import matplotlib.pyplot as plt
# import cv2
# import seaborn as sns
from sklearn import metrics, utils
from collections import Counter
import operator

groundtruth_dic = {'BENIGN': 0, 'MALIGNANT': 1}
inverted_groundtruth_dic = {0: 'benign', 1: 'malignant'}
views_allowed = ['LCC', 'LMLO', 'RCC', 'RMLO']
optimizer_params_dic = {'.mlo': 0, '.cc': 1, '_left.attention': 2, '_right.attention': 3, '_both.attention': 4}

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
cluster_data_path_prefix = './../../data/processed/'  # your image path


class MyCrop:
    """Randomly crop the sides."""

    def __init__(self, left=100, right=100, top=100, bottom=100):
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom

    def __call__(self, x):
        width, height = x.size
        size_left = random.randint(0, self.left)
        size_right = random.randint(width - self.right, width)
        size_top = random.randint(0, self.top)
        size_bottom = random.randint(height - self.bottom, height)
        x = TF.crop(x, size_top, size_left, size_bottom, size_right)
        return x


class MyGammaCorrection:
    def __init__(self, factor=0.2):
        self.lb = 1 - factor
        self.ub = 1 + factor

    def __call__(self, x):
        gamma = random.uniform(self.lb, self.ub)
        return TF.adjust_gamma(x, gamma)


class MyHorizontalFlip:
    """Flip horizontally."""

    def __init__(self):
        pass

    def __call__(self, x, breast_side):
        if breast_side == 'L':
            return TF.hflip(x)
        else:
            return x


class MyPadding:
    def __init__(self, breast_side, max_height, max_width, height, width):
        self.breast_side = breast_side
        self.max_height = max_height
        self.max_width = max_width
        self.height = height
        self.width = width

    def __call__(self, img):
        print(img.shape)
        print(self.max_height - self.height)
        if self.breast_side == 'L':
            image_padded = F.pad(img, (0, self.max_width - self.width, 0, self.max_height - self.height, 0, 0),
                                 'constant', 0)
        elif self.breast_side == 'R':
            image_padded = F.pad(img, (self.max_width - self.width, 0, 0, self.max_height - self.height, 0, 0),
                                 'constant', 0)
        print(image_padded.shape)
        return image_padded


class MyPaddingLongerSide:
    def __init__(self):
        self.max_height = 1600
        self.max_width = 1600

    def __call__(self, img):  # ,breast_side):
        width = img.size[0]
        height = img.size[1]
        if height < self.max_height:
            diff = self.max_height - height
            img = TF.pad(img, (0, math.floor(diff / 2), 0, math.ceil(diff / 2)), 0, 'constant')
        if width < self.max_width:
            diff = self.max_width - width
            # if breast_side=='L':
            #    img=TF.pad(img,(0,0,diff,0),0,'constant')
            # elif breast_side=='R':
            img = TF.pad(img, (diff, 0, 0, 0), 0, 'constant')
        return img


class BreastCancerDataset_generator(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, df, modality, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = df
        # print("df index:",self.df.index.values)
        self.modality = modality
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        breast_side = []
        flag = 0
        data = self.df.iloc[idx]
        studyuid_path = cluster_data_path_prefix + str(data["subject_id"]) + "_1-1.png"
        #         print(studyuid_path)
        img = Image.open(studyuid_path)

        if self.transform:
            img = self.transform(img)
        # print("after transformation:",img.shape)
        img = img[0, :, :]
        img = img.unsqueeze(0).unsqueeze(0)
        if flag == 0:
            image_tensor = img
            flag = 1
        else:
            image_tensor = torch.cat((image_tensor, img), 0)
        return idx, image_tensor.shape[0], image_tensor, torch.tensor(groundtruth_dic[data['class']])


def MyCollate(batch):
    i = 0
    index = []
    target = []
    bag_size = []
    for item in batch:
        if i == 0:
            data = batch[i][2]
            # views_names=item[4]
        else:
            data = torch.cat((data, batch[i][2]), dim=0)
            # views_names.extend(item[4])
        index.append(item[0])
        target.append(item[3])
        bag_size.append(item[1])
        i += 1
    index = torch.LongTensor(index)
    target = torch.LongTensor(target)
    return [index, bag_size, data, target]  # , views_names]


# def collect_images(acc_num_path,series_list):
#     #collect images for the model
#     breast_side=[]
#     image_read_list=[]
# #     views_saved=[]
# #     series_list.sort()
# #     for series in series_list:
# #         series_path=acc_num_path+'/'+series
# #         img_list=os.listdir(series_path)
# #         for image in img_list:
# #             if '_processed.png' in image:

#     img_path=acc_num_path
#     #img=cv2.imread(img_path)
#     img=Image.open(img_path)
#     #img=io.imread(img_path)
#     #img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
# #     series_des=series.split('_')[0].upper().replace(" ","")
# #     if series_des in views_allowed and series_des not in views_saved:
# #         views_saved.append(series_des)
#         #print(series_des,img.size)
#     image_read_list.append(img)
#     breast_side.append(series[0])
#     return image_read_list, breast_side, views_saved

def data_augmentation_train(mean, std_dev):
    preprocess_train = transforms.Compose([
        MyCrop(),
        transforms.Pad(100),
        transforms.RandomRotation(3),
        transforms.ColorJitter(brightness=0.20, contrast=0.20),
        transforms.RandomAdjustSharpness(sharpness_factor=0.20),
        MyGammaCorrection(0.20),
        MyPaddingLongerSide(),
        transforms.Resize((1600, 1600)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std_dev)
    ])
    return preprocess_train


def fetch_groundtruth(df, acc_num, modality):
    col_names = df.filter(regex='Acc_' + modality + '.*').columns.tolist()
    acc_num = int(acc_num)
    groundtruth = df.loc[(df[col_names].astype('Int64') == acc_num).any(axis=1)]['final_gt']
    if groundtruth.empty:
        groundtruth = -1
    else:
        groundtruth = groundtruth.item()
    return groundtruth


def freeze_layers(model, layer_keyword):
    for name, param in model.named_parameters():
        # print(name)
        if layer_keyword in name:
            param.requires_grad = False
            # print(name,param.requires_grad)
    return model


def views_distribution(df):
    views_dic = {}
    views_dic_allowed = {}
    single_views_dic = {}
    total = df.shape[0]
    for k in range(total):
        if k % 5 == 0:
            print(str(k) + "/" + str(total))
        study_folder = cluster_data_path_prefix + str(df.iloc[k]['StoragePath']) + '/' + str(
            df.iloc[k]['StudyInstanceUID']) + '_' + str(df.iloc[k]['AccessionNum'])
        series_list = os.listdir(study_folder)
        views_list = []
        views_list_allowed = []
        for series in series_list:
            view_name = series.split('_')[0]
            if view_name not in views_list:
                views_list.append(view_name)
            if view_name in views_allowed and view_name not in views_list_allowed:
                views_list_allowed.append(view_name)
            single_views_dic[view_name] = single_views_dic.get(view_name, 0) + 1

        views_list.sort()
        views_joined = '+'.join(views_list)
        # print(views_joined)
        views_dic[views_joined] = views_dic.get(views_joined, 0) + 1
        # print(views_dic)
        views_list_allowed.sort()
        views_joined_allowed = '+'.join(views_list_allowed)
        # print(views_joined_allowed)
        views_dic_allowed[views_joined_allowed] = views_dic_allowed.get(views_joined_allowed, 0) + 1
        # print(views_dic_allowed)
        df.loc[k, ['Views']] = views_joined_allowed
    print(df)
    pd.DataFrame.from_dict(views_dic, orient='index').to_excel('views_dic.xlsx')
    pd.DataFrame.from_dict(views_dic_allowed, orient='index').to_excel('views_dic_allowed.xlsx')
    pd.DataFrame.from_dict(single_views_dic, orient='index').to_excel('single_views_dic.xlsx')
    df.to_csv('/home/spathak/multiview_mammogram/input_data/MG_training_files_studyUID_accessionNum_final.csv', sep=';',
              na_rep='NULL', index=False)


def plot(filename):
    df = pd.read_excel(filename).sort_values(by=['Count'], ascending=False)
    print(df['Views'].tolist())
    print(df['Count'].tolist())
    plt.figure(figsize=(5, 5))
    plt.bar(df['Views'].tolist(), df['Count'].tolist())
    plt.xticks(rotation=45, ha='right')
    plt.savefig('view_distribution.png', bbox_inches='tight')


def stratified_class_count(df):
    class_count = df.groupby(by=['class']).size()
    return class_count


def class_distribution_weightedloss(df):
    df_groundtruth = df['Groundtruth'].map(groundtruth_dic)
    class_weight = utils.class_weight.compute_class_weight(class_weight='balanced', classes=np.array([0, 1]),
                                                           y=df_groundtruth)
    print(dict(Counter(df_groundtruth)))
    print(class_weight)
    return torch.tensor(class_weight, dtype=torch.float32).to(device)


def performance_metrics(conf_mat, y_true, y_pred, y_prob):
    prec = metrics.precision_score(y_true, y_pred, pos_label=1)
    rec = metrics.recall_score(y_true, y_pred)  # sensitivity, TPR
    spec = conf_mat[0, 0] / np.sum(conf_mat[0, :])  # TNR
    f1 = metrics.f1_score(y_true, y_pred)
    acc = metrics.accuracy_score(y_true, y_pred)
    bal_acc = (rec + spec) / 2
    cohen_kappa = metrics.cohen_kappa_score(y_true, y_pred)
    auc = metrics.roc_auc_score(y_true, y_prob)

    '''prec=conf_mat[1,1]/np.sum(conf_mat[:,1])
    rec=conf_mat[1,1]/np.sum(conf_mat[1,:])
    spec=conf_mat[0,0]/np.sum(conf_mat[0,:])
    f1=(2*prec*rec)/(prec+rec)
    acc=(conf_mat[0,0]+conf_mat[1,1])/np.sum(conf_mat)
    bal_acc=(rec+spec)/2'''
    each_model_metrics = [prec, rec, spec, f1, acc, bal_acc, cohen_kappa, auc]
    return each_model_metrics


def confusion_matrix_norm_func(conf_mat, fig_name, class_name):
    # class_name=['W','N1','N2','N3','REM']
    conf_mat_norm = np.empty((conf_mat.shape[0], conf_mat.shape[1]))
    # conf_mat=confusion_matrix(y_true, y_pred)
    for i in range(conf_mat.shape[0]):
        conf_mat_norm[i, :] = conf_mat[i, :] / sum(conf_mat[i, :])
    # print(conf_mat_norm)
    print_confusion_matrix(conf_mat_norm, class_name, fig_name)


def print_confusion_matrix(conf_mat_norm, class_names, fig_name, figsize=(2, 2), fontsize=5):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
        
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    # sns.set()
    # grid_kws = {"height_ratios": (.9, .05), "hspace": .3}
    fig, ax = plt.subplots(figsize=figsize)
    # cbar_ax = fig.add_axes([.93, 0.1, 0.05, 0.77])
    # fig = plt.figure(figsize=figsize)
    heatmap = sns.heatmap(
        yticklabels=class_names,
        xticklabels=class_names,
        data=conf_mat_norm,
        ax=ax,
        cmap='YlGnBu',
        cbar=False,
        # cbar_ax=cbar_ax,
        annot=True,
        annot_kws={'size': fontsize},
        fmt=".2f",
        square=True
        # linewidths=0.75
    )
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    ax.set_ylabel('True label', labelpad=0, fontsize=fontsize)
    ax.set_xlabel('Predicted label', labelpad=0, fontsize=fontsize)
    # cbar_ax.tick_params(labelsize=fontsize)
    # ax.get_yaxis().set_visible(False)
    # plt.tight_layout()
    # plt.show()
    ax.set_title(fig_name)
    fig.savefig(fig_name + '.pdf', format='pdf', bbox_inches='tight')

# conf_mat=np.array([[775,52],[170,166]])
# confusion_matrix_norm_func(conf_mat,'sota_MaxWelling_variableview_MG',class_name=['benign','malignant'])
