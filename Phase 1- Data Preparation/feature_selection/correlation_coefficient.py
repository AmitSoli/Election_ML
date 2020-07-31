import numpy as np
from sfs import get_selected_feature_set
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from relief import relief
from sklearn.svm import SVC

########################################################################################################################
# Path, input and output
########################################################################################################################
base_path = 'C:\\Users\\Aviel\\PycharmProjects\\ML_2\\'
input_file_path = base_path + 'prepossessing\\'
output_path = base_path + 'f_select\\'

x_train = np.load(input_file_path + 'x_train.npy')
y_train = np.load(input_file_path + 'y_train.npy')


########################################################################################################################
# Correlation coefficient
########################################################################################################################
color_dict = {}
i = 0
for color in y_train:
    if color not in color_dict:
        color_dict[color] = i
        i += 1
color_v = y_train
for i in range(len(color_v)):
    color_v[i] = color_dict[color_v[i]]

cor_list = []
for col in range(x_train.shape[1]):
    cor_mat_vote = np.corrcoef(x_train[:,col].astype(float), y_train.astype(float), rowvar=False)
    cor_list.append((col, cor_mat_vote[1][0], abs(cor_mat_vote[1][0])))

sorted_cor_raw = sorted(cor_list, key=lambda tup: tup[2])
sorted_cor = []
for element in sorted_cor_raw:
    sorted_cor.append((element[0], element[1]))
sorted_cor.reverse()
print(sorted_cor)
