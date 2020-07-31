import numpy as np
from sfs_random import sfs_random, get_selected_feature_set
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

########################################################################################################################
# Path, input and output
########################################################################################################################
base_path = 'C:\\Users\\Aviel\\PycharmProjects\\ML_2\\'
input_file_path = base_path + 'prepossessing\\'
output_path = base_path + 'f_select\\'

x_train = np.load(input_file_path + 'x_train.npy')
y_train = np.load(input_file_path + 'y_train.npy')
x_valid = np.load(input_file_path + 'x_valid.npy')
y_valid = np.load(input_file_path + 'y_valid.npy')


def cv_score_5_fold(cls, x, y):
    k_fold = StratifiedKFold(n_splits=5)
    k_fold_score = cross_val_score(cls, x, y, cv=k_fold)
    return sum(k_fold_score) / len(k_fold_score)


########################################################################################################################
# SFS Random
########################################################################################################################
forest = RandomForestClassifier(n_estimators=3, min_samples_leaf=5, max_features=None, criterion='entropy')
knn = KNeighborsClassifier(n_neighbors=3)

print('--------------------- SFS Random --------------------------')

sfs_random_with_knn = sfs_random(x_train, y_train, 16, knn, cv_score_5_fold)
sfs_random_classifiers = [(knn, 'knn'), (forest, 'forest')]
print('16 random sfs with knn', sfs_random_with_knn)
for c in sfs_random_classifiers:
    x_train_sfs_random = get_selected_feature_set(x_train, sfs_random_with_knn)
    x_valid_sfs_random = get_selected_feature_set(x_valid, sfs_random_with_knn)

    c[0].fit(x_train_sfs_random, y_train)
    y_predict = c[0].predict(x_valid_sfs_random)
    print('accuracy of', c[1], metrics.accuracy_score(y_valid, y_predict))
print('')
