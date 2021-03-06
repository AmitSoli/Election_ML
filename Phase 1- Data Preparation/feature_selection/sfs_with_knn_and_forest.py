import numpy as np
from sfs import sfs, get_selected_feature_set
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
# SFS
########################################################################################################################
forest = RandomForestClassifier(n_estimators=3, min_samples_split=13, max_features=None, criterion='entropy')
knn = KNeighborsClassifier(n_neighbors=5)

print('--------------------- SFS --------------------------')
sfs_with_knn = sfs(x_train, y_train, 25, knn, cv_score_5_fold)
sfs_with_forest = sfs(x_train, y_train, 25, forest, cv_score_5_fold)

sfs_classifiers = [(knn, 'knn', sfs_with_knn), (forest, 'forest', sfs_with_forest)]
for c_sfs in sfs_classifiers:
    curr_sfs_indexes = c_sfs[2]
    print('25 sfs with', c_sfs[1], curr_sfs_indexes)
    for c in sfs_classifiers:
        for i in range(1, len(curr_sfs_indexes)+1):
            x_train_sfs = get_selected_feature_set(x_train, curr_sfs_indexes[:i])
            x_valid_sfs = get_selected_feature_set(x_valid, curr_sfs_indexes[:i])

            c[0].fit(x_train_sfs, y_train)
            y_predict = c[0].predict(x_valid_sfs)
            print('first', i, 'accuracy of', c[1], metrics.accuracy_score(y_valid, y_predict))
        print('')
