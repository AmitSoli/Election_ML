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
x_valid = np.load(input_file_path + 'x_valid.npy')
y_valid = np.load(input_file_path + 'y_valid.npy')


def cv_score_5_fold(cls, x, y):
    k_fold = StratifiedKFold(n_splits=5)
    k_fold_score = cross_val_score(cls, x, y, cv=k_fold)
    return sum(k_fold_score) / len(k_fold_score)


########################################################################################################################
# Relief
########################################################################################################################
forest = RandomForestClassifier(n_estimators=3, min_samples_split=13, max_features=None, criterion='entropy')
knn = KNeighborsClassifier(n_neighbors=5)
svm = SVC()

print('-------------------- Relief ------------------------')
sorted_by_relief = relief(int(x_train.shape[0]*0.3), x_train, y_train)
print('all sorted by relief, best first', sorted_by_relief)

relief_classifiers = [(knn, 'knn'), (forest, 'forest'), (svm, 'svm')]
for c in relief_classifiers:
    for i in range(1, len(sorted_by_relief)):
        x_train_sfs = get_selected_feature_set(x_train, sorted_by_relief[:i])
        x_valid_sfs = get_selected_feature_set(x_valid, sorted_by_relief[:i])

        c[0].fit(x_train_sfs, y_train)
        y_predict = c[0].predict(x_valid_sfs)
        print('first', i, 'accuracy of', c[1], metrics.accuracy_score(y_valid, y_predict))
    print('')
