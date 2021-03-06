import numpy as np
import pandas as pd
import itertools as it
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
import matplotlib.pyplot as plt
import itertools
import operator
import random
import warnings

########################################################################################################################
# Table of contents:
#   0.0 Path, input and output, functions definition.
#   1.0 Predict distribution and winner.
#   2.0 Predict votes:
#       2.1 Find the best parameters for each classifier: id3, knn, forest, voting.
#       2.2 The Model Selection.
#       2.3 Selected model statistics.
#       2.4 Save predictions CSV file.
#   3.0 Using the selected model to predict distribution.
#   4.0 Find Factor - The most likely to change which party will win the elections.
########################################################################################################################
########################################################################################################################
# 0.0 Path, input and output, functions definition
########################################################################################################################
base_path = 'D:\\Courses\\ML\\Homeworks\\ML_3\\'
input_file_path = base_path + 'prepossessing\\'
output_path = base_path + 'f_select\\'

x_train = np.load(input_file_path + 'x_train.npy')
y_train = np.load(input_file_path + 'y_train.npy')
x_valid = np.load(input_file_path + 'x_valid.npy')
y_valid = np.load(input_file_path + 'y_valid.npy')
x_test = np.load(input_file_path + 'x_test.npy')
y_test = np.load(input_file_path + 'y_test.npy')

parties = list(np.unique(y_test))
num_parties = len(parties)


def scorer(estimator, x, y):
    y_pred = estimator.predict(x)
    cm = metrics.confusion_matrix(y, y_pred, labels=parties)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    min_precision = 1.1
    for i in range(num_parties):
        p_precision = cm[i, i] / sum(cm[:, i])
        if p_precision < min_precision:
            min_precision = p_precision
    return min_precision


def cv_5_fold(estimator, x, y):
    k_fold = StratifiedKFold(n_splits=5)
    k_fold_score = cross_val_score(estimator, x, y, cv=k_fold, scoring=scorer)
    return sum(k_fold_score) / len(k_fold_score)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


warnings.filterwarnings('ignore')
########################################################################################################################
#   1.0 Predict distribution and winner.
########################################################################################################################
print('###############################################################################################################')
print('# Predict distribution and winner')
print('###############################################################################################################')

voting_dist = {}
for vote in y_train:
    if vote not in voting_dist:
        voting_dist[vote] = 1
    else:
        voting_dist[vote] += 1

for vote in y_valid:
    if vote not in voting_dist:
        voting_dist[vote] = 1
    else:
        voting_dist[vote] += 1

for vote in y_test:
    if vote not in voting_dist:
        voting_dist[vote] = 1
    else:
        voting_dist[vote] += 1

predicted_winner = max(voting_dist.items(), key=operator.itemgetter(1))[0]

sum_votes = sum(voting_dist.values())
voting_dist.update({key: voting_dist[key] / sum_votes for key in voting_dist.keys()})

print('The predicted distribution:', voting_dist)
print('The predicted winner:', predicted_winner)


########################################################################################################################
# 2.0 Predict votes:
#       2.1 Find the best parameters for each classifier: id3, knn, forest, voting.
#       2.2 The Model Selection
#       2.3 Selected model statistics
#       2.4 Save predictions CSV file
########################################################################################################################
print('###############################################################################################################')
print('# Predict votes')
print('###############################################################################################################')


########################################################################################################################
# 2.1 Find the best parameters for each classifier: id3, knn, forest, voting.
########################################################################################################################
print('----------------------------------------------------------------------------------------------------------')
print('id3')
print('----------------------------------------------------------------------------------------------------------')
id3_max_cv_score_cls = None
id3_max_cv_score_val = -1
for criterion in ['entropy', 'gini']:
    for min_samples_split in range(3, 15):
        for min_samples_leaf in range(1, 9):
            id3 = DecisionTreeClassifier(criterion=criterion, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
            cv_score = cv_5_fold(id3, x_train, y_train)
            cv_score = round(cv_score, 3)

            id3.fit(x_train, y_train)
            y_predict = id3.predict(x_train)

            if cv_score > id3_max_cv_score_val:
                print(criterion, 'criterion,', min_samples_split, 'min_samples_split,', min_samples_leaf, 'min_samples_leaf:', cv_score)
                id3_max_cv_score_cls = id3
                id3_max_cv_score_val = cv_score


print('----------------------------------------------------------------------------------------------------------')
print('knn')
print('----------------------------------------------------------------------------------------------------------')
knn_max_cv_score_cls = None
knn_max_cv_score_val = -1
for n_neighbors in range(1, 10):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    cv_score = cv_5_fold(knn, x_train, y_train)
    cv_score = round(cv_score, 3)

    knn.fit(x_train, y_train)
    y_predict = knn.predict(x_train)

    if cv_score > knn_max_cv_score_val:
        print(n_neighbors, 'neighbors:', cv_score)
        knn_max_cv_score_cls = knn
        knn_max_cv_score_val = cv_score


print('----------------------------------------------------------------------------------------------------------')
print('forest')
print('----------------------------------------------------------------------------------------------------------')
forest_max_cv_score_cls = None
forest_max_cv_score_val = -1
for min_samples_split in range(3, 10, 2):
    for min_samples_leaf in range(1, 6, 2):
        for n_estimators in range(3, 15, 2):
            for criterion in ['entropy', 'gini']:
                forest = RandomForestClassifier(n_estimators=n_estimators,
                                                min_samples_split=min_samples_split,
                                                min_samples_leaf=min_samples_leaf,
                                                max_features=None,
                                                criterion=criterion)
                cv_score = cv_5_fold(forest, x_train, y_train)
                cv_score = round(cv_score, 3)

                forest.fit(x_train, y_train)
                y_predict = forest.predict(x_train)

                if cv_score > forest_max_cv_score_val:

                    print(min_samples_split, 'min_samples_split',
                          min_samples_leaf, 'min_samples_leaf',
                          n_estimators, 'n_estimators',
                          criterion, 'criterion:',
                          cv_score)
                    forest_max_cv_score_cls = forest
                    forest_max_cv_score_val = cv_score


print('----------------------------------------------------------------------------------------------------------')
print('svm')
print('----------------------------------------------------------------------------------------------------------')
svm_max_cv_score_cls = None
svm_max_cv_score_val = -1
for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
    for degree in range(1, 11):
        if kernel != 'poly' and degree != 3:
            break

        svm = SVC(kernel=kernel, degree=degree)
        cv_score = cv_5_fold(svm, x_train, y_train)
        cv_score = round(cv_score, 3)

        svm.fit(x_train, y_train)
        y_predict = svm.predict(x_train)

        if cv_score > svm_max_cv_score_val:
            print(kernel, 'kernel', degree, 'degree', cv_score)
            svm_max_cv_score_cls = svm
            svm_max_cv_score_val = cv_score


print('----------------------------------------------------------------------------------------------------------')
print('voting')
print('----------------------------------------------------------------------------------------------------------')
weights_dict = {'id3': id3_max_cv_score_val, 'knn': knn_max_cv_score_val, 'forest': forest_max_cv_score_val, 'svm': svm_max_cv_score_val}
estimators = [('id3', id3_max_cv_score_cls), ('knn', knn_max_cv_score_cls), ('forest', forest_max_cv_score_cls), ('svm', svm_max_cv_score_cls)]
estimators_combinations = []
for i in range(3, 5):
    estimators_combinations += list(it.combinations(estimators, i))

voting_max_cv_score_cls = None
voting_max_cv_score_val = -1
for comb in estimators_combinations:
    weights = []
    for cls in comb:
        weights.append(weights_dict[cls[0]])

    voting = VotingClassifier(estimators=comb, voting='hard', weights=weights)
    cv_score = cv_5_fold(voting, x_train, y_train)
    cv_score = round(cv_score, 3)

    voting.fit(x_train, y_train)
    y_predict = voting.predict(x_train)

    if cv_score > voting_max_cv_score_val:
        print(comb)
        voting_max_cv_score_cls = voting
        voting_max_cv_score_val = cv_score


########################################################################################################################
# 2.2 The Model Selection.
########################################################################################################################
models = [id3_max_cv_score_cls, knn_max_cv_score_cls, forest_max_cv_score_cls, svm_max_cv_score_cls, voting_max_cv_score_cls]

selected_model_cls = None
selected_model_val = -1
for model in models:
    model.fit(x_train, y_train)
    score = scorer(model, x_valid, y_valid)
    if score > selected_model_val:
        selected_model_cls = model
        selected_model_val = score

selected_model_cls.fit(x_train, y_train)
selected_model_predictions = selected_model_cls.predict(x_test)


########################################################################################################################
# 2.3 Selected model statistics.
########################################################################################################################
print('----------------------------------------------------------------------------------------------------------')
print('The Selected Model Is', selected_model_cls)
print('Accuracy:', metrics.accuracy_score(y_test, selected_model_predictions))
print('Error:', 1 - metrics.accuracy_score(y_test, selected_model_predictions))

cm = metrics.confusion_matrix(y_test, selected_model_predictions, labels=parties)
np.set_printoptions(precision=2)


plt.figure()
plot_confusion_matrix(cm, classes=parties, title='Confusion matrix, without normalization')

plt.figure()
plot_confusion_matrix(cm, classes=parties, normalize=True, title='Normalized confusion matrix')
plt.show()
print('----------------------------------------------------------------------------------------------------------')


########################################################################################################################
# 2.4 Save predictions CSV file.
########################################################################################################################
features = ['Number_of_valued_Kneset_members', 'Yearly_IncomeK', 'Overall_happiness_score',
            'Avg_Satisfaction_with_previous_vote', 'Will_vote_only_large_party',
            'Garden_sqr_meter_per_person_in_residancy_area', 'Weighted_education_rank',
            'Number_of_valued_Kneset_membersNull', 'Yearly_IncomeKNull', 'Overall_happiness_scoreNull',
            'Avg_Satisfaction_with_previous_voteNull', 'Most_Important_IssueNull', 'Will_vote_only_large_partyNull',
            'Garden_sqr_meter_per_person_in_residancy_areaNull', 'Weighted_education_rankNull', 'Education',
            'Environment', 'Financial', 'Foreign_Affairs', 'Healthcare', 'Military', 'Other', 'Social']


index_to_feature_name = {}
for i in range(len(features)):
    index_to_feature_name[i] = features[i]


df = pd.DataFrame(x_test)
df['Vote'] = selected_model_predictions
cols = df.columns.tolist()
df = df[[cols[-1]] + cols[:-1]]
df.rename(columns=index_to_feature_name, inplace=True)
df.to_csv(base_path + 'csv_dada_sets\\test_predicted_df.csv', sep=',', index=0)


########################################################################################################################
# 3.0 Using the selected model to predict distribution.
########################################################################################################################
print('###############################################################################################################')
print('# Using the selected model to predict distribution')
print('###############################################################################################################')
voting_dist = {}
for predict in selected_model_predictions:
    if predict not in voting_dist:
        voting_dist[predict] = 1
    else:
        voting_dist[predict] += 1

predicted_winner = max(voting_dist.items(), key=operator.itemgetter(1))[0]

sum_votes = sum(voting_dist.values())
voting_dist.update({key: voting_dist[key] / sum_votes for key in voting_dist.keys()})

print('The predicted distribution:', voting_dist)
print('The predicted winner:', predicted_winner)
print('Using predict votes classifier to predict distribution accuracy:', metrics.accuracy_score(y_test, selected_model_predictions))


########################################################################################################################
# 4.0 Find Factor - The most likely to change which party will win the elections.
########################################################################################################################
print('###############################################################################################################')
print('# Find Factor - most likely to change which party will win the elections')
print('###############################################################################################################')
score_up_feature = []
score_down_feature = []
normal_list = ['Avg_Satisfaction_with_previous_vote', 'Garden_sqr_meter_per_person_in_residancy_area', 'Yearly_IncomeK',
               'Weighted_education_rank', 'Overall_happiness_score']
binary_list = ['Education', 'Environment', 'Financial', 'Foreign_Affairs', 'Healthcare', 'Military', 'Other', 'Social']
bounded_list = ['Number_of_valued_Kneset_members', 'Yearly_IncomeK', 'Overall_happiness_score',
                'Avg_Satisfaction_with_previous_vote', 'Will_vote_only_large_party',
                'Garden_sqr_meter_per_person_in_residancy_area', 'Weighted_education_rank',
                'Number_of_valued_Kneset_membersNull', 'Yearly_IncomeKNull', 'Overall_happiness_scoreNull',
                'Avg_Satisfaction_with_previous_voteNull', 'Most_Important_IssueNull', 'Will_vote_only_large_partyNull',
                'Garden_sqr_meter_per_person_in_residancy_areaNull', 'Weighted_education_rankNull']

features_indexes = range(len(features))
feature_comb = []

for i in range(1, 4):
    feature_comb += list(it.combinations(features_indexes, i))

for comb in feature_comb:
    score_up_feature = np.copy(x_test)
    score_down_feature = np.copy(x_test)

    for i in comb:
        if features[i] in normal_list:
            score_up_feature[:, i] += 0.5
            score_down_feature[:, i] -= 0.5
        elif features[i] in bounded_list:
            score_up_feature[:, i] += 1
            score_up_feature[:, i] /= 2
            score_down_feature[:, i] -= 1
            score_down_feature[:, i] /= 2
        else:
            for row in range(score_up_feature.shape[1]):
                if random.random() < 0.8:
                    for j in range(len(features)):
                        if features[j] in binary_list:
                            score_up_feature[row, j] = -1
                    score_up_feature[row, i] = 1

                if random.random() < 0.8:
                    for j in range(len(features)):
                        if features[j] in binary_list:
                            score_down_feature[row, j] = 1
                    score_down_feature[row, i] = -1

    y_predict = selected_model_cls.predict(score_up_feature)
    voting_dist = {}
    for predict in y_predict:
        if predict not in voting_dist:
            voting_dist[predict] = 1
        else:
            voting_dist[predict] += 1

    winner = max(voting_dist.items(), key=operator.itemgetter(1))[0]

    if winner != 'Purples':
        print('Features:')
        for f in comb:
            print(' * ', features[f])

        print('Were scaled UP and the new winner prediction is', winner)
        print('-------------------------------------------------------------------------')

    y_predict = selected_model_cls.predict(score_down_feature)
    voting_dist = {}
    for predict in y_predict:
        if predict not in voting_dist:
            voting_dist[predict] = 1
        else:
            voting_dist[predict] += 1

    winner = max(voting_dist.items(), key=operator.itemgetter(1))[0]

    if winner != 'Purples':
        if len(comb) == 1 or 4 not in comb:
            print('Features:')
            for f in comb:
                print(' * ', features[f])

            print('Were scaled DOWN and the new winner prediction is', winner)
            print('-------------------------------------------------------------------------')


########################################################################################################################
