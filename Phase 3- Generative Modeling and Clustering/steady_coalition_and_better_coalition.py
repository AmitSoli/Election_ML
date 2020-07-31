import numpy as np
import itertools as it
from sklearn.model_selection import StratifiedKFold
import random
from sklearn.naive_bayes import GaussianNB
from sklearn.mixture import GaussianMixture
from sklearn.metrics import calinski_harabaz_score, silhouette_score, homogeneity_score

########################################################################################################################
# Path, input and output, functions definition
########################################################################################################################
base_path = 'D:\\Courses\\ML\\Homeworks\\ML_4\\'
input_file_path = base_path + 'prepossessing\\'

x_train = np.load(input_file_path + 'x_train.npy')
y_train = np.load(input_file_path + 'y_train.npy')
x_valid = np.load(input_file_path + 'x_valid.npy')
y_valid = np.load(input_file_path + 'y_valid.npy')
x_test = np.load(input_file_path + 'x_test.npy')
y_test = np.load(input_file_path + 'y_test.npy')

parties = list(np.unique(y_test))
num_parties = len(parties)

########################################################################################################################
# Get all possible coalitions in train
########################################################################################################################

voting_dist = {}
for vote in y_train:
    if vote not in voting_dist:
        voting_dist[vote] = 1
    else:
        voting_dist[vote] += 1

print(voting_dist)
print('------------------------')

parties_comb = []
for i in range(1, num_parties):
    parties_comb += list(it.combinations(parties, i))

coalitions = []
for comb in parties_comb:
    sum_voters = 0
    for p in comb:
        sum_voters += voting_dist[p]

    if sum_voters/(len(y_test)+len(y_valid)+len(y_train)) > 0.51:
        coalitions.append(comb)

########################################################################################################################
# Find the number of clusters which are dense and well separated
########################################################################################################################
k_fold = StratifiedKFold(n_splits=10)

for n_components in range(int(num_parties/2), 2*num_parties):
    gm = GaussianMixture(n_components=n_components)
    cv_calinski_harabaz_score_sum = 0
    cv_silhouette_score_sum = 0
    num_cv = 0
    for train_i, test_i in k_fold.split(x_train, y_train):
        gm.fit(x_train[train_i])
        y_clusters = gm.predict(x_train[train_i])
        cv_calinski_harabaz_score_sum += calinski_harabaz_score(x_train[train_i], y_clusters)
        cv_silhouette_score_sum += silhouette_score(x_train[train_i], y_clusters)
        num_cv += 1

    print("Number of components:", n_components)
    print("Calinski Harabaz score:", cv_calinski_harabaz_score_sum/num_cv)
    print("Silhouette score:", cv_silhouette_score_sum/num_cv)
    print("--------------------")


########################################################################################################################
# Get data for steady coalition
########################################################################################################################
n_components = 9
gm = GaussianMixture(n_components=n_components)
gm.fit(x_train)
y_clusters = gm.predict(x_train)
############################################################################
# Create histogram for each cluster
############################################################################
for i in range(n_components):
    clusters_hist = {}
    for j in range(len(y_clusters)):
        if y_clusters[j] == i:
            if y_train[j] not in clusters_hist:
                clusters_hist[y_train[j]] = 1
            else:
                clusters_hist[y_train[j]] += 1
    total_size = 0
    for p in clusters_hist.keys():
        total_size += clusters_hist[p]

    print('-----------------------------------')
    print(clusters_hist)
    print('Cluster ', i)
    print('Size = ', total_size)
    for p in clusters_hist.keys():
        print(p, ' = ', round(100*clusters_hist[p]/total_size), '%')

############################################################################
# Create distance matrix of the clusters
############################################################################
dist = np.zeros((len(gm.means_), len(gm.means_)))
max_dist = -1
for i in range(len(gm.means_)):
    for j in range(len(gm.means_)):
        dist_vec = []
        for k in range(len(gm.means_[0])):
            dist_vec.append(gm.means_[i, k] - gm.means_[j, k])
        dist[i, j] = np.linalg.norm(dist_vec)
        if dist[i, j] > max_dist:
            max_dist = dist[i, j]

for i in range(len(gm.means_)):
    for j in range(len(gm.means_)):
        dist[i, j] = dist[i, j]/max_dist

print(dist)

############################################################################
# Print the homogeneity score for each possible coalition
############################################################################
for c in [['Purples', 'Greens', 'Browns', 'Whites', 'Pinks'],
          ['Purples', 'Greens', 'Browns', 'Whites'],
          ['Purples', 'Greens', 'Browns', 'Pinks'],
          ['Purples', 'Greens', 'Browns']]:
    y_train_c = np.copy(y_train)
    for i in range(len(y_train_c)):
        if y_train_c[i] in c:
            y_train_c[i] = "Coal"

        else:
            y_train_c[i] = "Opo"

    y_predict = gm.predict(x_train)
    score = homogeneity_score(y_train_c, y_predict)
    print(score, c)

########################################################################################################################
# Apply on test
########################################################################################################################
y_clusters = gm.predict(x_test)

############################################################################
# Create histogram for each cluster
############################################################################
for i in range(n_components):
    clusters_hist = {}
    for j in range(len(y_clusters)):
        if y_clusters[j] == i:
            if y_test[j] not in clusters_hist:
                clusters_hist[y_test[j]] = 1
            else:
                clusters_hist[y_test[j]] += 1
    total_size = 0
    for p in clusters_hist.keys():
        total_size += clusters_hist[p]

    print('-----------------------------------')
    print(clusters_hist)
    print('Cluster ', i)
    print('Size = ', total_size)
    for p in clusters_hist.keys():
        print(p, ' = ', round(100*clusters_hist[p]/total_size), '%')

############################################################################
# Create distance matrix of the clusters
############################################################################
dist = np.zeros((len(gm.means_), len(gm.means_)))
max_dist = -1
for i in range(len(gm.means_)):
    for j in range(len(gm.means_)):
        dist_vec = []
        for k in range(len(gm.means_[0])):
            dist_vec.append(gm.means_[i, k] - gm.means_[j, k])
        dist[i, j] = np.linalg.norm(dist_vec)
        if dist[i, j] > max_dist:
            max_dist = dist[i, j]

for i in range(len(gm.means_)):
    for j in range(len(gm.means_)):
        dist[i, j] = dist[i, j]/max_dist

print(dist)

############################################################################
# Print the homogeneity score for each possible coalition
############################################################################
for c in [['Purples', 'Greens', 'Browns', 'Whites', 'Pinks'],
          ['Purples', 'Greens', 'Browns', 'Whites'],
          ['Purples', 'Greens', 'Browns', 'Pinks'],
          ['Purples', 'Greens', 'Browns']]:
    y_test_c = np.copy(y_test)
    for i in range(len(y_test_c)):
        if y_test_c[i] in c:
            y_test_c[i] = "Coal"

        else:
            y_test_c[i] = "Opo"

    y_predict = gm.predict(x_test)
    score = homogeneity_score(y_test_c, y_predict)
    print(score, c)

########################################################################################################################
# Strengthen the suggested coalition and construct a stronger coalition
########################################################################################################################
coalition = ['Purples', 'Greens', 'Browns', 'Whites']
print('------------------------------------------------------------------------------------')
print('Strengthen the suggested coalition and construct a stronger coalition')
print('------------------------------------------------------------------------------------')
print('-----------------------------------------------------')
print('ORIGINAL coalition opposition ration in clusters')
print('-----------------------------------------------------')
###########################################################################
# Print coalition opposition ration in clusters
###########################################################################
org_coalition_size = 0
org_n_pink = 0
y_clusters = gm.predict(x_test)
y_test_col_opo = np.copy(y_test)
for i in range(len(y_test_col_opo)):
    if y_test_col_opo[i] in coalition:
        y_test_col_opo[i] = "Coal"
        org_coalition_size += 1

    else:
        if y_test_col_opo[i] == 'Pinks':
            org_n_pink+=1
        y_test_col_opo[i] = "Opo"

############################################################################
# Create histogram for each cluster
############################################################################
for i in range(n_components):
    clusters_hist = {}
    for j in range(len(y_clusters)):
        if y_clusters[j] == i:
            if y_test_col_opo[j] not in clusters_hist:
                clusters_hist[y_test_col_opo[j]] = 1
            else:
                clusters_hist[y_test_col_opo[j]] += 1
    total_size = 0
    for p in clusters_hist.keys():
        total_size += clusters_hist[p]

    print('-----------------------------------')
    print(clusters_hist)
    print('Cluster ', i)
    print('Size = ', total_size)
    for p in clusters_hist.keys():
        print(p, ' = ', round(100*clusters_hist[p]/total_size), '%')

score = homogeneity_score(y_test_col_opo, y_predict)
print('-----------------------------------')
print('Homogeneity score:', score)
print('Coalition size', org_coalition_size)
print('Pinks party size', org_n_pink)
print('-----------------------------------')
###########################################################################
#
###########################################################################
features = ['Number_of_valued_Kneset_members', 'Yearly_IncomeK', 'Overall_happiness_score',
            'Avg_Satisfaction_with_previous_vote', 'Will_vote_only_large_party',
            'Garden_sqr_meter_per_person_in_residancy_area', 'Weighted_education_rank',
            'Number_of_valued_Kneset_membersNull', 'Yearly_IncomeKNull', 'Overall_happiness_scoreNull',
            'Avg_Satisfaction_with_previous_voteNull', 'Most_Important_IssueNull', 'Will_vote_only_large_partyNull',
            'Garden_sqr_meter_per_person_in_residancy_areaNull', 'Weighted_education_rankNull', 'Education',
            'Environment', 'Financial', 'Foreign_Affairs', 'Healthcare', 'Military', 'Other', 'Social']


def get_d_index(f_name):
    i = 0
    for f in features:
        if f == f_name:
            return i
        i += 1
    print(f_name,'f_name error')
    exit(2)


normal_list = ['Avg_Satisfaction_with_previous_vote', 'Garden_sqr_meter_per_person_in_residancy_area', 'Yearly_IncomeK',
               'Weighted_education_rank', 'Overall_happiness_score']

binary_list = ['Education', 'Environment', 'Financial', 'Foreign_Affairs', 'Healthcare', 'Military', 'Other', 'Social']

bounded_list = ['Number_of_valued_Kneset_members', 'Yearly_IncomeK', 'Overall_happiness_score',
                'Avg_Satisfaction_with_previous_vote', 'Will_vote_only_large_party',
                'Garden_sqr_meter_per_person_in_residancy_area', 'Weighted_education_rank']

changes = [('Number_of_valued_Kneset_members', 'up'), ('Weighted_education_rank', 'down'), ('Garden_sqr_meter_per_person_in_residancy_area', 'up')]


x_test_c = np.copy(x_test)
for f, up_down in changes:
    f_i = get_d_index(f)
    if f in normal_list:
        if up_down == 'up':
            x_test_c[:, f_i] += 0.5
        else:
            x_test_c[:, f_i] -= 0.5
    elif f in bounded_list:
        if up_down == 'up':
            x_test_c[:, f_i] += 1
        else:
            x_test_c[:, f_i] -= 1
            x_test_c[:, f_i] /= 2
    else:
        for row in range(x_test_c.shape[1]):
            if up_down == 'up':
                if random.random() < 0.8:
                    for j in range(len(features)):
                        if features[j] in binary_list:
                            x_test_c[row, j] = -1
                            x_test_c[row, f_i] = 1
            else:
                    if random.random() < 0.8:
                        for j in range(len(features)):
                            if features[j] in binary_list:
                                x_test_c[row, j] = 1
                                x_test_c[row, f_i] = -1


########################################################################################################################
# Apply on test
########################################################################################################################
cls = GaussianNB()
cls.fit(x_train, y_train)
new_y_test = cls.predict(x_test_c)
print('-----------------------------------------------------')
print('NEW coalition opposition ration in clusters')
print('-----------------------------------------------------')
###########################################################################
# Print coalition opposition ration in clusters after change
###########################################################################
new_coalition_size = 0
n_pink = 0
y_clusters = gm.predict(x_test_c)
y_test_col_opo = np.copy(new_y_test)
for i in range(len(y_test_col_opo)):
    if y_test_col_opo[i] in coalition:
        y_test_col_opo[i] = "Coal"
        new_coalition_size+=1
    else:
        if y_test_col_opo[i]== 'Pinks':
            n_pink+=1
        y_test_col_opo[i] = "Opo"

############################################################################
# Create histogram for each cluster
############################################################################
for i in range(n_components):
    clusters_hist = {}
    for j in range(len(y_clusters)):
        if y_clusters[j] == i:
            if y_test_col_opo[j] not in clusters_hist:
                clusters_hist[y_test_col_opo[j]] = 1
            else:
                clusters_hist[y_test_col_opo[j]] += 1
    total_size = 0
    for p in clusters_hist.keys():
        total_size += clusters_hist[p]

    print('-----------------------------------')
    print(clusters_hist)
    print('Cluster ', i)
    print('Size = ', total_size)
    for p in clusters_hist.keys():
        print(p, ' = ', round(100 * clusters_hist[p] / total_size), '%')

score = homogeneity_score(y_test_col_opo, y_predict)
print('-----------------------------------')
print('Homogeneity score:', score)
print('Coalition size', new_coalition_size)
print('Pinks party size', n_pink)
print('-----------------------------------')
