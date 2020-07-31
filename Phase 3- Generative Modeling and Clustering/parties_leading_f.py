import numpy as np
import operator
import random
from sklearn.naive_bayes import GaussianNB

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

features_indexes = range(len(features))

cls = GaussianNB()
cls.fit(x_train, y_train)


########################################################################################################################
# Changing features values in test set and its effect's
########################################################################################################################
original_voting_dist = {}
for vote in y_test:
    if vote not in original_voting_dist:
        original_voting_dist[vote] = 1
    else:
        original_voting_dist[vote] += 1

print('-----------------------------------------------------')
print('Test set voting distribution')
print('-----------------------------------------------------')
print(original_voting_dist)
print('-----------------------------------------------------')

print('-----------------------------------------------------')
print('Finding new winner')
print('-----------------------------------------------------')

parties_dict = {}
for p in parties:
    parties_dict[p] = []

for f_i in features_indexes:
    score_up_feature = np.copy(x_test)
    score_down_feature = np.copy(x_test)

    if features[f_i] in normal_list:
        score_up_feature[:, f_i] += 0.5
        score_down_feature[:, f_i] -= 0.5
    elif features[f_i] in bounded_list:
        score_up_feature[:, f_i] += 1
        score_up_feature[:, f_i] /= 2
        score_down_feature[:, f_i] -= 1
        score_down_feature[:, f_i] /= 2
    elif features[f_i] in binary_list:
        for row in range(score_up_feature.shape[1]):
            if random.random() < 0.8:
                for j in range(len(features)):
                    if features[j] in binary_list:
                        score_up_feature[row, j] = -1
                score_up_feature[row, f_i] = 1

                if random.random() < 0.8:
                    for j in range(len(features)):
                        if features[j] in binary_list:
                            score_down_feature[row, j] = 1
                    score_down_feature[row, f_i] = -1
    else:
        continue
    y_predict_up = cls.predict(score_up_feature)
    y_predict_down = cls.predict(score_down_feature)

    voting_dist_up = {}
    for predict in y_predict_up:
        if predict not in voting_dist_up:
            voting_dist_up[predict] = 1
        else:
            voting_dist_up[predict] += 1

    voting_dist_down = {}
    for predict in y_predict_down:
        if predict not in voting_dist_down:
            voting_dist_down[predict] = 1
        else:
            voting_dist_down[predict] += 1

    winner = max(voting_dist_up.items(), key=operator.itemgetter(1))[0]
    if winner != 'Purples':
        print(features[f_i])
        print('Were scaled UP and the new winner prediction is', winner)
        print('-------------------------------------------------------------------------')

    winner = max(voting_dist_down.items(), key=operator.itemgetter(1))[0]
    if winner != 'Purples':
        print(features[f_i])
        print('Were scaled DOWN and the new winner prediction is', winner)
        print('-------------------------------------------------------------------------')

    for p in parties:
        if p in original_voting_dist:
            if p in voting_dist_up:
                parties_dict[p].append((features[f_i], 'up', voting_dist_up[p] - original_voting_dist[p],
                                       abs(voting_dist_up[p] - original_voting_dist[p])))
            if p in voting_dist_down:
                parties_dict[p].append((features[f_i], 'down', voting_dist_down[p] - original_voting_dist[p],
                                        abs(voting_dist_down[p] - original_voting_dist[p])))


########################################################################################################################
# Printing the features that changed the the number of voters in more then 10% for each party
########################################################################################################################
print('-----------------------------------------------------')
print('Finding leading features')
print('-----------------------------------------------------')
for p in parties:
    parties_dict[p] = sorted(parties_dict[p], key=lambda tup: tup[3])
    list_to_print = []
    for f,up_down,change,abs_c in parties_dict[p]:
        if abs(change/original_voting_dist[p]) > 0.1 :
            list_to_print.append((f,up_down,change,str(round(100*change/original_voting_dist[p], 2)) + '%'))
    print(p)
    list_to_print.reverse()
    for (f, up_down, v_change, p_change) in list_to_print:
        print(f, '            (', up_down, ', voters changed: ', v_change, ',  % changed: ', p_change, ')')
    print('-----------------------------------------------------')

