#Implementation of Regression Tree using ID3/CART algorithm

__author__ = 'Vardhaman'
import sys
import csv
import math
import copy
import time
import numpy as np
from collections import Counter
from numpy import *#genfromtxt
feature = []

#normalize the entire dataset prior to learning using min-max normalization
def normalize(matrix):
    a= np.array(matrix)
    a = a.astype(np.float)
    b = np.apply_along_axis(lambda x: (x-np.min(x))/float(np.max(x)-np.min(x)),0,a)
    return b

# reading from the file using numpy genfromtxt
def load_csv(file):
    X = genfromtxt(file, delimiter=",",dtype=str)
    #print(X)
    return (X)

#method to randomly shuffle the array
def random_numpy_array(ar):
    np.random.shuffle(ar)
    #print(arr)
    arr = ar
    #print(arr)
    return arr

#Normalize the data and generate the training labels,training features, test labels and test training
def generate_set(X):
    #print(X.shape[0])
    Y = X[:,-1]
    #print(Y.shape,"Y is",Y)
    #print(Y)
    j = Y.reshape(len(Y),1)
    #print("J is",j)
    new_X = X[:,:-1]
    #normalizing the data step
    normalized_X = normalize(new_X)
    #print("Normal X",normalized_X)
    normalized_final_X = np.concatenate((normalized_X,j),axis=1)
    #print("np",normalized_final_X)
    X = normalized_final_X
    size_of_rows = X.shape[0]
    #test data size is 10%
    num_test = round(0.1*(X.shape[0]))
    start = 0
    end = num_test
    test_attri_list =[]
    test_class_names_list =[]
    training_attri_list = []
    training_class_names_list = []
    for i in range(10):
        X_test = X[start:end , :]
        tmp1 = X[:start, :]
        tmp2 = X[end:, :]
        X_training = np.concatenate((tmp1, tmp2), axis=0)
        #X_training = X[:start,:]+ X[end: , :]
        y_test = X_test[:, -1]
        y_test = y_test.flatten()
        y_training = X_training[:,-1]
        y_training = y_training.flatten()
        y_train = y_training.astype(np.float)
        y_test = y_test.astype(np.float)
        X_test = X_test[:,:-1]
        X_training = X_training[:,:-1]
        X_test = X_test.astype(np.float)
        X_training = X_training.astype(np.float)
        test_attri_list.append(X_test)
        test_class_names_list.append(y_test)
        training_attri_list.append(X_training)
        training_class_names_list.append(y_train)
        #print("start is",start)
        #print("end is",end)
        start = end
        end = end+num_test
    return test_attri_list,test_class_names_list,training_attri_list,training_class_names_list#(X_test,y_test,X_training,y_train)

#calculating the mean sqaured error
def mean_sqaured_error(attri_list):
    sum_list = 0
    for i in attri_list:
        sum_list += i
    se = [(x-sum_list)**2 for x in attri_list]
    #print(se)
    mse = sum(se)/len(attri_list)
    return mse

#calculate rmse error for each test instance
def accuracy_for_predicted_values(test_class_names1,l):
    mse = 0
    for i in range(len(test_class_names1)):
        error = test_class_names1[i] - l[i]
        square_error = error * error
        mse += square_error
    mse = float(mse)/len(test_class_names1)
    return mse
    #return true_count, false_count, float(true_count) / len(l)

#build a dictionary where the key is the class label and values are the features which belong to that class.
def build_dict_of_attributes_with_class_values(X,y): #,feature):
    dict_of_attri_class_values = {}
    fea_list =[]
    for i in xrange(X.shape[1]):
        fea = i
        l = X[:,i]
        #print(l)
        attribute_list =[]
        count = 0
        for j in l:
            attribute_value = []
            attribute_value.append(j)
            attribute_value.append(y[count])
            attribute_list.append(attribute_value)
            count += 1
        dict_of_attri_class_values[fea]= attribute_list
        fea_list.append(fea)
    return dict_of_attri_class_values,fea_list
    #features_with_max_gain_and_theta(dict_of_attri_class_values)

def return_features(Y):
    return feature

#Class node and explanation is self explaination
class Node(object):
    def __init__(self, val, lchild, rchild,the,leaf):
        self.root_value = val
        self.root_left = lchild
        self.root_right = rchild
        self.theta = the
        self.leaf = leaf

    #method to identify if the node is leaf
    def is_leaf(self):
        return self.leaf

    #method to return threshold value
    def ret_thetha(self):
        return self.theta

    def ret_root_value(self):
        return self.root_value

    def ret_llist(self):
        return self.root_left

    def ret_rlist(self):
        return self.root_right

    def __repr__(self):
        return "(%r, %r, %r, %r)" %(self.root_value,self.root_left,self.root_right,self.theta)

#Decision Tree object
class DecisionTree(object):

    fea_list = []
    def __init__(self):
        self.root_node = None

    #fit the decision tree
    def fit(self, dict_of_everything,cl_val,eta_min_val):
        root_node = self.create_decision_tree(dict_of_everything,cl_val,eta_min_val)#,fea_list)
        return root_node

    #calculate the mean values for all the class labels
    def cal_mean_class_values(self,class_values):
        mean_val = sum(class_values)/float (len(class_values))
        #print(mean_val)
        return mean_val

    #method to calculate best threshold value for each feature
    def cal_best_theta_value(self,ke,attri_list):
        data = []
        class_values = []
        #print(attri_list)
        for i in attri_list:
            #val = float(i[0])
            data.append(i[0])
            class_values.append(i[1])
        mse_parent = mean_sqaured_error(class_values)
        #print("mse for parent",mse_parent)
        #print("Entropy of parrent",entropy_of_par_attr)
        max_mean_sqaure = 0
        theta = 0
        best_index_left_list = []
        best_index_right_list = []
        class_labels_list_after_split = []
        #print(data)
        #data = list(data)
        data.sort()
        for i in range(len(data) - 1):
            cur_theta = float(float(data[i])+float(data[i+1]))/ 2
            #print("cur thetha",cur_theta)
            #print(data[i] +"ji"+ data[i+1],cur_theta)
            index_less_than_theta_list = []
            values_less_than_theta_list = []
            index_greater_than_theta_list = []
            values_greater_than_theta_list = []
            count = 0
            for c,j in enumerate(attri_list):
                #print(c,j[0])
                if j[0] <= cur_theta:
                    #print("J[0] less", j[0])
                    values_less_than_theta_list.append(j[1])
                    index_less_than_theta_list.append(c)
                else:
                    #print("J[0] grater",j[0])
                    values_greater_than_theta_list.append(j[1])
                    index_greater_than_theta_list.append(c)
                #count += 1
            #print("Len og less list",len(index_less_than_theta_list))
            #print("len og greater list",len(index_greater_than_theta_list))
            mse_left = mean_sqaured_error(values_less_than_theta_list)
            #print(entropy_of_less_attribute)
            mse_right = mean_sqaured_error(values_greater_than_theta_list)
            cur_mean_sqaure = mse_parent - mse_left - mse_right
            if cur_mean_sqaure > max_mean_sqaure:
                max_mean_sqaure = cur_mean_sqaure
                theta = cur_theta
                best_index_left_list = index_less_than_theta_list
                best_index_right_list = index_greater_than_theta_list
                class_labels_list_after_split = values_less_than_theta_list + values_greater_than_theta_list

        return max_mean_sqaure, theta, best_index_left_list, best_index_right_list, class_labels_list_after_split

    #method to select the best feature out of all the features.
    def best_feature(self,dict_rep):
        #dict_theta = {}
        #dict_theta = {}
        key_value = None
        best_mean_sqaure = -1
        best_theta = 0
        best_index_left_list = []
        best_index_right_list = []
        #best_mse_left = -1
        #best_mse_right = -1
        best_class_labels_after_split = []
        tmp_list = []
        for ke in dict_rep.keys():
            #print("Key now is",ke)
            mean_sqaure, theta, index_left_list, index_right_list,class_labels_after_split = self.cal_best_theta_value(ke,dict_rep[ke])
            #print("Best theta is", ke,info_gain,theta,index_left_list)#,index_right_list)
            if mean_sqaure > best_mean_sqaure:
                best_mean_sqaure = mean_sqaure
                best_theta = theta
                key_value = ke
                best_index_left_list = index_left_list
                best_index_right_list = index_right_list
                best_class_labels_after_split = class_labels_after_split
        tmp_list.append(key_value)
        #tmp_list.append(best_info_gain)
        tmp_list.append(best_theta)
        tmp_list.append(best_index_left_list)
        tmp_list.append(best_index_right_list)
        tmp_list.append(best_class_labels_after_split)
        return tmp_list

    def get_remainder_dict(self,dict_of_everything,index_split):
        #global fea_list
        splited_dict = {}
        for ke in dict_of_everything.keys():
            val_list = []
            modified_list = []
            l = dict_of_everything[ke]
            #print(ke,index_left_split)
            #print(l)
            for i,v in enumerate(l):
                #print(i,v)
                if i not in index_split:
                    #print(ke,i,v)
                    modified_list.append(v)
                    val_list.append(v[1])
            #print(modified_list)
            splited_dict[ke] = modified_list
        return splited_dict,val_list

    #method to create decision tree
    def create_decision_tree(self, dict_of_everything,class_val,eta_min_val):#,fea_list):
        if len(class_val) < eta_min_val:
            majority_val = self.cal_mean_class_values(class_val)
            #print("Leaf node for less than 8 is",majority_val, len(class_val))#,class_val)
            root_node = Node(majority_val,None,None,0,True)
            return root_node
        else:
            best_features_list = self.best_feature(dict_of_everything)
            #print(best_features_list)
            node_name = best_features_list[0]
            theta = best_features_list[1]
            index_left_split = best_features_list[2]
            #print("Length of left split",len(index_left_split))#,index_left_split)
            index_right_split = best_features_list[3]
            #print("Length of right split",len(index_right_split))#,index_right_split)
            class_values = best_features_list[4]
            #print ("Length of class values", len(class_values))
            left_dict,class_val1 = self.get_remainder_dict(dict_of_everything,index_left_split)
            #print("index of left split",len(index_left_split))
            #print("Left class values is",len(class_val1))
            right_dict,class_val2 = self.get_remainder_dict(dict_of_everything,index_right_split)
            #print("indx of right split",len(index_right_split))
            #print("right class values is",len(class_val2))
            leftchild = self.create_decision_tree(left_dict,class_val1,eta_min_val)
            #leftchild = None
            rightchild = self.create_decision_tree(right_dict,class_val2,eta_min_val)
            root_node = Node(node_name,rightchild,leftchild,theta,False)
            return root_node

    #method to predict the values for test data
    def predict(self, X, root):
        predicted_list = []
        for row in X:
            y_pred = self.classify(row,root)
            predicted_list.append(y_pred)
        return predicted_list

    def classify(self,row,root):
        dict_test ={}
        for k,j in enumerate(row):
            dict_test[k] = j
        #print(dict_test)
        current_node = root
        while not current_node.leaf:
            if dict_test[current_node.root_value] <= current_node.theta:
                current_node = current_node.root_left
            else:
                current_node = current_node.root_right
        #print(current_node.root_value,dict_test[current_node.root_value], current_node.theta)
        return current_node.root_value

#main entry for the code
def main(num_arr, eta_min):
    
    eta_min_val = round(eta_min*num_arr.shape[0])
    #randomly shuffle the array so that we can divide the data into test/training
    random_arr1 = random_numpy_array(num_arr)
    #divide data into test labels,test features,training labels, training features
    test_attri_list,test_class_names_list,training_attri_list,training_class_names_list = generate_set(random_arr1)
    accu_count = 0
    test_fin_mse = 0
    pred_fin = 0
    # ten fold iteration for each eta-min value
    for i in range(10):
        #build a dictionary with class labels and respective features values belonging to that class
        dict_of_input,fea = build_dict_of_attributes_with_class_values(training_attri_list[i],training_class_names_list[i])
        #instantiate decision tree instance
        build_dict = DecisionTree()
        # build the decision tree model.
        dec = build_dict.fit(dict_of_input,training_class_names_list[i],eta_min_val)
        #predict the class labels for test features
        l = build_dict.predict(test_attri_list[i],dec)
        #calculate the mean squared error measure for predicited test data
        mse = accuracy_for_predicted_values(test_class_names_list[i],l)
        #print("Number of right values are",right,"Wrong ones are",wrong)
        #accu_count += accu
        test_fin_mse += mse
        #pred_fin += pred
    print("Average MSE for eta min of",eta_min,"is",float(test_fin_mse)/10)

if __name__ == "__main__":
    if len(sys.argv) == 2:
        newfile = sys.argv[1]
        #load the data file and do the preprocessing
        num_arr = load_csv(newfile)
        #for each threshold value run the classifier for 10 cross-validation
        eta_min_list = [0.05,0.10,0.15,0.20]
        for i in eta_min_list:
            main(num_arr,i)
