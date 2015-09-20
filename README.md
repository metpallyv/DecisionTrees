# DecisionTrees
Goal of this project is to build Multiclass Decision Trees and Regression Decision trees without using any Machine learning libraries


#Build Multiclass decision Trees with Numerical features using C4.5/ID3 algorithm

IDE algorithm : https://en.wikipedia.org/wiki/ID3_algorithm.

C4.5 algorithm: https://en.wikipedia.org/wiki/C4.5_algorithm.
 
In this project, I worked with two datasets:


• Iris: has three classes and the task is to accurately predict one of the three sub-types of the
Iris flower given four different physical features using Decision trees. These features include the length and width
of the sepals and the petals. There are a total of 150 instances with each class having 50
instances. Here are the features for Iris dataset:


1. sepal length in cm 
2. sepal width in cm 
3. petal length in cm 
4. petal width in cm 
5. class: 
-- Iris Setosa 
-- Iris Versicolour 
-- Iris Virginica

• Spambase: is a binary classification task and the objective is to classify email messages as
being spam or not. To this end the dataset uses fifty seven text based features to represent
each email message. There are about 4600 instances. Here are what each feature means :

The last column of 'spambase.data' denotes whether the e-mail was considered spam (1) or not (0), i.e. unsolicited commercial e-mail. Most of the attributes indicate whether a particular word or character was frequently occuring in the e-mail. The run-length attributes (55-57) measure the length of sequences of consecutive capital letters. For the statistical measures of each attribute, see the end of this file. Here are the definitions of the attributes: 

48 continuous real [0,100] attributes of type word_freq_WORD 
= percentage of words in the e-mail that match WORD, i.e. 100 * (number of times the WORD appears in the e-mail) / total number of words in e-mail. A "word" in this case is any string of alphanumeric characters bounded by non-alphanumeric characters or end-of-string. 

6 continuous real [0,100] attributes of type char_freq_CHAR] 
= percentage of characters in the e-mail that match CHAR, i.e. 100 * (number of CHAR occurences) / total characters in e-mail 

1 continuous real [1,...] attribute of type capital_run_length_average 
= average length of uninterrupted sequences of capital letters 

1 continuous integer [1,...] attribute of type capital_run_length_longest 
= length of longest uninterrupted sequence of capital letters 

1 continuous integer [1,...] attribute of type capital_run_length_total 
= sum of length of uninterrupted sequences of capital letters 
= total number of capital letters in the e-mail 

1 nominal {0,1} class attribute of type spam 
= denotes whether the e-mail was considered spam (1) or not (0), i.e. unsolicited commercial e-mail. 



Instead of growing full trees, I used an early stopping strategy. To this end, we will impose
a limit on the minimum number of instances at a leaf node, let this threshold be denoted as ηmin,
where ηmin is described as a percentage relative to the size of the training dataset. For example if
the size of the training dataset is 150 and ηmin = 5, then a node will only be split further if it has
more than eight instances.


(a) For the Iris dataset I used ηmin ∈ {5, 10, 15, 20}, and calculate the accuracy using ten fold
cross-validation for each value of ηmin.


(b) For the Spambase dataset use ηmin ∈ {5, 10, 15, 20, 25}, and calculate the accuracy using ten
fold cross-validation for each value of ηmin

Since, both datasets have continuous features I implemented decision trees that have binary
splits. For determining the optimal threshold for splitting searched over all possible
thresholds for a given feature and used information gain to measure node impurity in your implementation.
I used ten fold cross-validation for each value of ηmin to implement my model and reported the average accuracy and standard
deviation across the folds.


#Build Regression decision Trees with Numerical features using ID3/CART algorithm.
For this project, I implemented regression trees on housing dataset:

• Housing: This is a regression dataset where the task is to predict the value of houses in the
suburbs of Boston based on thirteen features that describe different aspects that are relevant
to determining the value of a house, such as the number of rooms, levels of pollution in the
area, etc. Here are the definitions of the features:

1. CRIM: per capita crime rate by town 
2. ZN: proportion of residential land zoned for lots over 25,000 sq.ft. 
3. INDUS: proportion of non-retail business acres per town 
4. CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise) 
5. NOX: nitric oxides concentration (parts per 10 million) 
6. RM: average number of rooms per dwelling 
7. AGE: proportion of owner-occupied units built prior to 1940 
8. DIS: weighted distances to five Boston employment centres 
9. RAD: index of accessibility to radial highways 
10. TAX: full-value property-tax rate per $10,000 
11. PTRATIO: pupil-teacher ratio by town 
12. B: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town 
13. LSTAT: % lower status of the population 
14. MEDV: Median value of owner-occupied homes in $1000's




I used mean squared error (MSE) to define the splits. I also used an early stopping strategy
used ηmin ∈ {5, 10, 15, 20}. 

I Calculate the MSE using ten fold cross-validation for each value of ηmin and report the average and standard
deviation across the folds

More details of this can be found at http://www.ccs.neu.edu/course/cs6140sp15/1_intro_DT_RULES_REG/hw1/Assignment_1_Spring15.pdf

I implemented the code using python. I divided the data into 90% training set and 10% test set for each data set. 
