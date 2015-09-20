# DecisionTrees
Goal of this project is to build Multiclass Decision Trees and Regression Decision trees without using any Machine learning libraries


#Build Multiclass decision Trees with Numerical features using C4.5/ID3 algorithm

IDE algorithm : https://en.wikipedia.org/wiki/ID3_algorithm.

C4.5 algorithm: https://en.wikipedia.org/wiki/C4.5_algorithm.
 
In this project, I worked with two datasets:


• Iris: has three classes and the task is to accurately predict one of the three sub-types of the
Iris flower given four different physical features using Decision trees. These features include the length and width
of the sepals and the petals. There are a total of 150 instances with each class having 50
instances.


• Spambase: is a binary classification task and the objective is to classify email messages as
being spam or not. To this end the dataset uses fifty seven text based features to represent
each email message. There are about 4600 instances

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
area, etc.

I used mean squared error (MSE) to define the splits. I also used an early stopping strategy
used ηmin ∈ {5, 10, 15, 20}. 

I Calculate the MSE using ten fold cross-validation for each value of ηmin and report the average and standard
deviation across the folds

More details of this can be found at http://www.ccs.neu.edu/course/cs6140sp15/1_intro_DT_RULES_REG/hw1/Assignment_1_Spring15.pdf

I implemented the code using python. I divided the data into 90% training set and 10% test set for each data set. 
