import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter

def kNNClassifier(dataset):
    print()
    print("######################## KNN Classifier ############################")
    print()

    y = dataset['target']
    X = dataset.drop(['target'], axis = 1)

    def Euclidean_distance(a, b):
        
        # No.of dimensions of point a
        length = len(a)
        
        # intialising the distance
        distance = 0
        
        # Calculating the euclidean distance between points a and b
        for i in range(length):
            distance += abs(a[i] - b[i])**2
            
        distance = distance**(1/2)
        
        #returning the distance
        return distance

    #Splitting the data into train and test data by assigning the percentage to test_size
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

    # Scaling the train set and then the test set
    standaradScaler = StandardScaler()
    X_train = standaradScaler.fit_transform(X_train)
    X_test = standaradScaler.transform(X_test)

    def knn_implement(X_train, X_test, y_train, y_test, k):
        yImpTest = []

        for testPoint in X_test:
            #intialising the distances
            distances = []

            for trainPoint in X_train:
                distance = Euclidean_distance(testPoint, trainPoint)
                distances.append(distance)
            
            #Storing the calculated euclidean distance in Data Frames
            DFDistance = pd.DataFrame(data=distances, columns=['dist'], 
                                    index=y_train.index)
            
            #Sorting the Distances and getting the k closest points 
            DFClosest = DFDistance.sort_values(by=['dist'], axis=0)[:k]

            # Creating counter to track the closest points
            counter = Counter(y_train[DFClosest.index])

            #Getting the common among the closest points
            predict = counter.most_common()[0][0]
            
            #Appending all the predicted list
            yImpTest.append(predict)
            
        return yImpTest

    #intiliasing the scores
    scores = []

    #Looping the k values from 1 to 10
    for k in range(1,10):
        yImpTest = knn_implement(X_train, X_test, y_train, y_test, k)
        #Getting the accuracy score
        scores.append(accuracy_score(y_test, yImpTest))

    print(scores)   
    #[1.0, 0.9557522123893806, 0.9292035398230089, 0.9026548672566371, 0.8790560471976401, 0.8967551622418879, 
    #0.8790560471976401, 0.8967551622418879, 0.8967551622418879]
        # [1.0, 0.9675324675324676, 0.9383116883116883, 0.9090909090909091, 0.8766233766233766, 0.8928571428571429, 
        # 0.8733766233766234, 0.8928571428571429, 0.8993506493506493]

    #plotting the graph for k and scores calculated
    plt.plot([k for k in range(1, 10)], scores, color = 'green')
    for i in range(1,10):
        plt.text(i, scores[i-1], (i, scores[i-1]))

    #x-axis and y-axis lables and titles for graph    
    plt.title('Graph for K Neighbors classifier scores')
    plt.xlabel('Neighbors (K)')
    plt.ylabel('Scores')