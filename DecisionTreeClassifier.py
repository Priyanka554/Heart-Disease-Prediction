import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def decisionTreeClassifier(data):
    print()
    print("######################## Decision Tree Classifier ############################")
    print()
    x = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    x1 = data.iloc[:, :-1]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    x_test = ss.transform(x_test)

    # Decision Tree Classifier
    classifier = DecisionTreeClassifier()
    classifier.fit(x_train, y_train)

    y_predict_test = classifier.predict(x_test)

    # accuracy_score(y_test, y_predict_test)
    y_predict_train = classifier.predict(x_train)

    cm_test = confusion_matrix(y_predict_test, y_test)
    cm_train = confusion_matrix(y_predict_train, y_train)

    print('Accuracy score for training set: {}'.format((cm_train[0][0] + cm_train[1][1])/len(y_train)))
    print('Accuracy score for test set: {}'.format((cm_test[0][0] + cm_test[1][1])/len(y_test)))

    print(pd.DataFrame(confusion_matrix(y_test, y_predict_test), columns=[1, 0], index=[1, 0]))

    from sklearn import tree
    dotfile = open("tree.dot", 'w')
    tree.export_graphviz(classifier, out_file = dotfile, feature_names=x1.columns)
    dotfile.close()

    from subprocess import call
    call(['dot', '-T', 'png', 'tree.dot', '-o', 'tree.png'])
    print()

    img = mpimg.imread('tree.png')
    plt.imshow(img)
    plt.show()