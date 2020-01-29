import pandas as pd
from sklearn import model_selection
from sklearn import preprocessing as pre
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB as Gb
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from yellowbrick.classifier import ClassificationReport


def rmoutlier(y, Const):
    # -87.9977,-87.5336,41.5600,42.1860
    y.drop(y[y['Longitude'] <= Const[0]].index, inplace=True, axis=0)
    y.drop(y[y['Longitude'] >= Const[1]].index, inplace=True, axis=0)
    y.drop(y[y['Latitude'] <= Const[2]].index, inplace=True, axis=0)
    y.drop(y[y['Latitude'] >= Const[3]].index, inplace=True, axis=0)
    return y


class model():
    Crime = pd.read_csv('chicago2019.csv')
    Crime.drop(
        ['Date', 'Description', 'Block', 'Location Description', 'Beat', 'Case Number', 'IUCR', 'Location', 'FBI Code',
         'X Coordinate', 'Y Coordinate', 'Updated On'], axis=1, inplace=True)
    Crime.dropna(inplace=True)  # removing null values
    mapConstraints = [-87.9977, -87.5336, 41.5600, 42.1860]
    Crime = rmoutlier(Crime, mapConstraints)
    Crime = pd.concat([Crime, pd.get_dummies(Crime['Primary Type'], prefix='Crime_Type')], axis=1)
    Crime.drop(['Primary Type'], axis=1, inplace=True)
    label_encoder = pre.LabelEncoder()
    Crime['Arrest'] = label_encoder.fit_transform(Crime['Arrest'])
    Crime['Domestic'] = label_encoder.fit_transform(Crime['Domestic'])
    x = Crime.drop('Arrest', axis=1)
    y = Crime.Arrest
    xtrain, xtest, ytrain, ytest = model_selection.train_test_split(x, y, test_size=0.3, random_state=42)
    sc_x = StandardScaler()
    xtrain = sc_x.fit_transform(xtrain)
    xtest = sc_x.transform(xtest)
    print(x.shape)
    print(y.shape)
    print(xtrain.shape)
    print(xtest.shape)
    print(ytrain.shape)
    print(ytest.shape)
    from sklearn.linear_model import LogisticRegression
    logreg_clf = LogisticRegression()
    Lr_pred = logreg_clf.fit(xtrain, ytrain).predict(xtest)
    # logreg_clf.predict(test_features)
    # test_pred= longreg_clf.predict([[0,2]])
    cm = confusion_matrix(ytest, Lr_pred)
    print("Confusion Matrix : \n", cm)

    acc = accuracy_score(ytest, Lr_pred)
    print('Accuracy : ', acc * 100)
    gb = Gb()
    gb_pred = gb.fit(xtrain, ytrain).predict(xtest)
    acc = accuracy_score(ytest, gb_pred)
    print('Accuracy : ', acc * 100)

    # import the necessary modules
    # create an object of type LinearSVC
    svc_model = LinearSVC(random_state=0)
    # train the algorithm on training data and predict using the testing data
    svc_pred = svc_model.fit(xtrain, ytrain).predict(xtest)
    # print the accuracy score of the model
    print("LinearSVC accuracy : ", accuracy_score(ytest, svc_pred, normalize=True))

    # Instantiate the classification model and visualizer
    visualizer = ClassificationReport(svc_model, classes=['Arrest', 'No-Arrest'])
    visualizer.fit(xtrain, ytrain)  # Fit the training data to the visualizer
    visualizer.score(xtest, ytest)  # Evaluate the model on the test data
    g = visualizer.poof()  # Draw/show/poof the data
