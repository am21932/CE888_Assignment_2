

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, StratifiedKFold
import pandas as pd


# Returns a list of trained random forest classifiers/predictions per cluster
def trainingClusters(X, y, k, chosen_km):
    true = np.zeros(k)
    false = np.zeros(k)
    X_cluster_list = [[] for i in range(k)]
    y_cluster_list = [[] for i in range(k)]
    result_list = [None for i in range(k)]

    for i in range(len(chosen_km.labels_)):
        true[chosen_km.labels_[i]] += 1 if y[i] == 1 else 0  # Count 1 labels per cluster
        false[chosen_km.labels_[i]] += 1 if y[i] == 0 else 0  # Count 0 labels per cluster
        X_cluster_list[chosen_km.labels_[i]].append(X[i])  # Stores feature arrays per cluster
        y_cluster_list[chosen_km.labels_[i]].append(y[i])  # Stores labels per cluster

    # Stores selected prediction for clusters with only one class
    for i in range(k):
        if true[i] == 0:
            result_list[i] = 1
        if false[i] == 0:
            result_list[i] = 0

    # Stores trained random forest classifier for clusters with more than one class
    for i in range(k):
        if result_list[i] == None:
            # train a random forest classifier
            clf = RandomForestClassifier(random_state=50, n_estimators=700, max_depth=6)
            # save it to result_list
            result_list[i] = clf.fit(X_cluster_list[i], y_cluster_list[i])

    return result_list


# Creating the function that returns the percentage of each class in the dataset
def imbalance(elements):
    percent=[]
    sum_of_elements=np.sum(elements)
    for i in range(len(elements)):
        percent.append((elements[i]/sum_of_elements)*100)
    return percent



#function to create imbalance in the skin - non skin dataset...
def create_imbalanced_data_for_stars_new(imb_percent, size_of_one_class):
    read_data =  pd.read_table('Skin_NonSkin.txt', delimiter = '\t')
    size = int(size_of_one_class/imb_percent)
    read_data.columns =['B', 'G', 'R', 'Skin']
    drop_indices2 = np.random.choice(read_data[read_data["Skin"]==2].index, 190000, replace=False)
    drop_indices1 = np.random.choice(read_data[read_data["Skin"]==1].index, 46000, replace=False)
    read_data = read_data.drop(drop_indices1)
    read_data = read_data.drop(drop_indices2)
    data_0 = read_data.query("Skin==1").sample(size-size_of_one_class)
    read_data.Skin[read_data.Skin == 2] = 0
    data_1 = read_data.query("Skin==0")
    return pd.concat([data_0, data_1])

# Function to create imbalanced dataset from the forex_chfjpy.csv file.
def create_imbalanced_data_for_forex(imb_percent, size_of_one_class):
    size = int(size_of_one_class/imb_percent)
    dforex = pd.read_csv("FOREX_chfjpy.csv")
    data_0 = dforex.query("Class==0").sample(size-size_of_one_class)
    data_1 = dforex.query("Class==1")
    return pd.concat([data_0, data_1])

#function to preprocess forex dataset
def preprocess_forex(data):
    # encoding the Class with label encoder
    label_encoder = preprocessing.LabelEncoder()
    data['Class'] = label_encoder.fit_transform(data['Class'])
    # changing the datatype of Timestamp column to datetime
    # dforex['Timestamp'] = pd.to_datetime(dforex['Timestamp'])
    data = data.drop("Timestamp", 1)
    return data


# Function to create imbalanced datasets from CrimeClassification.csv file.
def create_imbalanced_data_for_crime(imb_percent, size_of_one_class):
    size = int(size_of_one_class/imb_percent)
    data = pd.read_csv("CrimeClassification.csv")
    drop_indices2 = np.random.choice(data[data["ViolentCrime"]=='Yes'].index, 267000, replace=False)
    drop_indices1 = np.random.choice(data[data["ViolentCrime"]=='No'].index, 267000, replace=False)
    data = data.drop(drop_indices1)
    data = data.drop(drop_indices2)
    data_0 = data.query("ViolentCrime=='Yes'").sample(size-size_of_one_class)
    data_1 = data.query("ViolentCrime=='No'")
    return pd.concat([data_0, data_1])

#function to preprocess crime dataset.
def preprocessing_crime(data):
    data = data.drop('Address', axis=1)
    label_encoder = preprocessing.LabelEncoder()
    data['ViolentCrime']= label_encoder.fit_transform(data['ViolentCrime'])

    # Encoding the PdDistrict with OneHotEncoder() using the get_dummies function from pandas...
    data = pd.get_dummies(data)
    return data

# Displays the Silhouette and elbow graphs, receives user input 
# and returns the optimal k and trained kMeans clusterer
# def silhouette_elbow_cluster_selection(X, y):
def select_clusters_from_sil_elbow(X, y):
    Sum_of_squared_distances = []
    silhouette = []
    K = range(2,12)
    max_score = -1
    max_k = -1
    # Kmeans trained with 12 possible k
    for k in K:
        km = KMeans(n_clusters=k).fit(X)
        Sum_of_squared_distances.append(km.inertia_)
        silhouette.append(silhouette_score(X, km.labels_, metric='euclidean'))
        
    # Silhouette graph
    plt.plot(K, silhouette, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.title('Clusters as per Silhouette Method')
    plt.show()

    # User input
    input_silhouette = int(input("Enter number of clusters for the silhouette graph: "))

    # Elbow graph
    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum of squared distances')
    plt.title('Elbow Method')
    plt.show()
    
    # User input
    input_elbow = int(input("Enter number of clusters for the elbow graph: "))
    
    # Upper and Lower bounds to test
    min_test_k = input_elbow if input_elbow < input_silhouette else input_silhouette
    max_test_k = input_elbow if input_elbow > input_silhouette else input_silhouette
    
    final_k = 0
    final_score = 0
    final_clustering = KMeans(n_clusters=min_test_k).fit(X)
    
    # Selecting optimal k according to f1 score
    for i in range(min_test_k, max_test_k + 1):
        km_test = KMeans(n_clusters=i).fit(X)
        score = f1_score(km_test.labels_, y, average='weighted')
        if score > final_score:
            final_k = i
            final_clustering = km_test
            final_score = score
            
    print('Best performing clustering F1-score is:', final_score,  "for n_clusters =", final_k)

    return final_k, final_clustering


# Returns a list of predictions generated by the classifier of the predicted cluster
def testingClusters(X, chosen_km, clfs):
    cluster_labels = chosen_km.predict(X) # Predict cluster of each example
    y_predicted = []
    # Predict label according to each cluster
    for index, label in enumerate(cluster_labels):
        # print("index - ", index, "label - ", label)
        if clfs[label] == 0:
            y_predicted.append(0)
        elif clfs[label] == 1:
            y_predicted.append(1)
        else:
            y_predicted.append(clfs[label].predict(X[index].reshape(1, -1))[0])

    return y_predicted

# Returns the f1 score and prints f1 score, precision, recall and accuracy
def metrices(y_true, y_pred):
    acc, f1, prec, rec = np.array([]), np.array([]), np.array([]), np.array([])

    for i in range(len(y_true)):
        acc = np.append(acc, accuracy_score(np.array(y_pred[i]), y_true[i]))
        f1 = np.append(f1, f1_score(np.array(y_pred[i]), y_true[i], average='weighted'))
        prec = np.append(prec, precision_score(np.array(y_pred[i]), y_true[i], average='weighted'))
        rec = np.append(rec, recall_score(np.array(y_pred[i]), y_true[i], average='weighted'))
    print("Result:")
    print("RECALL: %0.4f +/- %0.4f" % (rec.mean(), rec.std()))
    print("PRECISION: %0.4f +/- %0.4f" % (prec.mean(), prec.std()))
    print("ACCURACY: %0.4f +/- %0.4f" % (acc.mean(), acc.std()))
    print("F1: %0.4f +/- %0.4f" % (f1.mean(), f1.std()))
    return f1


