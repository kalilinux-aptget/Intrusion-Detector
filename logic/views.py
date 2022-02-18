from pydoc import render_doc
from django.shortcuts import render
import pandas as pd
import numpy as np
from numpy import argmax
import warnings
warnings.filterwarnings('ignore')
# Create your views here.
def home(request):
    if request.method =="GET":
        return render(request,'logic/home.html')


def design(request):
    return render(request,'logic/design.html')

def output(request):
    train = pd.read_csv(r"C:\Users\FIREBLZE\Desktop\Python Codes and Datasets\Train_data.csv")
    test = pd.read_csv(r"C:\Users\FIREBLZE\Desktop\Python Codes and Datasets\Test_data.csv")
    #'num_outbound_cmds' is a redundant column so remove it from both train & test datasets
    train.drop(['num_outbound_cmds'], axis=1, inplace=True)
    test.drop(['num_outbound_cmds'], axis=1, inplace=True)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

# extract numerical attributes and scale it to have zero mean and unit variance  
    cols = train.select_dtypes(include=['float64','int64']).columns
    sc_train = scaler.fit_transform(train.select_dtypes(include=['float64','int64']))
    sc_test = scaler.fit_transform(test.select_dtypes(include=['float64','int64']))

    # turn the result back to a dataframe
    sc_traindf = pd.DataFrame(sc_train, columns = cols)
    sc_testdf = pd.DataFrame(sc_test, columns = cols)
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()

    # extract categorical attributes from both training and test sets 
    cattrain = train.select_dtypes(include=['object']).copy()
    cattest = test.select_dtypes(include=['object']).copy()

    # encode the categorical attributes
    traincat = cattrain.apply(encoder.fit_transform)
    testcat = cattest.apply(encoder.fit_transform)

    # separate target column from encoded data 
    enctrain = traincat.drop(['class'], axis=1)
    cat_Ytrain = traincat[['class']].copy()
    train_x = pd.concat([sc_traindf,enctrain],axis=1)
    train_y = cat_Ytrain
    test_df = pd.concat([sc_testdf,testcat],axis=1)
    #Recursive feature elimination
    from sklearn.feature_selection import RFE
    import itertools
    from sklearn.ensemble import RandomForestClassifier
    rfc = RandomForestClassifier()

    # create the RFE model and select 15 attributes
    rfe = RFE(rfc, n_features_to_select=15)
    rfe = rfe.fit(train_x, train_y)

    # summarize the selection of the attributes
    feature_map = [(i, v) for i, v in itertools.zip_longest(rfe.get_support(), train_x.columns)]
    selected_features = [v for i, v in feature_map if i==True]
    a = [i[0] for i in feature_map]

    train_x = train_x.iloc[:,a]
    test_df = test_df.iloc[:,a]
    from sklearn.model_selection import train_test_split
    X_train,X_test,Y_train,Y_test = train_test_split(train_x,train_y,train_size=0.70, random_state=2)

    #Fitting Models
    # Importing the Keras libraries and packages
    import keras
    from keras.models import Sequential
    from keras.layers import Dense

    # Initialising the ANN
    classifier = Sequential()

    # Adding the input layer and the first hidden layer
    classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim = 15))

    # Adding the second hidden layer
    classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))

    # Adding the output layer
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

    # Compiling the ANN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    # Fitting the ANN to the Training set
    classifier.fit(X_train, Y_train, batch_size = 10, epochs = 10)
    if request.method =='GET':
        return render(request,'logic/home.html')
    if request.method =='POST':
        src=float(request.POST['srcbytes'])
        dstbytes=float(request.POST['dstbytes'])
        hot=float(request.POST['hot'])
        logged_in=float(request.POST['logged_in'])
        count=float(request.POST['count'])
        srv_count=float(request.POST['srv_count'])
        same_srv_rate=float(request.POST['same_srv_rate'])
        diff_srv_rate=float(request.POST['diff_srv_rate'])
        dst_host_same_srv_rate=float(request.POST['dst_host_same_srv_rate'])
        dst_host_srv_count=float(request.POST['dst_host_srv_count'])
        dst_host_diff_srv_rate=float(request.POST['dst_host_diff_srv_rate'])
        dst_host_same_src_port_rate=float(request.POST['dst_host_same_src_port_rate'])
        protocol_type=int(request.POST['protocol_type'])
        service=int(request.POST['service'])
        flag=int(request.POST['flag'])
        final=[]
        final.append(src)
        final.append(dstbytes)
        final.append(hot)
        final.append(logged_in)
        final.append(count)
        final.append(srv_count)
        final.append(same_srv_rate)
        final.append(diff_srv_rate)
        final.append(dst_host_same_srv_rate)
        final.append(dst_host_srv_count)
        final.append(dst_host_diff_srv_rate)
        final.append(dst_host_same_src_port_rate)
        final.append(protocol_type)
        final.append(service)
        final.append(flag)
        finaldt =pd.DataFrame(final)
        pred_ann = classifier.predict(finaldt.T)
        np.argmax(pred_ann)
        return render(request,'logic/output.html',{'src':np.argmax(pred_ann)})

def homes(request):
    return render(request, 'logic/basic.html')
 
def new_page(request):
    data = request.POST['fulltextarea']
    return render(request, 'logic/basic2.html', {'data':data})