import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from numpy import array
from numpy import argmax


def main():
    st.title("Different Algorithms for Modeling of EMDAT Dataset")
    st.markdown("Explore different classifier. Which one do you Like?")
    @st.cache(persist=True)
    def load_data():
        data = pd.read_csv(r"C:\Users\NISARG MEHTA\Downloads\emdat_cleaned_data.csv")
        # data = data.values.reshape(-1, 1)
        # label = LabelEncoder()
        # integer encode
        for col in data.columns:
            label_encoder = LabelEncoder()
            integer_encoded = label_encoder.fit_transform(data[col])
            print(integer_encoded)
            # binary encode
            onehot_encoder = OneHotEncoder(sparse=False)
            integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
            data[col] = onehot_encoder.fit_transform(integer_encoded)
        # onehot_encoder = OneHotEncoder()

        #for col in data.columns:
         #   data[col] = onehot_encoder.fit_transform(data[col])

        return data

    @st.cache(persist=True)
    def split(df):
        # y = df.Total_Damages
        # x = df.Disaster_Subtype
        # x = df.drop(columns = ['Total_Damages'],axis=0)
        x = np.asarray(df[['Disaster_Subtype', 'Entry_Criteria','Country']])
        # x = x.reshape(-1,2)
        y = np.asarray(df[['Total_Damages','Total_Deaths','Total_Affected']])
        # y = y.reshape(-1,1)
        print(x.shape)
        print(y.shape)
        # x = x.values.reshape(x[:-1])
        # x = x.transpose()
        # y = y.values.reshape(1, -1) 
        # y = y.values.reshape(178,1)
        # print(y.shape)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        return x_train, x_test, y_train, y_test

    df = load_data()
    x_train, x_test, y_train, y_test = split(df)
    x_train= x_train.reshape(-1, 1)
    y_train= y_train.reshape(-1, 1)
    x_test = x_test.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier",
                                      ("Support Vector Machine (SVM)", "K Nearest Neighbours (KNN)","Logistic Regression", "Random Forest"))

    if classifier == "Support Vector Machine (SVM)":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='auto')

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Support Vector Machine (SVM) Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, average='weighted').round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, average='weighted').round(2))
            # plot_metrics(metrics)
            # pred_svm = svm.predict(X_test)
            cm=confusion_matrix(y_test,y_pred)
            st.write('Confusion matrix: ', cm)
            
    if classifier == "K Nearest Neighbours (KNN)":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C')
        K = st.sidebar.slider('K', 1, 15)
        params = dict()
        params['K'] = K

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("K Nearest Neighbours (KNN) Results")
            model = KNeighborsClassifier(n_neighbors=params['K'])
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, average='weighted').round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, average='weighted').round(2))
            # plot_metrics(metrics)
            # pred_svm = svm.predict(X_test)
            cm=confusion_matrix(y_test,y_pred)
            st.write('Confusion matrix: ', cm)

    if classifier == "Logistic Regression":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter')

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, average='weighted').round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, average='weighted').round(2))
            cm=confusion_matrix(y_test,y_pred)
            st.write('Confusion matrix: ', cm)

    if classifier == "Random Forest":
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10,
                                               key='n_estimators')
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key='max_depth')
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'), key='bootstrap')

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap,
                                           n_jobs=-1)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, average='weighted').round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, average='weighted').round(2))
            cm=confusion_matrix(y_test,y_pred)
            st.write('Confusion matrix: ', cm)

    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Emdat Data Set")
        st.write(df)


if __name__ == '__main__':
    main()
    
    
 ## Code for pandas Profiling        
with st.sidebar.header('Upload your CSV data for Graphs'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    st.sidebar.markdown("""
[Example CSV input file](https://github.com/Nisarg-1406/Extra-Project/blob/main/emdat_cleaned_data.csv)
""")

# Pandas Profiling Report
if uploaded_file is not None:
    @st.cache
    def load_csv():
        csv = pd.read_csv(uploaded_file)
        return csv
    df = load_csv()
    pr = ProfileReport(df, explorative=True)
    st.header('**Input DataFrame**')
    st.write(df)
    st.write('---')
    st.header('**Pandas Profiling Report**')
    st_profile_report(pr)
else:
    if st.button('Press to use Example Dataset for Graphs'):
        # Example data
        @st.cache
        def load_data():
            a = pd.DataFrame(
                np.random.rand(100, 5),
                columns=['a', 'b', 'c', 'd', 'e']
            )
            return a
        df = load_data()
        pr = ProfileReport(df, explorative=True)
        st.header('**Input DataFrame**')
        st.write(df)
        st.write('---')
        st.header('**Pandas Profiling Report**')
        st_profile_report(pr)
