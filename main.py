
import streamlit as st

import pandas as pd

import matplotlib
import seaborn as sns


from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
matplotlib.use("Agg")


def main():
    st.title("BASIC AUTO ML MODEL")
    activities = ["EDA", "PLOTTING", "MODEL BUILDING"]
    choice = st.sidebar.selectbox("Select Activities:",activities)

    if choice == "EDA":
        st.subheader("Exploratory data analysis")
        data = st.file_uploader("Upload Dataset", type=["csv", "txt","DATA"])
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())

            if st.checkbox("Show Shape"):
                st.write(df.shape)

            if st.checkbox("Show Summary"):
                st.write(df.describe())

            if st.checkbox("Show Target Data Count"):
                st.write(df.iloc[:, -1].value_counts())

            if st.checkbox("Show Selected Columns"):
                all_columns = df.columns.tolist()
                selected_columns = st.multiselect("Show Columns", all_columns)
                new_df = df[selected_columns]
                st.dataframe(new_df)

    elif choice == "PLOTTING":
        st.subheader("Data Visualization")

        data = st.file_uploader("Upload Dataset", type=["csv", "txt", "DATA"])
        if data is not None:
            df = pd.read_csv(data)

            if st.checkbox("Correlation with Seaborn"):
                st.write(sns.heatmap(df.corr(), annot=True))
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot()

            if st.checkbox("Pie Chart"):
                all_columns = df.columns.tolist()
                columns_to_plot = st.selectbox("Select 1 Columns", all_columns)
                pie_plot = df[columns_to_plot].value_counts().plot.pie(autopct = "%1.1f%%")
                st.write(pie_plot)
                st.pyplot()

            all_columns_names = df.columns.tolist()
            type_of_plot = st.selectbox("Select Type of Plot", ["Area", "Bar", "Line"])
            selected_columns_names = st.multiselect("Select columns To Plot",all_columns_names)

            if st.button("Generate PLot"):
                st.success("Generating Customizable PLot of {} for {}".format(type_of_plot, selected_columns_names))

                if type_of_plot == "Area":
                    cust_data = df[selected_columns_names]
                    st.area_chart(cust_data)

                elif type_of_plot == "Bar":
                    cust_data = df[selected_columns_names]
                    st.bar_chart(cust_data)

                elif type_of_plot == "Line":
                    cust_data = df[selected_columns_names]
                    st.line_chart(cust_data)

                elif type_of_plot:
                    cust_data = df[selected_columns_names].plot(kind=type_of_plot)
                    st.write(cust_data)
                    st.pyplot()

    elif choice == "MODEL BUILDING":
        st.subheader("Building Data Model")
        data = st.file_uploader("Upload Dataset", type=["csv", "txt", "DATA"])
        if data is not None:
            df = pd.read_csv(data)
            X = df.iloc[:, 0:-1]
            Y = df.iloc[:, -1]


            models = []

            models.append(("LR", LogisticRegression()))
            models.append(("DTC", DecisionTreeClassifier()))
            models.append(("SVC", LinearSVC()))
            models.append(("KNN", KNeighborsClassifier()))

            model_names = []
            model_mean = []
            model_std = []
            all_models = []
            scoring = "accuracy"

            for name, model in models:
                Kfold = model_selection.KFold(n_splits=10 , random_state=None)
                cv_results = model_selection.cross_val_score(model, X, Y, cv=Kfold, scoring= scoring)
                model_names.append(name)
                model_mean.append(cv_results.mean())
                model_std.append(cv_results.std())

                accuracy_results = {"model_names": name, "model accuracy": cv_results.mean(), "standard_deviation": cv_results.std()}
                all_models.append(accuracy_results)

            st.checkbox("Metrics as Table")
            st.dataframe(pd.DataFrame(zip(model_names,model_mean,model_std)))


if __name__ == "__main__":
    main()




















