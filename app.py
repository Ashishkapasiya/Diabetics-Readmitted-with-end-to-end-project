import streamlit as st
import pickle

pipe = pickle.load(open('pipe.pkl', 'rb'))
pipe_dtc = pickle.load(open('pipe_dtc.pkl', 'rb'))
pipe_rtc = pickle.load(open('pipe_rtc.pkl', 'rb'))
pipe_xgb = pickle.load(open('pipe_xgb.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

def classify(num):
    if num < 0.5:
        return 'patient is not readmitted'
    else:
        return 'patient is readmitted'

def main():
    st.title("Diabetics Readmitted Or Not")
    html_temp = """
     <div style="background-color:teal ;padding:10px">
     <h2 style="color:white;text-align:center;">Diabetes Predictor ML App</h2>
     </div>
     """
    st.markdown(html_temp, unsafe_allow_html=True)
    activities = ['Logistic Regression', 'DecisionTreeClassifier', 'RandomForestClassifier', 'XGBClassifier']
    option = st.sidebar.selectbox('Which model would you like to use?', activities)
    st.subheader(option)
    race = st.selectbox('race', df['race'].unique())
    gender = st.selectbox('gender', df['gender'].unique())
    age = st.text_input('age', "Type Here")
    admission_type_id = st.selectbox('admission_type_id', df['admission_type_id'].unique())
    discharge_disposition_id = st.selectbox('discharge_disposition_id', df['discharge_disposition_id'].unique())
    admission_source_id = st.selectbox('admission_source_id', df['admission_source_id'].unique())
    time_in_hospital = st.text_input('time_in_hospital', "Type Here")
    num_lab_procedures = st.text_input('num_lab_procedures', "Type Here")
    num_procedures = st.text_input('num_procedures', "Type Here")
    num_medications = st.text_input('num_medications', "Type Here")
    number_emergency = st.text_input('number_emergency', "Type Here")
    number_diagnoses = st.text_input('number_diagnoses', "Type Here")
    max_glu_serum = st.text_input('max_glu_serum', "Type Here")
    A1Cresult = st.text_input('A1Cresult', "Type Here")
    change = st.text_input('change', "Type Here")
    diabetesMed = st.text_input('diabetesMed', "Type Here")
    count = st.text_input('count', "Type Here")
    number_patients = st.text_input('number_patients', "Type Here")
    steady = st.text_input('steady', "Type Here")
    up = st.text_input('up', "Type Here")
    down = st.text_input('down', "Type Here")
    inputs = [[race, gender, age, admission_type_id, discharge_disposition_id, admission_source_id,
                                         time_in_hospital, num_lab_procedures, num_procedures, num_medications,
                                        number_emergency, number_diagnoses, max_glu_serum, A1Cresult, change,
                                        diabetesMed, count, number_patients, steady, up, down]]
    if st.button('predict'):
        if option == 'Logistic Regression':
            st.success(classify(pipe.predict(inputs)))
        elif option == 'DecisionTreeClassifier':
            st.success(classify(pipe_dtc.predict(inputs)))
        elif option == 'RandomForestClassifier':
            st.success(classify(pipe_rtc.predict(inputs)))
        else:
            st.success(classify(pipe_xgb.predict(inputs)))

    if st.button("Thanks"):
        st.balloons()
        st.info("Thank u")

    st.sidebar.header("About App")
    st.sidebar.info("Checking the patient is readmitted or not through diabetics")

    st.sidebar.header("About")
    st.sidebar.markdown("[LinkedInProfile](""https://www.linkedin.com/in/ashish-kapasiya-b89892138/"")")
    st.sidebar.markdown("[GithhubProfile](""https://github.com/Ashishkapasiya"")")
    st.sidebar.markdown("[KaggleInProfile](""https://www.kaggle.com/ashishkapasiya"")")

    st.sidebar.text("Built with Streamlit")
    st.sidebar.text("Maintained by Ashish Kapasiya")

if __name__=='__main__':
    main()