import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# --- Load Model and Data ---
@st.cache_data
def load_model_and_data():
    """Load the saved model, original data, and column list."""
    try:
        with open('student_model.pkl', 'rb') as file:
            data = pickle.load(file)
        model = data['model']
        model_columns = data['columns']
        df_original = pd.read_csv('student_performance.csv')
        df_original.columns = df_original.columns.str.strip()
        return model, model_columns, df_original
    except FileNotFoundError:
        st.error("Model file ('student_model.pkl') or data file ('Students Performance .csv') not found.")
        st.error("Please run the `train_model.py` script first to generate the model file.")
        return None, None, None

model, model_columns, df_original = load_model_and_data()

# --- App Header ---
st.title('🎓 Student Performance Prediction')
st.markdown("""
This application predicts a student's final grade based on various academic and personal factors.
Use the sidebar to enter the student's information and click 'Predict Grade' to see the result.
You can also explore visualizations of the original dataset below.
""")

# --- Sidebar for User Input ---
st.sidebar.header('Enter Student Information')

def get_user_input():
    """Get input from user using sidebar widgets."""
    # Using the unique values from the original dataframe for dropdowns
    student_age = st.sidebar.selectbox('Student Age', df_original['Student_Age'].unique())
    sex = st.sidebar.selectbox('Sex', df_original['Sex'].unique())
    high_school_type = st.sidebar.selectbox('High School Type', df_original['High_School_Type'].unique())
    scholarship = st.sidebar.selectbox('Scholarship', df_original['Scholarship'].unique())
    additional_work = st.sidebar.selectbox('Additional Work?', df_original['Additional_Work'].unique())
    sports_activity = st.sidebar.selectbox('Plays Sports?', df_original['Sports_activity'].unique())
    transportation = st.sidebar.selectbox('Transportation', df_original['Transportation'].unique())
    weekly_study_hours = st.sidebar.selectbox('Weekly Study Hours', df_original['Weekly_Study_Hours'].unique())
    attendance = st.sidebar.selectbox('Attendance', df_original['Attendance'].unique())
    reading = st.sidebar.selectbox('Reads Regularly?', df_original['Reading'].unique())
    notes = st.sidebar.selectbox('Takes Notes?', df_original['Notes'].unique())
    listening_in_class = st.sidebar.selectbox('Listens in Class?', df_original['Listening_in_Class'].unique())
    project_work = st.sidebar.selectbox('Completes Project Work?', df_original['Project_work'].unique())

    # Create a dictionary from the inputs
    input_dict = {
        'Student_Age': student_age,
        'Sex': sex,
        'High_School_Type': high_school_type,
        'Scholarship': scholarship,
        'Additional_Work': additional_work,
        'Sports_activity': sports_activity,
        'Transportation': transportation,
        'Weekly_Study_Hours': weekly_study_hours,
        'Attendance': attendance,
        'Reading': reading,
        'Notes': notes,
        'Listening_in_Class': listening_in_class,
        'Project_work': project_work
    }
    return pd.DataFrame([input_dict])

if model:
    input_df = get_user_input()

    # --- Prediction Logic ---
    if st.sidebar.button('Predict Grade', type="primary"):
        # 1. Preprocess user input to match model's training format
        input_encoded = pd.get_dummies(input_df)
        
        # 2. Align columns with the model's training columns
        #    - Add missing columns and fill with 0
        #    - Ensure the order is the same
        final_input = pd.DataFrame(columns=model_columns)
        final_input = pd.concat([final_input, input_encoded])
        final_input = final_input.reindex(columns=model_columns, fill_value=0)
        
        # Ensure no NaN values exist after reindexing
        final_input = final_input.fillna(0)

        # 3. Make prediction
        prediction = model.predict(final_input)
        prediction_proba = model.predict_proba(final_input)

        # --- Display Prediction ---
        st.subheader('Prediction Result')
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Predicted Grade", prediction[0])
            if prediction[0] == 'Fail':
                st.error("The model predicts the student is at risk of failing.")
            else:
                st.success("The model predicts a passing grade.")

        with col2:
            st.write("Prediction Probabilities:")
            proba_df = pd.DataFrame(prediction_proba, columns=model.classes_, index=['Probability'])
            st.dataframe(proba_df.style.format("{:.2%}"))

    # --- Data Visualizations ---
    st.markdown("---")
    st.header("📊 Data Visualizations")
    st.markdown("Explore the distribution of various factors from the original dataset.")

    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Grade Distribution", "Demographics", "Study Habits"])

    with tab1:
        st.subheader("Distribution of Final Grades")
        fig_grade = px.bar(df_original, x='Grade', title='Count of Students by Final Grade',
                           color='Grade', template='plotly_white',
                           labels={'Grade': 'Final Grade', 'count': 'Number of Students'})
        st.plotly_chart(fig_grade, use_container_width=True)

    with tab2:
        st.subheader("Demographic Information")
        col1, col2 = st.columns(2)
        with col1:
            fig_sex = px.pie(df_original, names='Sex', title='Gender Distribution', hole=0.3)
            st.plotly_chart(fig_sex, use_container_width=True)
        with col2:
            fig_school = px.bar(df_original, y='High_School_Type', title='High School Type Distribution',
                                color='High_School_Type', template='plotly_white',
                                labels={'High_School_Type': 'Type of High School'})
            st.plotly_chart(fig_school, use_container_width=True)

    with tab3:
        st.subheader("Study Habits vs. Grade")
        fig_study = px.sunburst(df_original, path=['Weekly_Study_Hours', 'Grade'],
                                title='Weekly Study Hours vs. Final Grade',
                                color='Weekly_Study_Hours')
        st.plotly_chart(fig_study, use_container_width=True)

