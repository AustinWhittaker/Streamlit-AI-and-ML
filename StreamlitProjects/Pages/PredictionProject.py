import streamlit as st
import pandas as pd
import numpy as np
import sklearn.ensemble as sk


st.set_page_config(
    page_title="Penguin Prediction App",
    page_icon=":penguin:"
)

st.markdown("<h1 style='text-align: center;'> <u>Penguin Prediction App</u> </h2>", unsafe_allow_html=True)
st.image('Images/Penguins.jpg')
st.markdown('---')

with st.expander('Penguin Data'):
    st.write("**Raw Data**")
    df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
    df

    st.write("**X**")

    x_raw = df.drop('species', axis=1)

    st.write("**Y**")
    y_raw = df.species

with st.expander('Data Visualization via Scatter Chart'):
    st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species', x_label='Bill Length', y_label='Body Mass')


with st.expander('Predict Penguin'):

    st.header('Input Features')
    island = st.selectbox('Island', options=('Torgersen', 'Biscoe', 'Dream'))
    sex = st.selectbox('Sex', options=('male', 'female'))
    bill_length = st.slider('Bill Length (mm)', 33.0, 60.0, 43.0)
    bill_depth = st.slider('Bill Depth (mm)', 13.1, 21.5, 17.2)
    flipper_length = st.slider('Flipper Length (mm)', 172.0, 231.0, 201.0)
    body_mass = st.slider('Body Mass (g)', 2700.0, 6300.0, 4200.0)

    allData = {'island': island,
               'bill_length_mm': bill_length,
               'bill_depth_mm': bill_depth,
               'flipper_length_mm': flipper_length,
               'body_mass_g': body_mass,
               'sex': sex}

    input_df = pd.DataFrame(allData, index=[0])

    input_penguins = pd.concat([input_df, x_raw], axis=0)

    # Encode X
    encode = ['island', 'sex']
    df_penguins = pd.get_dummies(input_penguins, prefix=encode)

    x = df_penguins[1:]
    input_row = df_penguins[:1]

    # st.write('***Input Penguin***')
    # input_df
    # st.write('***Combined Penguin Data***')
    # input_penguins
    # st.write('Encoded input penguins')
    # input_row

    # Encode y
    target_mapper = {'Adelie': 0,
                     'Chinstrap': 1,
                     'Gentoo': 2}

    def target_encode(val):
        return target_mapper[val]

    y = y_raw.apply(target_encode)

    # Model Training and Inference
    clf = sk.RandomForestClassifier()
    clf.fit(x, y)

    # Apply model to make predictions
    prediction = clf.predict(input_row)
    prediction_proba = clf.predict_proba(input_row)

    df_pred_proba = pd.DataFrame(prediction_proba)
    df_pred_proba.columns = ['Adelie', 'Chinstrap', 'Gentoo']
    df_pred_proba.rename(columns={0: 'Adelie',
                                  1: 'Chinstrap',
                                  2: 'Gentoo'})

    st.dataframe(df_pred_proba,
                 column_config={
                     'Adelie': st.column_config.ProgressColumn(
                         'Adelie',
                         format='%f',
                         width='medium',
                         min_value=0,
                         max_value=1
                     ),
                     'Chinstrap': st.column_config.ProgressColumn(
                         'Chinstrap',
                         format='%f',
                         width='medium',
                         min_value=0,
                         max_value=1
                     ),
                     'Gentoo': st.column_config.ProgressColumn(
                         'Gentoo',
                         format='%f',
                         width='medium',
                         min_value=0,
                         max_value=1
                     )
                 }, hide_index=True)

    penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
    st.success(str(penguins_species[prediction[0]]))

    if str(penguins_species[prediction[0]]) == 'Adelie':
        st.image('Images/Adelie.jpg')
    elif str(penguins_species[prediction[0]]) == 'Chinstrap':
        st.image('Images/Chinstrap.jpg')
    elif str(penguins_species[prediction[0]]) == 'Gentoo':
        st.image('Images/Gentoo.jpg')

st.markdown('---')
st.text('Inspired by "Data Professor" at https://www.youtube.com/watch?v=Eai1jaZrRDs')