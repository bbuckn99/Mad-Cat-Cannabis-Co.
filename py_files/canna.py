import streamlit as st
import pandas as pd
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rec import recommendations, weed, similarity_matrix

# set page config
st.set_page_config(page_title='Mad Cat Cannabis Co', page_icon=':cat:', layout='wide')

# custom CSS for styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #24451F;
        color: #FFB000;
    }
    .centered-content {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        color: #FFB000;
    }
    .centered-content img {
        max-width: 60%;
        align-self: flex-end;
    }
    .custom-dropdown .stMultiSelect, .custom-dropdown .stTextInput {
        width: 100px !important;
        margin: 0 auto;
        color: #FFB000;
    }
    .stMultiSelect div {
        color: #FFB000 !important;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #FFB000 !important;
    }
    label {
        color: #FFB000 !important;
    }
    .custom-label {
        font-size: 24px !important;
    }
    .footer {
        text-align: center;
        color: #FFB000;
        margin-top: 50px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# centered container... did not center the logo at all by the way
st.markdown('<div class="centered-content">', unsafe_allow_html=True)

# display the logo
logo = Image.open('logo.png')
st.image(logo, width=450) 


# mission Statement
st.markdown("## Mission Statement")
st.markdown("""
Here at Mad Cat Cannabis Co. we help you Discover Your Roar by providing personalized strain recommendations based on your individual symptoms.
""")

# dropdown for symptoms/effects
symptoms_effects_list = [
    'ADD/ADHD', 'Alzheimer\'s', 'Anorexia', 'Anxiety', 'Appetite Loss', 'Arthritis',
    'Asthma', 'Autism', 'Bipolar Disorder', 'Cancer', 'Chronic Pain', 'Cramps',
    'Crohn\'s Disease', 'Depression', 'Epilepsy', 'Eye Pressure', 'Fatigue', 
    'Fibromyalgia', 'Gastrointestinal Disorder', 'Glaucoma', 'Headaches', 
    'HIV/AIDS', 'Hypertension', 'Inflammation', 'Insomnia', 'Irritable Bowel Syndrome',
    'Loss of Appetite', 'Migraines', 'Mood Swings', 'Multiple Sclerosis', 
    'Muscle Spasms', 'Narcolepsy', 'Nausea', 'Neuropathy', 'Nightmares', 
    'Parkinson\'s', 'PMS', 'PTSD', 'Seizures', 'Spasticity', 'Spinal Cord Injury', 
    'Stress', 'Tinnitus', 'Tremors', 'Focused', 'Energetic', 'Creative', 'Aroused', 
    'Talkative', 'Relaxed', 'Calm', 'Hungry', 'Euphoric', 'Giggly', 'Uplifted', 
    'Sleepy', 'Happy', 'Energized'
]

# add a CSS class to the dropdown
st.markdown('<div class="custom-dropdown">', unsafe_allow_html=True)
# get user input from dropdown with custom label class
st.markdown('<div class="custom-label">Select Symptom(s) or Effect(s):</div>', unsafe_allow_html=True)
selected_symptoms_or_effects = st.multiselect('', symptoms_effects_list)
st.markdown('</div>', unsafe_allow_html=True)

# combine dropdown values
combined_input = ','.join(selected_symptoms_or_effects)

# button to trigger recommendations
if st.button('Get Recommendations'):
    # Call your recommendation function
    recommendations_df = recommendations(combined_input, similarity_matrix=similarity_matrix, weed=weed)

    # display recommendations
    if isinstance(recommendations_df, pd.DataFrame):
        st.subheader('Top Recommendations:')
        st.table(recommendations_df[['strain', 'type', 'rating', 'flavor']])
    else:
        st.write(recommendations_df)

# close centered container
st.markdown('</div>', unsafe_allow_html=True)


# footer
st.markdown(
    """
    <div class="footer">
        <p>As always, rolled with Love<br>- Bailee Buckner <3<br>Founder and CEO</p>
    </div>
    """,
    unsafe_allow_html=True
)