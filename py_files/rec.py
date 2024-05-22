import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

file_path = r"C:\Users\Baile\Documents\DSI 24\capstone\clean_weed.csv"
weed = pd.read_csv(file_path)

# symptom to effects mapping
symptom_to_effects = {
    'ADD/ADHD': ['Focused', 'Energetic', 'Creative', 'Aroused', 'Talkative'],
    'Alzheimer\'s': ['Relaxed', 'Calm'],
    'Anorexia': ['Hungry', 'Relaxed', 'Euphoric', 'Giggly'],
    'Anxiety': ['Calm', 'Relaxed'],
    'Appetite Loss': ['Hungry'],
    'Arthritis': ['Relaxed', 'Calm'],
    'Asthma': ['Relaxed', 'Calm'],
    'Autism': ['Calm', 'Relaxed'],
    'Bipolar Disorder': ['Calm', 'Uplifted'],
    'Cancer': ['Relaxed', 'Hungry', 'Giggly', 'Happy', 'Euphoric'],
    'Chronic Pain': ['Relaxed', 'Calm'],
    'Cramps': ['Relaxed', 'Euphoric'],
    'Crohn\'s Disease': ['Relaxed', 'Hungry'],
    'Depression': ['Happy', 'Uplifted', 'Euphoric'],
    'Epilepsy': ['Calm', 'Relaxed'],
    'Eye Pressure': ['Calm', 'Relaxed', 'Sleepy'],
    'Fatigue': ['Energized', 'Energetic'],
    'Fibromyalgia': ['Relaxed', 'Calm'],
    'Gastrointestinal Disorder': ['Hungry', 'Relaxed'],
    'Glaucoma': ['Calm', 'Relaxed', 'Happy'],
    'Headaches': ['Relaxed', 'Calm'],
    'HIV/AIDS': ['Hungry', 'Relaxed'],
    'Hypertension': ['Calm', 'Relaxed'],
    'Inflammation': ['Calm', 'Relaxed'],
    'Insomnia': ['Sleepy', 'Relaxed'],
    'Irritable Bowel Syndrome': ['Hungry', 'Relaxed'],
    'Loss of Appetite': ['Hungry'],
    'Migraines': ['Relaxed', 'Calm'],
    'Mood Swings': ['Calm', 'Uplifted'],
    'Multiple Sclerosis': ['Relaxed', 'Calm'],
    'Muscle Spasms': ['Relaxed'],
    'Narcolepsy': ['Energetic', 'Energized'],
    'Nausea': ['Relaxed', 'Calm', 'Giggly'],
    'Neuropathy': ['Calm', 'Relaxed'],
    'Nightmares': ['Calm', 'Relaxed'],
    'Parkinson\'s': ['Calm', 'Relaxed'],
    'PMS': ['Calm', 'Relaxed', 'Euphoric'],
    'PTSD': ['Calm', 'Relaxed', 'Happy', 'Euphoric'],
    'Seizures': ['Calm', 'Relaxed'],
    'Spasticity': ['Relaxed'],
    'Spinal Cord Injury': ['Relaxed'],
    'Stress': ['Calm', 'Relaxed', 'Happy', 'Euphoric'],
    'Tinnitus': ['Calm', 'Relaxed', 'Uplifted'],
    'Tremors': ['Calm', 'Relaxed']
}

# map effects to symptoms
def map_effects_to_symptoms(effects, symptom_to_effects):
    """Map effects to corresponding symptoms."""
    effects_list = effects.split(',')
    symptoms = set()
    for symptom, effect_list in symptom_to_effects.items():
        if any(effect.strip() in effects_list for effect in effect_list):
            symptoms.add(symptom)
    return ','.join(symptoms)

# apply the mapping to the dataframe
weed['symptoms'] = weed['effects'].apply(lambda x: map_effects_to_symptoms(x, symptom_to_effects))

# combine symptoms and effects
weed['symptoms_and_effects'] = weed.apply(lambda row: ','.join(filter(None, [row['symptoms'], row['effects']])), axis=1)

# convert the symptoms_and_effects column to lowercase
weed['symptoms_and_effects'] = weed['symptoms_and_effects'].str.lower()

# vectorize the combined symptoms_and_effects column using TF-IDF
vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(','))
combined_matrix = vectorizer.fit_transform(weed['symptoms_and_effects'])

# cosine similarity matrix
similarity_matrix = cosine_similarity(combined_matrix)

# normalize values
def normalize(series):
    """Normalize a pandas Series to the range [0, 1]."""
    if series.max() == series.min():
        return np.zeros_like(series)
    return (series - series.min()) / (series.max() - series.min())

# recommendation system
def recommendations(*symptoms_or_effects, similarity_matrix, weed, top_n=5, weight_similarity=0.5, weight_rating=0.5):
    """
    Get content-based recommendations for strains based on given symptoms or effects.
    
    Parameters:
    - *symptoms_or_effects (str): The symptoms or effects to base recommendations on.
    - similarity_matrix (ndarray): Precomputed cosine similarity matrix.
    - weed (DataFrame): The DataFrame containing strain data.
    - top_n (int): The number of top recommendations to return.
    - weight_similarity (float): The weight for similarity in the hybrid score.
    - weight_rating (float): The weight for rating in the hybrid score.
    
    Returns:
    - DataFrame: The top recommended strains sorted by the hybrid score.
    """
    # convert the input to lowercase to ensure case insensitivity
    symptoms_or_effects = [symptom_or_effect.lower() for symptom_or_effect in symptoms_or_effects]

    # array to store average similarity scores
    average_similarity_scores = np.zeros(len(weed))

    # track whether any symptoms or effects matched
    any_match = False

    for symptom_or_effect in symptoms_or_effects:
        # strains with the given symptom or effect
        try:
            indices = weed[weed['symptoms_and_effects'].str.contains(symptom_or_effect)].index
            any_match = True
        except IndexError:
            continue
        
        # add the average similarity scores for all matching indices
        for idx in indices:
            similarity_scores = similarity_matrix[idx]
            average_similarity_scores[idx] += similarity_scores.mean()

    if not any_match:
        return "No strains found with the given symptom(s) or effect(s)."

    # normalize the average similarity scores
    normalized_similarity = normalize(average_similarity_scores)

    # top_n indices based on average similarity scores
    similar_strains_indices = normalized_similarity.argsort()[-top_n:][::-1]

   # create a DataFrame with top_n strains and their relevant attributes
    similar_strains = []
    for idx in similar_strains_indices:
        similar_strains.append({
            'strain': weed.loc[idx, 'strain'],
            'type': weed.loc[idx, 'type'],
            'rating': weed.loc[idx, 'rating'],
            'flavor': weed.loc[idx, 'flavor']
        })
    return pd.DataFrame(similar_strains)