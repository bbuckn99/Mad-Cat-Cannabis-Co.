# Capstone Project: Mad Cat Cannabis Co. Strain Recommendation System

## Project Overview
Welcome to the Mad Cat Cannabis Co. Strain Recommendation System! This project aims to help users discover the best cannabis strains tailored to their specific symptoms and desired effects. By leveraging natural language processing and machine learning techniques, this system provides personalized strain recommendations based on user inputs.


## Introduction
Cannabis strains have a wide range of effects and benefits, making it challenging for users to find the right strain for their needs. This recommendation system uses user-provided symptoms and effects to suggest the most suitable strains, enhancing the user's experience and satisfaction.


## Features
- Personalized Recommendations: Get strain suggestions based on specific symptoms or desired effects

- Interactive User Interface: A user-friendly web interface built with Streamlit. To try this on your own, go to [https://streamlit.io/] and run the canna.py file in this repo

- Data Visualization: Visual insights into the effects, flavors, and ratings of different strains

- Advanced NLP and ML Techniques: Utilizing TF-IDF Vectorization and Cosine Similarity to process and compare strain data


## Data and Data Dictionary
The dataset used in this project is
[https://www.dolthub.com/repositories/dolthub/marijuana-data/query/master?q=SELECT+*+FROM+%60leafly%60+ORDER+BY+%60strain%60+ASC%2C+%60effects%60+ASC+LIMIT+1000]
It includes information about various cannabis strains, their effects, flavors, ratings and descriptions

| Attribute           | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| `strain`              | Name of the cannabis strain.                                                |
| `type`                | Type of the strain (e.g., indica, sativa, hybrid).                          |
| `rating`              | Rating of the strain.                                                       |
| `effects`             | Effects produced by the strain.                                              |
| `flavor`              | Flavor profile of the strain.                                                |
| `description`         | Description of the strain.                                                   |
| `symptom_to_effects`  | Dictionary mapping symptoms to corresponding effects.                        |
| `symptoms`            | Comma-separated string of symptoms related to the effects.                   |
| `symptoms_and_effects`| Combined string of symptoms and effects, converted to lowercase.             |
| `combined_matrix`     | TF-IDF vector representation of symptoms and effects.                         |
| `similarity_matrix`   | Cosine similarity matrix based on TF-IDF vectors.                             |
| `normalize(series)`   | Function to normalize a pandas Series to the range [0, 1].                    |
| `recommendations(...)`| Function to generate content-based recommendations for strains.              |
| Output DataFrame      | - `strain`: Name of the recommended strain.                                   |
|                        | - `type`: Type of the recommended strain (indica, sativa, hybrid).            |
|                        | - `rating`: Rating of the recommended strain.                                 |
|                        | - `hybrid_score`: Combined score based on similarity and rating.             |


## Model
The recommendation model uses the following steps:

- Data Preprocessing: Cleaning and combining symptoms and effects data

- TF-IDF Vectorization: Converting text data into numerical vectors

- Cosine Similarity: Calculating similarity scores between strains

- Normalization: Standardizing rating values for balanced weighting

- Hybrid Scoring: Combining similarity and normalized ratings to generate recommendations.


## Results
The system provides a list of top recommended strains based on user inputs. It also visualizes the distribution of effects and flavors using word clouds and various graphs giving users an intuitive understanding of the available options. The Streamlit app gives users an interactive experience with the model to see how it would work in a real world application. 


## Future Work
- Better Dataset: The DoltHub dataset hasn’t been updated in four years, so the model is not up to date with current strains 

- Improve UI/UX: Enhance the user interface for better interaction

- More Diverse Effects and Symptoms: A more comprehensive list of symptoms and their effects would improve the model’s recommendations. The model tends to recommend the same strains for different symptoms if the effects for the symptoms are the same or too similar

- Advanced Filtering: Allow users to filter recommendations based on additional criteria like potency, price or terpene profile

- Enhanced Model: Experiment with more sophisticated machine learning models to improve recommendation accuracy
  
- Collaboration with Experts: Collaborate with cannabis industry experts, healthcare professionals, researchers, and regulatory entities to validate recommendations
  
- User Response Data: In order to see if the model is properly working and fine tune it, user data should be collected and added to see if the recommended strain is accurately treating their symptoms
  
- The Future: If used and implemented properly, this could cut down on time spent in dispensaries, less time and money spent on cannabis that doesn’t work for you. All in all, our model promotes responsible use and informed decision-making

