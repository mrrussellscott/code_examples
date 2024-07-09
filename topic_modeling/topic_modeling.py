import pandas as pd
import matplotlib.pyplot as plt
import nltk
# from sklearn.feature_extraction.text import CountVectorizer
import gensim
from gensim import corpora
from gensim.models import CoherenceModel
from gensim import corpora, models
from gensim.models import Phrases
from gensim.models.phrases import Phraser
import pyLDAvis.gensim_models

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

def process_columns(df: pd.DataFrame) -> list:
    df.columns = [col.lower() for col in df.columns]
    df.columns = [col.replace(' ', '_') for col in df.columns]
    df.columns = [col.replace('/', '_') for col in df.columns]
    df['other_activities'] = df['other_activities'].fillna('')
    df['other_interactions'] = df['other_interactions'].fillna('')

    df['other_activities'] = df['other_activities'].apply(lambda x: str(x) + ' ')
    df['other_'] = df['other_activities'] + df['other_interactions'] # df['basic_action_text'] +

    return [row for row in df['other_']]

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    return tokens

# find num of topics 
def compute_coherence_values(dictionary, corpus, texts, limit=10, start=2, step=1):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=42)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values

def run(get_optimal_num_topics=False):
    df = pd.read_csv("data-science-test-main/data/2018_Central_Park_Squirrel_Census_-_Squirrel_Data.csv", parse_dates=["Date"])
    texts = process_columns(df)

    # Preprocess Text Data
    processed_texts = [preprocess_text(text) for text in texts]
    bigram = Phrases(processed_texts, min_count=3, threshold=1)
    bigram_mod = Phraser(bigram)
    bigram_texts = [bigram_mod[text] for text in processed_texts]
    
    # Create a dictionary representation of the documents.
    dictionary = corpora.Dictionary(bigram_texts)
    corpus = [dictionary.doc2bow(text) for text in bigram_texts]

    if get_optimal_num_topics:
        # Compute coherence scores
        model_list, coherence_values = compute_coherence_values(dictionary, corpus, bigram_texts)

        # Plot coherence scores
        x = range(2, 10, 1)
        plt.plot(x, coherence_values)
        plt.xlabel("Number of Topics")
        plt.ylabel("Coherence Score")
        plt.title("Coherence Score by Number of Topics")
        plt.show()
        exit(1)

    # Set the number of topics
    num_topics = 5

    # Train the LDA model
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10, random_state=999)
    # Print the top 5 topics
    top_topics = lda_model.print_topics(num_topics=num_topics, num_words=10)
    for topic in top_topics:
        print(topic)
    
    def get_topic_words(x):
        return top_topics[int(x)][1:] # dom topic
    
    def get_topic_distribution(text_bow):
        return lda_model[text_bow]

    def get_dominant_topic(topic_distribution):
        return max(topic_distribution, key=lambda x: x[1])[0]

    # Apply the LDA model to each document and get the dominant topic
    df['topic_distribution'] = [get_topic_distribution(bow) for bow in corpus]
    df['dominant_topic'] = df['topic_distribution'].apply(get_dominant_topic)
    df['topic_words'] = df['dominant_topic'].apply(get_topic_words)
    
    df.to_csv('data-science-test-main/data/squirrels_data_tm.csv', index=False)

    vis = gensimvis.prepare(lda_model, corpus, dictionary)
    pyLDAvis.show(vis, local=False)


if __name__ == '__main__':
    run(get_optimal_num_topics=False)