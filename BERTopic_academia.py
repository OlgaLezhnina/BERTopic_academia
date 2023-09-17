import sys, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import nltk
from pathlib import Path
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer # ,util
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from metapub import PubMedFetcher
from spacy.lang.en import English
from tqdm import tqdm
import plotly
from bertopic.representation import MaximalMarginalRelevance
from transformers import pipeline
import networkx as nx
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.evaluation_metrics.coherence_metrics import Coherence

os.chdir(r"C:\Python")
store_folder = Path('Remo/scrapped_data15') 
store_folder.mkdir(parents = True, exist_ok = True) 
parser = English()
plotly.io.renderers.default='browser'
nltk.download('wordnet')
nltk.download('stopwords')


query = "(depression[Title/Abstract] OR depressed[Title/Abstract] OR depressive[Title/Abstract] \
OR anxiety[Title/Abstract] OR burnout[Title/Abstract])    \
AND ((research assistant[Title/Abstract] OR  PhD candidate[Title/Abstract]\
OR ((student[Title/Abstract]) NOT (Student t-test)) \
OR postdoc[Title/Abstract] OR postdoctoral[Title/Abstract] \
OR predoc[Title/Abstract] OR pre-doc[Title/Abstract] \
OR graduate[Title/Abstract] OR postgraduate[Title/Abstract] \
OR early career researcher[Title/Abstract] OR early-career researcher[Title/Abstract] \
OR early career scientist[Title/Abstract] OR early stage researcher[Title/Abstract]) \
AND (eng[la]) AND (adult[mh]) AND (humans[mh]) AND (hasabstract))"         
### 
def download_abstracts(query, store_folder):
    fetch = PubMedFetcher()
    #
    article_ids = fetch.pmids_for_query(query, retmax= 3000) 
    len(article_ids)
    def convert(article):
        attributes = [
            'abstract', 'citation', 'citation_html', 
            'doi', 'issn', 'issue', 'journal', 
            'publication_types', 'pubmed_type', 'title',            
            'url', 'volume', 'volume_issue', 'xml', 'year']
        data = {}
        for attr_name in attributes:
            attrib = getattr(article, attr_name)
            try:
                pickle.dumps(attrib)
                data[attr_name] = getattr(article, attr_name)
            except Exception:
                print("!!! bad attribute found -", attr_name)
        return data      
    
    for article_id in tqdm(article_ids):
        article = fetch.article_by_pmid(article_id)
        data = convert(article)
        with open(store_folder/article_id, 'wb') as file:
            pickle.dump(data, file) 

# download_abstracts(query, store_folder)            

### load the stored data
def load_info():
    info = []
    for child in sorted(store_folder.iterdir()):
        with open(child, 'rb') as file:
            data = pickle.load(file)
        abstract = data['abstract']        
        try:
            year = int(data['year'])            
        except:
            year = np.nan 
        doi = data['doi']   
        journal = data['journal']            
        citation = data['citation']            
        info.append([abstract, year, doi, journal, citation])
    info = pd.DataFrame(info, columns=['abstract', 'year', 'doi', 'journal', 'citation'])
    return info
#
data_full = load_info()
# remove missing abstracts
data_full = data_full.dropna(subset = ['abstract'])
# check for unique doi
seen = set()
dupes = []
for i, doi in enumerate(data_full['doi']):
    if doi == None:
        continue  
    if doi not in seen:
        seen.add(doi)
    else:
        dupes.append(i)
min(data_full.year)
max(data_full.year)
sns.histplot(data_full['year'])
len(data_full['journal'].unique()) 

docs = data_full.abstract.values

# instantiate BERTopic
# extract embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
# reduce dimensionality
umap_model = UMAP(
    n_neighbors = 15, n_components = 5, min_dist = 0.0, 
    metric = 'cosine', random_state = 34)
# cluster reduced embeddings
hdbscan_model = HDBSCAN(
    min_cluster_size = 30, min_samples = 10,  
    metric = 'euclidean', cluster_selection_method = 'eom', 
    prediction_data = True)
# tokenize topics
vectorizer_model = CountVectorizer(stop_words = "english")
# create topic representation
ctfidf_model = ClassTfidfTransformer()
# fine tune with relevance
representation_model = MaximalMarginalRelevance(diversity = 0.2)
# all steps together
topic_model = BERTopic(
  embedding_model = embedding_model,    # Step 1 - Extract embeddings
  umap_model = umap_model,              # Step 2 - Reduce dimensionality
  hdbscan_model = hdbscan_model,        # Step 3 - Cluster reduced embeddings
  vectorizer_model = vectorizer_model,  # Step 4 - Tokenize topics
  ctfidf_model = ctfidf_model,          # Step 5 - Extract topic words        
  calculate_probabilities = True,        
  verbose = True,
  representation_model = representation_model # Diversify topic words
)

# fit the model
topics, probs = topic_model.fit_transform(docs)
topic_model.get_topic_info() 
# info about all docs assigned to topics
documents = topic_model.get_document_info(docs) 

# get TC and TD metrics
documents_per_topic = documents.groupby(['Topic'], as_index = False).agg({'Document': ' '.join})
cleaned_docs = topic_model._preprocess_text(documents_per_topic.Document.values)
analyzer = topic_model.vectorizer_model.build_analyzer()
tokens = [analyzer(doc) for doc in cleaned_docs]
bertopic_topics = [
    [topicwords[0] for topicwords in topic_model.get_topic(i)[:10]]
    for i in range(len(set(topics)) - 1)]
TC = Coherence(texts = tokens, topk = 10, measure = 'c_v').score({'topics': bertopic_topics}) 
TD = TopicDiversity().score({'topics': bertopic_topics})
print('TC = ', TC, 'TD = ', TD)


# take a topic of interest
MY_TOPIC = 5
# get ten key words for a specific topic
topic_model.get_topic(MY_TOPIC) 
# get docs assigned to a specific topic
assigned_docs = documents[documents.Topic == MY_TOPIC] 
for abstract in assigned_docs.Document:
    print(abstract)
    print('---')

# representarive docs for a specific topic
representative_docs = topic_model.get_representative_docs(MY_TOPIC)
# a specific citation 
data_full[data_full.abstract == representative_docs[1]].iloc[0].citation
    
# classify labels for a specific topic
classifier = pipeline("zero-shot-classification", model = "facebook/bart-large-mnli")
sequence_to_classify =  " ".join([word for word, _ in topic_model.get_topic(MY_TOPIC)])
candidate_labels = ["Psychometrics of depression"]
classifier(sequence_to_classify, candidate_labels)

### Visualise results ###

# barchart for the first six topics
barchart = topic_model.visualize_barchart(top_n_topics = 6, n_words = 5, width = 400, title = "")
barchart.show()

# intertopic map
topic_model.visualize_topics().show()

# heatmap
topic_model.visualize_heatmap(n_clusters = 7)

# hierarchical clustering
hierarchical_topics = topic_model.hierarchical_topics(docs)
topic_model.visualize_hierarchy(hierarchical_topics = hierarchical_topics)

# documents embeddings 
embeddings = embedding_model.encode(docs, show_progress_bar = False)
topic_model.visualize_documents(docs, embeddings = embeddings)   
topic_model.visualize_hierarchical_documents(docs, hierarchical_topics, embeddings = embeddings) 
 
# topics over time for specific topics
time = data_full.year.to_list() 
topics_over_time = topic_model.topics_over_time(docs, time)
topic_model.visualize_topics_over_time(topics_over_time, topics = [0, 4, 5, 20])

# topics relevant for a search term 
def draw_simil(keyword):
    topicsF, similarity = topic_model.find_topics(keyword, top_n = 3)
    # the vertices for the graph
    G = nx.DiGraph()
    G.add_node(-1)
    G.add_node(0)
    G.add_node(1)
    G.add_node(2)
    node_labels = dict(enumerate(topicsF))
    node_labels[-1] = keyword
    scale = (len(keyword) + 12) * 0.032
    x_offset = 0.25 * scale
    y_offset = -.5
    pos = {
        -1: (0, 0),
        0: (-x_offset * scale, y_offset),
        1: (0, y_offset),
        2: (x_offset * scale, y_offset)
    }
    nx.draw(
        G, pos = pos, labels = node_labels, with_labels = True, node_shape = "s", 
        bbox = dict(facecolor = "lightgreen", edgecolor = 'black', boxstyle = 'round,pad=0.5'))
    # the edges for the graph (the separate graph to change the edges' positions)
    H = nx.DiGraph()
    for i, sim_val in enumerate(similarity):
        H.add_edge(i + 3, i, label = round(sim_val, 2)) # 3->0, 4->1, 5->2
    y_offset = -.48
    pos = {
        0: (-x_offset * scale, y_offset),
        1: (0, y_offset),
        2: (x_offset * scale, y_offset),
        3: (-x_offset * scale, 0),
        4: (0, 0),
        5: (x_offset * scale, 0)
    }
    edge_labels = nx.get_edge_attributes(H, "label")
    nx.draw_networkx(H, labels = {}, node_color = "white", arrows = True, pos = pos)
    nx.draw_networkx_edge_labels(H, pos, edge_labels)
    # set margins for the axes 
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    ax = plt.gca()
    ax.margins(0.20)
    plt.axis("off")
    plt.show()
#
draw_simil('internet addiction')
draw_simil('burnout measures')

###