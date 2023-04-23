import json
from revChatGPT.V1 import Chatbot
from itertools import islice, tee
import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi


def retrieve_texts_matching_to_query(query):
    file = open("arguments.json")
    data = json.load(file)
    relevant_texts = []

    for line in data['lines']:
        text = line['text']
        original_query = line['query']

        if original_query.lower() == query.lower():
            relevant_texts.append(text)
        
    return relevant_texts  


def parse_topics():
    xml_tree = ET.parse('topics.xml')
    root = xml_tree.getroot()
    topics = []
    
    for topic in root.findall('topic'):
        query = topic.find('title').text
        topics.append(query)
    
    return topics


def jaccard(query, text): 
    query_4_grams = set(list(zip(*(islice(seq, index, None) for index, seq in enumerate(tee(query.lower(), 4))))))
    text_4_grams = set(list(zip(*(islice(seq, index, None) for index, seq in enumerate(tee(text.lower(), 4))))))
    
    return float(len(query_4_grams.intersection(text_4_grams)) / len(query_4_grams.union(text_4_grams)))


def n_most_relevant(most_similar_indices, corpus, limit):
    n_most_relevant = []
    l = 0

    for i in most_similar_indices:
        n_most_relevant.append(corpus[i])
        l += 1
        if l == limit:
            break

    return n_most_relevant


def retrieve_arguments_with_jaccard(query, text):
    sentences = sent_tokenize(text)
    arguments = []
    for sentence in sentences:
        jaccard_coefficient = jaccard(query.lower(), sentence)
        if jaccard_coefficient > 0.1:
            arguments.append({"text": sentence, "jaccard": jaccard_coefficient})

    top_10 = sorted(arguments, key=lambda k: k['jaccard'], reverse=True)[:5]
    top_10 = [argument['text'] for argument in top_10]

    return top_10


def retrieve_arguments_with_bm25(query, text):
    corpus = sent_tokenize(text)
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    tokenized_query = query.lower().split(" ")
    bm25 = BM25Okapi(tokenized_corpus)

    return bm25.get_top_n(tokenized_query, corpus, n=5)

   
def retrieve_arguments_with_bert(query, text, model):
    sentences = sent_tokenize(text)
    query_embedding = model.encode(query.lower())
    arguments = []

    for sentence in sentences:
        sentence_embedding = model.encode(sentence)
        cos_sim_output = str(util.cos_sim(query_embedding, sentence_embedding)[0][0])
        cosine_similarity = float(cos_sim_output[cos_sim_output.rfind('(')+1:cos_sim_output.rfind(')')].strip())
        if cosine_similarity > 0.5:
            arguments.append({"text": sentence, "score": cosine_similarity})
    
    top_10 = sorted(arguments, key=lambda k: k['score'], reverse=True)[:5]
    top_10 = [argument['text'] for argument in top_10]

    return top_10


def retrieve_arguments_with_tfidf(query, text):
    corpus = sent_tokenize(text)
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    query_vector = tfidf_vectorizer.transform([query.lower()])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix)

    most_similar_indices = cosine_similarities.argsort()[0][::-1]
    five_most_relevant_args = n_most_relevant(most_similar_indices, corpus, 5)

    return five_most_relevant_args


def retrieve_arguments_with_chatgpt(query, text, chatbot):
    for data in chatbot.ask(f"This is a text: \"{text}\". Return arguments that you can inference from this text matching to this query: \"{query}\". You should only give me arguments that can be inferred fom the text. Give me the arguments in a list split in PRO and CONTRA arguments and please give me the line for each argument from where it can be inferenced."):
        response = data["message"]
    
    return response


def main():
    # Initialize ChatBot for ChatGPT
    chatbot = Chatbot(config=
        json.load(open("access_token.json"))
    )

    # Initialize model for BERT
    model = SentenceTransformer('all-MiniLM-L6-v2')

    queries = parse_topics()
    for query in queries:
        relevant_texts = retrieve_texts_matching_to_query(query)
        for text in relevant_texts:
            #print(f"Text: \n{text}\n\n")
            jaccard_args = retrieve_arguments_with_jaccard(query, text)
            print(f"Jaccard: {jaccard_args}\n\n")
            bm25_args = retrieve_arguments_with_bm25(query, text)
            print(f"BM25: {bm25_args}\n\n")
            bert_args = retrieve_arguments_with_bert(query, text, model)
            print(f"BERT: {bert_args}\n\n")
            tfidf_args = retrieve_arguments_with_tfidf(query, text)
            print(f"TF-IDF: {tfidf_args}\n\n")
            chatgpt_args = retrieve_arguments_with_chatgpt(query, text, chatbot)
            print(f"GPT: {chatgpt_args}\n\n\n")
    

if __name__ == "__main__":
    main()