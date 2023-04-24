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


def split_chatgpt_args(gpt_args, chunk_size):
    for i in range(0, len(gpt_args), chunk_size):
        yield gpt_args[i:i + chunk_size]


def n_best_chatgpt_arguments(texts, query, chatbot, n):
    arguments = []
    for text in texts:
        arguments.append(retrieve_arguments_with_chatgpt(query, text, chatbot))

    first_round = list(split_chatgpt_args(arguments, 4))
    candidates = []
    i = 1
    j = 1
    print("First Round: \n")
    for section in first_round:
        print(f"{i} / {len(first_round)}")
        i += 1
        for data in chatbot.ask(f"These are some arguments you gave me: {str(section)}. Give the {n} best arguments that fit the most to this query: {query}"):
            response = data["message"]
        candidates.append(response)
    
    second_round = list(split_chatgpt_args(candidates, 3))
    finalists = []
    print("Second Round: \n")
    for section in second_round:
        print(f"{j} / {len(second_round)}")
        j += 1
        for data in chatbot.ask(f"These are some arguments you gave me: {str(section)}. Give the {n} best arguments that fit the most to this query: {query}"):
            response = data["message"]
        finalists.append(response)
    
    for data in chatbot.ask(f"These are some arguments you gave me: {str(finalists)}. Give the {n} best arguments that fit the most to this query: {query}"):
        response = data["message"]
    
    return response
    

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
    try:
        for data in chatbot.ask(f"This is a text: \"{text}\". Return arguments that you can inference from this text matching to this query: \"{query}\". You should only give me arguments that can be inferred fom the text. Give me the arguments in a list split in PRO and CONTRA arguments and please give me the line for each argument from where it can be inferenced."):
            response = data["message"]
    except:
        response = ""
    finally:
        return response
    
    
def main():
    # Initialize ChatBot for ChatGPT
    chatbot = Chatbot(config=
        json.load(open("access_token.json"))
    )

    # Initialize model for BERT
    model = SentenceTransformer('all-MiniLM-L6-v2')

    queries = parse_topics()[0:30]
    gpt_args = []
    i = 0
    for query in queries:
        relevant_texts = retrieve_texts_matching_to_query(query)
        joined_texts = " ".join(map(str, relevant_texts))

        jaccard_args = retrieve_arguments_with_jaccard(query, joined_texts)
        bm25_args = retrieve_arguments_with_bm25(query, joined_texts)
        bert_args = retrieve_arguments_with_bert(query, joined_texts, model)
        tfidf_args = retrieve_arguments_with_tfidf(query, joined_texts)
        gpt_args = n_best_chatgpt_arguments(relevant_texts, query, chatbot, 5)


if __name__ == "__main__":
    main()