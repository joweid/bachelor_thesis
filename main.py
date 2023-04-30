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
import numpy as np
import random
import time
import openai
from retry import retry


def retrieve_resolved_texts_matching_to_query(query):
    file = open("resolved_arguments.json")
    data = json.load(file)
    relevant_texts = []

    for line in data['lines']:
        text = line['text']
        original_query = line['query']

        if original_query.lower() == query.lower():
            relevant_texts.append(text)
        
    return relevant_texts  


def retrieve_original_texts_matching_to_query(query):
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


def n_best_chatgpt_arguments(texts, query, chatbot, n, expanded):
    arguments = []
    i = 0
    for text in texts:
        print(f"{i} / {len(texts)}")
        i+=1
        arguments.append(retrieve_arguments_with_chatgpt(query, text, chatbot, expanded))

    first_round = list(split_chatgpt_args(arguments, 4))
    candidates = []
    i = 1
    j = 1
    print("First Round: \n")
    for section in first_round:
        print(f"{i} / {len(first_round)}")
        i += 1

        prompt = ""
        if not expanded:
            prompt = f"These are some arguments you gave me: {str(section)}. Give the {n} best arguments that fit the most to this query: {query}"
        else:
            prompt = f"These are some arguments you gave me: {str(section)}. Give the {n} best arguments that fit the most to these queries: {str(query)}"
        
        for data in chatbot.ask(prompt):
            response = data["message"]

        candidates.append(response)
    
    second_round = list(split_chatgpt_args(candidates, 3))
    finalists = []
    print("Second Round: \n")
    for section in second_round:
        print(f"{j} / {len(second_round)}")
        j += 1

        prompt = ""
        if not expanded:
            prompt = f"These are some arguments you gave me: {str(section)}. Give the {n} best arguments that fit the most to this query: {query}"
        else:
            prompt = f"These are some arguments you gave me: {str(section)}. Give the {n} best arguments that fit the most to these queries: {str(query)}"
        
        for data in chatbot.ask(prompt):
            response = data["message"]

        finalists.append(response)
    
    prompt = ""
    if not expanded:
        prompt = f"These are some arguments you gave me: {str(finalists)}. Give the {n} best arguments that fit the most to this query: {query}"
    else:
        prompt = f"These are some arguments you gave me: {str(section)}. Give the {n} best arguments that fit the most to these queries: {str(query)}"
    

    for data in chatbot.ask(prompt):
        response = data["message"]
    
    return response


def deduplicate_arguments(arguments, original_text):
    candidates = sorted(arguments, key=lambda k: k['score'], reverse=True)[:30]
    for arg in candidates:
        arg['is_resolved'] = is_resolved(arg['text'], original_text)

    unique_texts = set()
    unique_candidates = []

    for candidate in candidates:
        if candidate['text'] not in unique_texts:
            unique_texts.add(candidate['text'])
            unique_candidates.append(candidate)


    top_five = [f"{argument['text']} (originated_from: {argument['comes_from']}, resolved_corefs: {argument['is_resolved']})" for argument in unique_candidates[:5]]

    return top_five


def is_resolved(sentence, unresolved_text):
    return sentence not in sent_tokenize(unresolved_text)
    

def retrieve_arguments_with_jaccard(query, text, original_text):
    sentences = sent_tokenize(text)
    arguments = []
    for sentence in sentences:
        jaccard_score = jaccard(query, sentence)
        arguments.append({"text": sentence, "jaccard": jaccard_score})

    top_five = sorted(arguments, key=lambda k: k['jaccard'], reverse=True)[:5]
    top_five = [f"{argument['text']} (resolved_corefs: {is_resolved(argument['text'], original_text)})" for argument in top_five]

    return top_five


def retrieve_arguments_with_jaccard_expanded(queries, text, original_text):
    sentences = sent_tokenize(text)
    arguments = []

    for sentence in sentences:
        for query in queries:
            jaccard_score = jaccard(query, sentence)
            arguments.append({"text": sentence, "score": jaccard_score, 'comes_from': query})

    return deduplicate_arguments(arguments, original_text)


def retrieve_arguments_with_bm25(query, text, original_text):
    corpus = sent_tokenize(text)
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    
    tokenized_query = query.lower().split(" ")

    scores = bm25.get_scores(tokenized_query)
    score_matcher = {}
    doc_matcher = []

    for index, score in enumerate(scores):
        score_matcher[index] = score
    
    top_n = sorted(score_matcher.items(), key=lambda x: x[1], reverse=True)[:5]
    for identifier in top_n:
        index = identifier[0]
        score = identifier[1]
        doc_matcher.append({'score': score, 'text': corpus[index], 'comes_from': query})
        
    return deduplicate_arguments(doc_matcher, original_text)


def retrieve_arguments_with_bm25_expanded(queries, text, original_text):
    corpus = sent_tokenize(text)
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    doc_matcher = []

    for query in queries:
        tokenized_query = query.lower().split(" ")
        scores = bm25.get_scores(tokenized_query)
        score_matcher = {}

        for index, score in enumerate(scores):
            score_matcher[index] = score
        
        top_n = sorted(score_matcher.items(), key=lambda x: x[1], reverse=True)[:5]
        for identifier in top_n:
            index = identifier[0]
            score = identifier[1]
            doc_matcher.append({'score': score, 'text': corpus[index], 'comes_from': query})
        
    return deduplicate_arguments(doc_matcher, original_text)
        

def retrieve_arguments_with_bert(query, text, model, original_text):
    sentences = sent_tokenize(text)
    arguments = []
    i = 0
        
    for sentence in sentences:
        printProgressBar(i, len(sentences), prefix="BERT", suffix="Complete")
        i += 1
        query_embedding = model.encode(query.lower())
        sentence_embedding = model.encode(sentence)

        cos_sim_output = str(util.cos_sim(query_embedding, sentence_embedding)[0][0])
        cosine_similarity = float(cos_sim_output[cos_sim_output.rfind('(')+1:cos_sim_output.rfind(')')].strip())
        arguments.append({"text": sentence, "score": cosine_similarity})
    
    
    top_five = sorted(arguments, key=lambda k: k['score'], reverse=True)[:5]
    top_five = [f"{argument['text']} (resolved_corefs: {is_resolved(argument['text'], original_text)})" for argument in top_five]

    return top_five


def retrieve_arguments_with_bert_expanded(queries, text, model, original_text):
    sentences = sent_tokenize(text)
    arguments = []
    i = 0

    for sentence in sentences:
        printProgressBar(i, len(sentences), prefix="BERT", suffix="Complete")
        i += 1
        sentence_embedding = model.encode(sentence)
        for query in queries:
            query_embedding = model.encode(query.lower())

            cos_sim_output = str(util.cos_sim(query_embedding, sentence_embedding)[0][0])
            cosine_similarity = float(cos_sim_output[cos_sim_output.rfind('(')+1:cos_sim_output.rfind(')')].strip())
            arguments.append({"text": sentence, "score": cosine_similarity, 'comes_from': query})
    
    return deduplicate_arguments(arguments, original_text)

def retrieve_arguments_with_tfidf(query, text, original_text):
    corpus = sent_tokenize(text)
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    query_vector = tfidf_vectorizer.transform([query.lower()])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix)

    most_similar_indices = cosine_similarities.argsort()[0][::-1]
    five_most_relevant_args = n_most_relevant(most_similar_indices, corpus, 5)
    top_n = [f"{corpus[i]} (resolved_corefs: {is_resolved(corpus[i], original_text)})" for i in most_similar_indices[: 5]]

    return top_n


def retrieve_arguments_with_tfidf_expanded(queries, text, original_text):
    corpus = sent_tokenize(text)
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    doc_matcher = []
    
    for query in queries:
        score_matcher = {}
        query_vector = tfidf_vectorizer.transform([query.lower()])
        cosine_similarities = cosine_similarity(query_vector, tfidf_matrix)
        scores = cosine_similarities[0]

        for index, score in enumerate(scores):
            score_matcher[index] = score
        
        top_n = sorted(score_matcher.items(), key=lambda x: x[1], reverse=True)[:5]
        for identifier in top_n:
            index = identifier[0]
            score = identifier[1]
            doc_matcher.append({'score': score, 'text': corpus[index], 'comes_from': query})
        
    return deduplicate_arguments(doc_matcher, original_text)

        
def retrieve_arguments_with_chatgpt(query, text, chatbot, expanded=False):
    try:
        prompt = ""
        if not expanded:
            prompt = f"This is a text: \"{text}\". Return arguments that you can inference from this text matching to this query: \"{query}\". You should only give me arguments that can be inferred fom the text. Give me the arguments in a list split in PRO and CONTRA arguments and please give me the line for each argument from where it can be inferenced."
        else:
            prompt = f"This is a text: \"{text}\". Return arguments that you can inference from this text matching to this queries: \"{str(query)}\". You should only give me arguments that can be inferred fom the text. Give me the arguments in a list split in PRO and CONTRA arguments and please give me the line for each argument from where it can be inferenced."
        
        for data in chatbot.ask(prompt):
            response = data["message"]
    except:
        response = ""
    finally:
        return 
    

def ask_chat_gpt(prompt):
    try:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=f"Human: {prompt}:",
            temperature=0.7,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["Human: ", "AI: "]
        )

        return response.choices[0].text
    except:
        print("Error detected...")
        time.sleep(30)
        return ""


@retry(exceptions=Exception, delay=10)
def chat_completion(text, prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            max_tokens=500,
            messages= [
                    {"role": "user", "content": text},
                    {'role': 'user', 'content': prompt}
                ]
        )
            
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(e)
        time.sleep(30)
        return ""


def split_long_text(text, chunk_size=300):
    words = text.split()
    chunks = [words[i:i+chunk_size] for i in range(0, len(words), chunk_size)]
    chunks = [" ".join(chunk) for chunk in chunks]
    return chunks


def discord_chatgpt_arguments(texts, query):
    arguments = []
    i = 0
    for text in texts:
        printProgressBar(i, len(texts), prefix="Text iteration: ", suffix="Complete", length=10)
        i+=1
        arguments.append(chat_completion(text, f"Return five arguments you can get from this text in a list matching to the query: {query}"))

    first_round = list(split_chatgpt_args(arguments, 4))
    candidates = []
    i = 1
    j = 1
    print("First Round: \n")
    for section in first_round:
        printProgressBar(i, len(first_round), prefix="First Round: ", suffix="Complete", length=10)
        i += 1

        response = chat_completion(f"{str(section)}", f"Give the five best arguments from these arguments that fit the most to this query: {query}")
        candidates.append(response)
    
    second_round = list(split_chatgpt_args(candidates, 3))
    finalists = []
    print("Second Round: \n")
    for section in second_round:
        printProgressBar(j, len(second_round), prefix="Second round: ", suffix="Complete", length=10)
        j += 1

        response = chat_completion(f"{str(section)}", f"Give the five best arguments from these arguments that fit the most to this query: {query}")
        finalists.append(response)
    
    final_response = chat_completion(f"{str(finalists)}", f"Give the five best arguments from these arguments that fit the most to this query: {query}")
    return final_response


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

    
def main():
    # Initialize ChatBot for ChatGPT
    chatbot = Chatbot(config=
        json.load(open("access_token2.json"))
    )

    # Discord Proxy
    #openai.api_key = json.load(open("discord_chatgpt_key.json"))['key']
    #openai.api_base = 'https://api.pawan.krd/v1'

    # China Proxy
    #openai.api_key = 'sk-0PfcSdT723UR44igwVxvEWvLoZJgi0FJyZWy0WCCATp5ka2a'
    #openai.api_base = 'https://api.chatanywhere.com.cn/v1'

    # Initialize model for BERT
    model = SentenceTransformer('all-MiniLM-L6-v2')

    output_json = {'outputs': []}

    queries = json.load(open('query_expansion.json'))
    for object in queries['query_expansion']:
        # Query Expansion
        if object['id'] == 13:
            break
        query = object['query']
        additional_queries = object['expansion']

        comparison = {'query': object['query'], 'query_expansion': additional_queries, 'results': {}}
        method_identifiers = [1, 2, 3, 4, 5]
        print(f"Text: {object['id']}/{len(queries['query_expansion'])}\n")
        
        print(f"Original Query: {query}\n\n")
        print(f"Query Expansion: {str(additional_queries)}\n\n")

        relevant_texts = retrieve_resolved_texts_matching_to_query(query)
        original_texts = retrieve_original_texts_matching_to_query(query)
        joined_texts = " ".join(map(str, relevant_texts))
        joined_original_texts = " ".join(map(str, original_texts))


        # JACCARD
        jaccard_args = retrieve_arguments_with_jaccard(query, joined_texts, joined_original_texts)
        jaccard_args_expanded = retrieve_arguments_with_jaccard_expanded(additional_queries, joined_texts, joined_original_texts)

        comparison['results']["jaccard"] = {'normal': jaccard_args, 'expanded': jaccard_args_expanded}
        print(f"Jaccard normal: {str(jaccard_args)} \n\n")
        print(f"Jaccard expanded: {str(jaccard_args_expanded)} \n\n")

        
        # BM25
        bm25_args = retrieve_arguments_with_bm25(query, joined_texts, joined_original_texts)
        bm25_args_expanded = retrieve_arguments_with_bm25_expanded(additional_queries, joined_texts, joined_original_texts)

        comparison['results']["bm25"] = {'normal': bm25_args, 'expanded': bm25_args_expanded}
        print(f"BM25 normal: {str(bm25_args)} \n\n")
        print(f"BM25 expanded: {str(bm25_args_expanded)} \n\n")
        

        # BERT
        print("BERT normal...")
        bert_args = retrieve_arguments_with_bert(query, joined_texts, model, joined_original_texts)
        print("BERT expanded...")
        bert_args_expanded = retrieve_arguments_with_bert_expanded(additional_queries, joined_texts, model, joined_original_texts)
        
        comparison['results']["bert"] = {'normal': bert_args, 'expanded': bert_args_expanded}
        print(f"BERT normal: {str(bert_args)} \n\n")
        print(f"BERT expanded: {str(bert_args_expanded)} \n\n")

        
        # TF-IDF
        tfidf_args = retrieve_arguments_with_tfidf(query, joined_texts, joined_original_texts)
        tfidf_args_expanded = retrieve_arguments_with_tfidf_expanded(additional_queries, joined_texts, joined_original_texts)
        
        comparison['results']["tfidf"] = {'normal': tfidf_args, 'expanded': tfidf_args_expanded}
        print(f"TF-IDF normal: {str(tfidf_args)} \n\n")
        print(f"TF-IDF expanded: {str(tfidf_args_expanded)} \n\n")
        
        comparison['results']["chatgpt"] = {'normal': "", 'expanded': ""}
        
        output_json['outputs'].append(comparison)
    
    json_data = json.dumps(output_json, indent=2)
    with open("outputs.json", "w") as file:
        file.write(json_data)
    

if __name__ == "__main__":
    main()