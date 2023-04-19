from revChatGPT.V1 import Chatbot

import json

from itertools import islice, tee


def jaccard(query, text): 
    query_4_grams = set(list(zip(*(islice(seq, index, None) for index, seq in enumerate(tee(query.lower(), 4))))))
    text_4_grams = set(list(zip(*(islice(seq, index, None) for index, seq in enumerate(tee(text.lower(), 4))))))
    
    return float(len(query_4_grams.intersection(text_4_grams)) / len(query_4_grams.union(text_4_grams)))

def retrieve_texts_matching_to_query(query):
    file = open("arguments.json")
    data = json.load(file)
    relevant_texts = []

    for line in data['lines']:
        text = line['text']
        original_query = line['query']

        #if (jaccard(keyword, text.lower()) > 0.05 or jaccard(keyword, original_query) > 0.05) and len(text) > 200:
        #    relevant_texts.append(text)
        if jaccard(query, text) > 0.05:
            relevant_texts.append(text)
        
    return relevant_texts   

def main():
    chatbot = Chatbot(config=
        json.load(open("access_token.json"))
    )

    query = "Are genetically modified foods healthy?"
    texts = retrieve_texts_matching_to_query(query)

    
    response = ""
    for text in texts:
        print(f"{text} \n\n")
        for data in chatbot.ask(f"This is a text: \"{text}\". Return arguments that you can inference from this text matching to this query: \"{query}\". You should only give me arguments that can be inferred fom the text. Give me the arguments in a list split in PRO and CONTRA arguments and please give me the line for each argument from where it can be inferenced."):
            response = data["message"]
        print(f"{response}\n\n\n")
    

if __name__ == "__main__":
    main()