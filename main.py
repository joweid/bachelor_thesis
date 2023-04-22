from revChatGPT.V1 import Chatbot

import json

from itertools import islice, tee
import xml.etree.ElementTree as ET


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
    
def main():
    chatbot = Chatbot(config=
        json.load(open("access_token.json"))
    )

    queries = parse_topics()

    """query = "Are genetically modified foods healthy?"
    texts = retrieve_texts_matching_to_query(query)

    
    response = ""
    for text in texts:
        print(f"{text} \n\n")
        for data in chatbot.ask(f"This is a text: \"{text}\". Return arguments that you can inference from this text matching to this query: \"{query}\". You should only give me arguments that can be inferred fom the text. Give me the arguments in a list split in PRO and CONTRA arguments and please give me the line for each argument from where it can be inferenced."):
            response = data["message"]
        print(f"{response}\n\n\n")"""
    

if __name__ == "__main__":
    main()