import json

def annotate():
    con = input("Continue? (True/False): ")
    continue_json = True if con == "True" else False
    annotation_data = json.load(open('annotation.json')) if continue_json else {'annotation': []}
    if continue_json:
        line_stop = len(annotation_data['annotation'])

    queries_json = json.load(open("query_expansion.json"))
    

    print("This script annotates the outputs of our research. While annotating please take a look at the outputs file. For each argument enter a score:\n\
          0 = useless (doesn`t answer the query)\n\
          1 = mediocre (has something to do with the query, but does not give an answer) \n\
          2 = good (suitable argument for the given query)\n\n")
    queries = queries_json['query_expansion']
    for object in queries:
        if continue_json:
            if object['id'] <= line_stop:
                continue

        object_annotation = {'query': object['query'], 'expansion': object['expansion']}

        print(f"Query: {object['query']}")

        #JACCARD
        print("\n\n" + "Jaccard: \n")
        object_annotation['jaccard'] = score()

        #BM25
        print("\n\n" + "BM25 \n")
        object_annotation['bm25'] = score()

        #BERT
        print("\n\n" + "BERT \n")
        object_annotation['bert'] = score()

        #TF-IDF
        print("\n\n" + "TF-IDF \n")
        object_annotation['tfidf'] = score()

        #CHATGPT
        print("\n\n" + "ChatGPT \n")
        object_annotation['chatgpt'] = score()

        annotation_data['annotation'].append(object_annotation)

        with open('annotation.json', 'w') as file:
            file.write(json.dumps(annotation_data, indent=2))

        stop = input("Stop annotating? (True/False): ")
        if stop == "True":
            break
    
    with open('annotation.json', 'w') as file:
        file.write(json.dumps(annotation_data, indent=2))

def score():
    data = {'normal': [], 'expanded': []}
    scores_normal = []
    scores_expanded = []

    try:
        print('Enter the scores (normal): \n')
        for i in range(1, 6):
            score = int(input(f"Argument {i}: "))
            scores_normal.append({'id': i, 'score': score})
        
        print('\nEnter the scores (expanded): \n')
        for j in range(1, 6):
            score = int(input(f"Argument {j}: "))
            scores_expanded.append({'id': j, 'score': score})
        
        data['normal'] = scores_normal
        data['expanded'] = scores_expanded

        return data
    except:
        print("Invalid Input. Try again.")
        return ""


if __name__ == '__main__':
    annotate()