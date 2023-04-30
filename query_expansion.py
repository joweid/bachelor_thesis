from PyDictionary import PyDictionary
from nltk.corpus import wordnet
import nltk
import requests
import json
from revChatGPT.V1 import Chatbot

def rewrite_with_paraphrasation(query):
    tokens = query.split()

    # Find synonyms for each token
    dictionary=PyDictionary()
    new_tokens = []
    for token in tokens:
        synonyms = []
        for syn in wordnet.synsets(token):
            for lemma in syn.lemmas():
                synonyms.append(lemma.name())
        if synonyms:
            new_token = dictionary.meaning(token)
            if new_token:
                new_token = new_token.popitem()[1][0]
            else:
                new_token = synonyms[0]
        else:
            new_token = token
        new_tokens.append(new_token)

    # Create new query with expanded terms
    new_query = " ".join(new_tokens)
    return new_query

def rewrite_with_synonyms_by_thesaurus(query):
    nouns = extract_nouns(query)
    adjectives = extract_adjectives(query)
    noun_dict = {}
    adjective_dict = {}
    new_tokens = []

    for token in nouns:
        noun_dict[token] = create_synonyms_with_thesaurus_api(token, all=False)
    
    for token in adjectives:
        adjective_dict[token] = create_synonyms_with_thesaurus_api(token, all=False)
    
    for word in nltk.word_tokenize(query):
        new_token = word

        if word in noun_dict.keys():
            new_token = noun_dict[word]
        if word in adjective_dict.keys():
            new_token = adjective_dict[word]

        new_tokens.append(new_token)
    
    return " ".join(new_tokens)


def rewrite_with_synonyms_by_wordnet(query):
    nouns = extract_nouns(query)
    adjectives = extract_adjectives(query)
    noun_dict = {}
    adjective_dict = {}
    new_tokens = []

    for token in nouns:
        noun_dict[token] = create_synonyms_with_wordnet(token, all=False)
    
    for token in adjectives:
        adjective_dict[token] = create_synonyms_with_wordnet(token, all=False)
    
    for word in nltk.word_tokenize(query):
        new_token = word

        if word in noun_dict.keys():
            new_token = noun_dict[word]
        if word in adjective_dict.keys():
            new_token = adjective_dict[word]

        new_tokens.append(new_token)
    
    return " ".join(new_tokens)


def expand_with_synonyms(query):
    nouns = extract_nouns(query)
    adjectives = extract_adjectives(query)
    synonyms = []

    for token in nouns:
        synonyms += create_synonyms_with_thesaurus_api(token, all=True)
    for token in adjectives:
        synonyms += create_synonyms_with_thesaurus_api(token, all=True)
    
    return " ".join(synonyms)


def expand_with_chatgpt(query, chatbot):
    for data in chatbot.ask(f"This is a query: {query}. Return 5 similar queries based on this query in a JSON List with the key \"similar_queries\"."):
        response = data["message"]
    output = "{" + response[str(response).index("{") + 1: str(response).rindex("}")] + "}" 
    json_data = json.loads(output)
    return json_data['similar_queries']
    

def create_synonyms_with_wordnet(token, all=False):
    first_synonym = token
    synonyms = []

    syns = []
    for syn in wordnet.synsets(token):
        for lemma in syn.lemmas():
            syns.append(lemma.name())
    
    if not all:
        if syns:
            for syn in syns:
                if not all:
                    if syn.lower() != token.lower():
                        first_synonym = syn
                        break

            return first_synonym
    else:
        if syns:
            return syns
        else:
            return []


def create_synonyms_with_thesaurus_api(token, all=False):
    first_synonym = token
    response = requests.get(f'https://tuna.thesaurus.com/pageData/{token}').json()
    if response['data'] != None:
        dict_synonyms = response['data']['definitionData']['definitions'][0]['synonyms']
        syns = [r["term"] for r in dict_synonyms]
        
        if not all:
            return syns[0]
        else:
            return syns

    else:
        if not all:
            return ""
        else:
            return []


def extract_nouns(sentence):
    return [word for (word, pos) in nltk.pos_tag(nltk.word_tokenize(sentence)) if pos[0] == 'N']


def extract_adjectives(sentence):
    return [word for (word, pos) in nltk.pos_tag(nltk.word_tokenize(sentence)) if pos[0] == 'J']

import openai

openai.api_key = json.load(open("discord_chatgpt_key.json"))['key']
openai.api_base = 'https://api.pawan.krd/v1'
#openai.api_key = 'sk-0PfcSdT723UR44igwVxvEWvLoZJgi0FJyZWy0WCCATp5ka2a'
#openai.api_base = 'https://api.chatanywhere.com.cn/v1'

"""response = openai.Completion.create(
  model="text-davinci-003",
  prompt="Human: This is a text: \"Reason 1 - Teacher tenure creates complacency because teachers know they are unlikely to lose their jobs: If teachers know that they reached the period where they get special defence from most accusations - it would send the message to them that they can then do whatever they want to do in the classroom and really slack with their teaching duties. Reason 2 - Tenure makes it difficult to remove under-performing teachers because the process involves months of legal wrangling by the principal, the school board, the union, and the courts: Most schools stop trying to fire a certain teacher because the proccess is just too difficult. \" A June 1, 2009 study by the New Teacher Project found that 81% of school administrators knew a poorly performing tenured teacher at their school; however, 86% of administrators said they do not always pursue dismissal of teachers because of the costly and time consuming process. It can take up to 335 days to remove a tenured teacher in Michigan before the courts get involved. \" (. http://teachertenure.procon.org...) (Patrick McGuinn, \"Ringing the Bell for K-12 Teacher Tenure Reform,\" www. americanprogress. org). This quote means that 86 OUT OF 100 SCHOOL ADMINISTRATORS WANT A TEACHER TO BE FIRED - but will not do so because the proccess is to draining. But what does that leave our learning and growing generation with? Many teachers who do not care, teach well, or put effort in their work? That is certaintly what this is going to result into if we do not abolish it quickly. Also check out this statistic of who is in favor (people in general) \"An Apr. -May 2011 survey of 2,600 Americans found that 49% oppose teacher tenure while 20% support it. Among teachers, 53% support tenure while 32% oppose it. According to a Sep. 2010 report by the Thomas B. Fordham Institute, 86% of education professors favor \"making it easier to terminate unmotivated or incompetent teachers - even if they are tenured. \u201d Of course you cannot expect most teachers to be against it sinse that it their profession and it effects them - but for bystanders with accurate and unbiased opinions, look how many people are against it. Also, \"56% of school board presidents disagreed with the statement that teacher tenure ensures academic freedom. \" (M. J. Stephey, \"A Brief History of Tenure,\" www. time. com). Reason 3 - Most people are against teature tenure: \"In an Oct. 1, 2006 survey, 91% of school board presidents either agreed or strongly agreed that tenure impedes the dismissal of under-performing teachers. 60% also believed that tenure does not promote fair evaluations. \" (. http://teachertenure.procon.org...) This means that most teachers OF SUCH A LARGE PERCENTAGE are not in favor of the teacher tenure. Reason 4 - Teacher tenure does nothing to promote the education of children: \"Former DC Schools Chancellor Michelle Rhee said in 2008, \"Tenure is the holy grail of teacher unions, but it has no educational value for kids; it only benefits adults. \u201d(\"Rhee-Forming D. C. Schools,\" www. wsj. com). This piece of evidence means that the only people actually benefiting from this tenure are the teachers who are employed - not any students. Isint education suppost to be focused on the younger generation and their best interest? Since when did school become all about the teachers - this tenure undermines what it means to actually be a teacher. If anything, it is only a BAD THING for students - and why would we keep something in our school systems that MAKES THE GENERATIONS' LEARNING LESS VALUEABLE? It does not make any sense. Reason 5 - Tenure at the K-12 level is not earned, but given to nearly everyone: \"To receive tenure at the university level, professors must show contributions to their fields by publishing research. At the K-12 level, teachers only need to \"stick around\u201d for a short period of time to receive tenure. A June 1, 2009 study by the New Teacher Project found that less than 1% of evaluated teachers were rated unsatisfactory. \" (Marcus A. Winters, \"Challenging Tenure in D. C. ,\" www. manhattan-institute. org). This statistic is absolutely upsetting and degrating. Basically, this quote is explaning how 99% of teachers have free protection handed to them if they just stay in that profession for a certain amount of time. What if that teacher was already slacking in many areas? Now we are going to award them for poor effort and teaching abilities? It is not fair to the students involved with these teachers and it is not fair that they do not actually have to WORK to recieve a benefit of protection unlike most other professions that require some form of acomplishment to recieve that/those benefits in question. Because \"with most states granting tenure after three years, teachers have not had the opportunity to \"show their worth, or their ineptitude. \" (Rose Garrett, \"What Is Teacher Tenure? ,\" www. education. com), (. http://teachertenure.procon.org...).Reason 6 - Tenure makes it costly for schools to remove a teacher with poor performance or who is guilty of wrongdoing: \"It costs an average of $250,000 to fire a teacher in New York City. New York spent an estimated $30 million a year paying tenured teachers accused of incompetence and wrongdoing to report to reassignment centers (sometimes called \"rubber rooms\u201d) where they were paid to sit idly. Those rooms were shut down on June 28, 2010. \" (\"Rhee-Forming D. C. Schools,\" www. wsj. com), (Steven Brill, \"The Rubber Room,\" New Yorker). This is just sad, now it even costs the school boards money for teachers not doing their job? Should'nt that be the opposite? Reason 7 - Tenure is not needed to recruit teachers: \"Sacramento Charter High School, which does not offer tenure, had 900 teachers apply for 80 job openings. \" (Nanette Asimov, \"Teacher Job Security Fuels Prop. 74 Battle,\" San Francisco Chronicle). This quote further proves why tenure is pretty much useless and unfair because teachers DO NOT NEED TENURE to continue their job as a teacher at their shchool, past school, future school, or school they are applying for. Reason 8 - With job protections granted through court rulings, collective bargaining, and state and federal laws, teachers today no longer need tenure to protect them from dismissal: \"For this reason, few other professions offer tenure because employees are adequately protected with existing laws. \" (Tenure Reforms and NJSBA Policy: Report of the NJSBA Tenure Task Force,\" New Jersey School Boards Association website, www. njsba. org), (Scott McLeod, JD, PhD, \"Does Teacher Tenure Have a Future? ,\" www. dangerouslyirrelevant. org). This is the most important fact out of all these because it shows how the WHOLE REASON teacher tenure is here in the first place is NOT NEEDED not have the protections that teachers have without tenure. The teacher tenure is not benefitial for anyone except teachers - they get unfair advantages in MANY ways, some I have just listed. Why should we let this continue if unnessisary? Citations: . http://teachertenure.procon.org...http://teachertenure.procon.org...http://teachertenure.procon.org...Wanda Marie Thibodeaux, \"Pro & Cons of Teacher Tenure,\" www. ehow. comPatrick McGuinn, \"Ringing the Bell for K-12 Teacher Tenure Reform,\" www. americanprogress. org. http://teachertenure.procon.org... \"Rhee-Forming D. C. Schools,\" www. wsj. comMarcus A. Winters, \"Challenging Tenure in D. C. ,\" www. manhattan-institute. orgM. J. Stephey, \"A Brief History of Tenure,\" www. time. comRose Garrett, \"What Is Teacher Tenure? ,\" www. education. com. http://teachertenure.procon.org... \"Rhee-Forming D. C. Schools,\" www. wsj. comSteven Brill, \"The Rubber Room,\" New YorkerTenure Reforms and NJSBA Policy: Report of the NJSBA Tenure Task Force,\" New Jersey School Boards Association website, www. njsba. orgScott McLeod, JD, PhD, \"Does Teacher Tenure Have a Future? ,\" www. dangerouslyirrelevant. orgNanette Asimov, \"Teacher Job Security Fuels Prop. 74 Battle,\" San Francisco Chronicle\". Return arguments that you can inference from this text matching to this queries: \"Should teachers get tenure?\". You should only give me arguments that can be inferred fom the text. Give me the arguments in a list split in PRO and CONTRA arguments and please give me the line for each argument from where it can be inferenced.:",
  temperature=0.7,
  max_tokens=256,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0,
  stop=["Human: ", "AI: "]
)

print(response.choices[0].text)"""

"""response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
            {"role": "system", "content": "You are an assistant that retrieves information from texts."},
            {"role": "user", "content": "Who won the world series in 2020?"},
            {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
            {"role": "user", "content": "Where was it played?"}
        ]
)"""



data = json.load(open("outputs.json"))