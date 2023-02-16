import csv
from _csv import writer

import nltk as nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.snowball import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

leaveStatements = ["bye", "that's all", "nothing",
                   "no", "nope", "goodbye", "nothing else",
                   "that's enough", "that's it"]

bookingStatements = ["i'd like to make a booking", "book a table",
                     "i want to book", "can i book a table", "i'd like to make a reservation",
                     "make a reservation", "can i make a booking", "i want to make a booking"]

nameStatements = ["my name is", "call me", "change my name to", "change name to",
                  "i'm called"]


# preprocessing helper function
# boolean filtering to define whether to filter stopwords

def preProc(query, filtering):
    tokenizer = nltk.RegexpTokenizer(r"\w+")  # only keep words
    stemmer = PorterStemmer()
    english_stopwords = stopwords.words('english')
    tokenized_query = tokenizer.tokenize(query)
    lowered_query = [word.lower() for word in tokenized_query]
    if filtering:
        lowered_query = [word for word in lowered_query
                         if word not in english_stopwords]
    stem_query = [stemmer.stem(word) for word in lowered_query]
    return stem_query


def stemWords(document):
    analyzer = CountVectorizer().build_analyzer()
    stemmer = PorterStemmer()
    return (stemmer.stem(item).lower() for item in analyzer(document))


def findName(query):
    tokenizer = nltk.RegexpTokenizer(r"\w+")  # only keep words
    tokenized_query = tokenizer.tokenize(query)
    unwanted = ["my", "the", "name", "is", "change", "to", "call", "me", "it", "s", "i", "m", "called"]
    name_array = [word.capitalize() for word in tokenized_query if word.lower() not in unwanted]
    newname = " ".join(name_array)  # adds spaces inbetween multiple names e.g. firstname surname
    return newname


def takeBooking():
    print("Sure, I'll make a booking in your name, " + name + ".")
    print("How many people will be coming?")
    people_num = input(": ")
    print("What date would you like to come?")
    date = input(": ")
    print("And finally, what time would you like to book for?")
    time = input(": ")
    print("Please enter your email.")
    email = input(": ")
    with open('bookings.csv', mode='r', encoding='utf-8-sig', errors='ignore') as bookings:
        last_line = bookings.readlines()[-1]
        new_id = int(last_line[0]) + 1
    with open('bookings.csv', 'a+', newline="") as write_booking:
        csv_writer = writer(write_booking)
        csv_writer.writerow([new_id, name, people_num, date, time, email])
    print("Great, we'll get in contact with you soon to let you know the status of your booking.")


def giveResponse(query, file):
    responses = []
    train_set = []
    train_set.append(query)     # need to remove?
    with open(file, mode='r', encoding='utf-8-sig', errors='ignore') as file:
        reader = csv.reader(file)
        for rows in reader:
            train_set.append(rows[0])
            responses.append(rows[1])

    count_vectorizer = CountVectorizer(lowercase=True, analyzer=stemWords)  # no stopwords e.g. "not bad"
    train_set_matrix = count_vectorizer.fit_transform(train_set)
    tfidf_transformer = TfidfTransformer(use_idf=True, sublinear_tf=True)
    tfidf_matrix = tfidf_transformer.fit_transform(train_set_matrix)

    sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)
    response_sim = np.delete(sim, 0)
    best_answer = response_sim.max()
    if best_answer > 0.5:
        location = np.where(response_sim == best_answer)
        response = responses[location[0][0]]
        return response
    else:
        return


# introduction and sentiment analysis

print("Hello, I'm the restaurant bot. How are you today?")
sentiment_response = input(": ")

emotion_data = []
emotion_labels = []
with open("./responses/positiveResponses.txt", mode='r', encoding='utf8', errors='ignore') as f:
    for line in f:
        line = line.replace("\n", "")
        emotion_data.append(line)
        emotion_labels.append("positive")
with open("./responses/negativeResponses.txt", mode='r', encoding='utf8', errors='ignore') as f:
    for line in f:
        line = line.replace("\n", "")
        emotion_data.append(line)
        emotion_labels.append("negative")

vectorizer = CountVectorizer(lowercase=True, analyzer=stemWords)  # no stopwords e.g. "not bad"
sentiment_Xcounts = vectorizer.fit_transform(emotion_data)

tfidf_transformer = TfidfTransformer(use_idf=True, sublinear_tf=True)
emotion_tfidf_X = tfidf_transformer.fit_transform(sentiment_Xcounts)

emotion_classifier = LogisticRegression(random_state=0).fit(emotion_tfidf_X, emotion_labels)

emotion_response_vector = vectorizer.transform([sentiment_response])
emotion_response_vector = tfidf_transformer.transform(emotion_response_vector)

emotion = emotion_classifier.predict(emotion_response_vector)

if emotion == "positive":
    print("That's good to hear.")
else:
    print("I'm sorry to hear that.")

# initial identity management

print("Could I take your name?")
name_response = input(": ")
name = findName(name_response)

# prepare data for intent matching in main conversation loop

intent_data = []
intent_labels = []
with open('questionAnswers.csv', mode='r', encoding='utf-8-sig', errors='ignore') as questionAnswers:
    csv_reader = csv.reader(questionAnswers, delimiter=',')
    for row in csv_reader:
        intent_data.append(row[0])
        intent_labels.append("question")
with open('smallTalkAnswers.csv', mode='r', encoding='utf-8-sig', errors='ignore') as smallTalkAnswers:
    csv_reader = csv.reader(smallTalkAnswers, delimiter=',')
    for row in csv_reader:
        intent_data.append(row[0])
        intent_labels.append("talk")

intent_vectorizer = CountVectorizer(lowercase=True, analyzer=stemWords, stop_words=stopwords.words('english'))
intent_Xcounts = intent_vectorizer.fit_transform(intent_data)
intent_tfidf_X = tfidf_transformer.fit_transform(intent_Xcounts)

intent_classifier = LogisticRegression(random_state=0).fit(intent_tfidf_X, intent_labels)

# main conversation loop

print("Hi " + name + ", what can I do for you?")

finished = False
while not finished:
    response = input(": ")

    if any(statement in response.lower() for statement in nameStatements):
        name = findName(response)
        print("I'll call you " + name + " from now on.")
        print("Anything else I can help you with, " + name + "?")
    elif any(statement in response.lower() for statement in leaveStatements):
        finished = True
        print("Thanks for contacting the restaurant bot. Goodbye!")
    elif any(statement in response.lower() for statement in bookingStatements):
        takeBooking()
        print("Anything else I can help you with, " + name + "?")
    else:
        intent_response_vector = intent_vectorizer.transform([response])
        intent_response_tfidf = tfidf_transformer.transform(intent_response_vector)
        intent = intent_classifier.predict(intent_response_tfidf)
        if intent == "question":
            response = giveResponse(response, 'questionAnswers.csv')
            if response is not None:
                print(response)
            else:
                print("Sorry, I don't know the answer to your question. Please call us or check our website for more information.")
        elif intent == "talk":
            response = giveResponse(response, 'smallTalkAnswers.csv')
            if response is not None:
                print(response)
            else:
                print("Sorry, I can't respond to that.")
        print("Anything else I can help you with, " + name + "?")
