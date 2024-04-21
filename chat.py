import os
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')


intents = [
    {
        'tag': 'greeting',
        'patterns': ['Hi', 'Hello', 'Hey', 'How are you', 'What\'s up'],
        'responses': ['Hi there', 'Hello', 'Hey', 'I\'m fine, thank you', 'Nothing much'],
    },
    {
        'tag': 'goodbye',
        'patterns': ['Bye', 'See you later', 'Goodbye', 'Take care'],
        'responses': ['Goodbye', 'See you later', 'Take care'],
    },
    {
        'tag': 'thanks',
        'patterns': ['Thank you', 'Thanks', 'Thanks a lot', 'I appreciate it'],
        'responses': ['You\'re welcome', 'No problem', 'Glad I could help'],
    },
    {
        'tag': 'help',
        'patterns': ['Help', 'I need help', 'Can you help me', 'What should I do'],
        'responses': ['Sure, what do you need help with?', 'I\'m here to help. What\'s the problem?', 'How can I assist you?'],
    },
    {
        'tag': 'jedi',
        'patterns': ['What is a jedi', 'What is the jedi order', 'What is a jedi master', ],
        'responses': ['A jedi was a devotee to the ways of the Jedi Order, an ancient order of protectors.', 
                      'The Jedi Order was a noble monastic and nontheistic religious order united in their devotion to the light side of the Force.', 
                      'Jedi Master was a rank in the Jedi Order given to wise and powerful Jedi, many of whome were prominent leaders within the Order. Only Masters were allowed to serve on the Jedi High Council',
                      ]
    },
    {
        'tag': 'code',
        'patterns': ['What is the Code of the jedi', 'What is the mantra of the jedi', ],
        'responses': [
                      'The Jedi Code was a set of rules on tenets in the Jedi Order. The code evolved over the course of centuries an applied to all members of the Order. Among the precepts of the Code was a rule forbidding Jedi from training more than one Padawan at any given time. The Code also embodied the philosophical ideals of the Order, such as discipline, self control, a introspection, and was developed to help the Jedi maintain their devotion to the light side of the Force by rejecting the temptations of the dark side.', 
                      'There is no emotion, there is peace. There is no ignorance, there is knowledge. There is no passion, there is serenity. There is no chaos, there is harmony. There is no deat, there is the Force.',
                    
                    ]
    },
    {
        'tag': 'padawans',
        'patterns': ['what is a padawan'],
        'responses': ['A Padawan, also known as a Padawan learner, Padawan apprentice, Jedi apprentice, or apprentice, was the stage of Jedi training where a Force-sensitive of the Jedi Order trained under a Jedi Knight or Master.',
                      ]
    },
    {
        'tag': 'council',
        'patterns': ['What is the High Council', 'Who was on the last council', ],
        'responses': ['The Jedi High Council, simply known as the Jedi Council, was a body of twelve Jedi Masters that governed the Jedi Order. Headquatered in the Jedi Grand Temple on Coruscant, the High Council worked with the Glactic Senate to maintain peace and justice in the Glactic Republic.',
                      'The final Jedi High Council was during the Clone Wars and consisted of: Yaddle (deceased), Plo Koon, Mace Windu, Yoda, Ki-Adi-Mundi. Obi-Wan Kenobi, Saesee Tiin, Eeth Koth, Agen Kolar, Shaak Ti, Kit Fisto, Adi Gallia (deceased), Even Piell (deceased), Oppo Rancisis, Coleman Kcaj, Depa Billaba, Stass Allie, Anakin Skywalker']
    },
    {
        'tag': 'joy',
        'patterns': ['joy', 'happy', 'glad', 'joyful',],
        'responses': [
            "'For God so loved the world that He gave His only begotten Son, that whoever believes in Him should not perish but have everlasting life. For God did not send His Son into the world to condemn the world, but that the world through Him might be saved.' - John 3:16-17",
            "' Then He opened His mouth and taught them, saying: “Blessed are the poor in spirit, For theirs is the kingdom of heaven. Blessed are those who mourn, For they shall be comforted. Blessed are the meek, For they shall inherit the [a]earth. Blessed are those who hunger and thirst for righteousness, For they shall be filled. Blessed are the merciful, For they shall obtain mercy. Blessed are the pure in heart, For they shall see God. Blessed are the peacemakers, For they shall be called sons of God. Blessed are those who are persecuted for righteousness’ sake, For theirs is the kingdom of heaven. Blessed are you when they revile and persecute you, and say all kinds of evil against you falsely for My sake. Rejoice and be exceedingly glad, for great is your reward in heaven, for so they persecuted the prophets who were before you.' - Matthew 5:2-12",
            "'Blessed is the man Who walks not in the counsel of the ungodly, Nor stands in the path of sinners, Nor sits in the seat of the scornful; But his delight is in the law of the Lord, And in His law he meditates day and night.' - Psalms 1:1-2",
            "'What profit has the worker from that in which he labors? I have seen the God-given task with which the sons of men are to be occupied. He has made everything beautiful in its time. Also He has put eternity in their hearts, except that no one can find out the work that God does from beginning to end. I know that nothing is better for them than to rejoice, and to do good in their lives, and also that every man should eat and drink and enjoy the good of all his labor—it is the gift of God.' - Ecclesiastes 3:9-13",
            "'Rejoice in the Lord always. Again I will say, rejoice!' - Philippians 4:4",
            "'My brethren, count it all joy when you fall into various trials, knowing that the testing of your faith produces patience.' - James 1:2-3",
            "'Beloved, do not think it strange concerning the fiery trial which is to try you, as though some strange thing happened to you; but rejoice to the extent that you partake of Christ’s sufferings, that when His glory is revealed, you may also be glad with exceeding joy.' - 1 Peter 4:12–13"
        ]
    },
    {
        'tag': 'sad',
        'patterns': ['sad', 'sorrow', 'depressed',],
        'responses': [
            "'Answer me speedily, O Lord; My spirit fails! Do not hide Your face from me, Lest I be like those who go down into the pit. Cause me to hear Your lovingkindness in the morning, For in You do I trust; Cause me to know the way in which I should walk, For I lift up my soul to You.' - Psalm 143:7-8",
            "'Anxiety in the heart of man causes depression, But a good word makes it glad.' - Proverbs 12:25",
            "'For godly sorrow produces repentance leading to salvation, not to be regretted; but the sorrow of the world produces death.' - 2 Corinthians 7:10",
            "' and those the Lord has rescued will return. They will enter Zion with singing; everlasting joy will crown their heads. Gladness and joy will overtake them, and sorrow and sighing will flee away.' - Isaiah 35:10",
            "'Beloved, do not think it strange concerning the fiery trial which is to try you, as though some strange thing happened to you; but rejoice to the extent that you partake of Christ’s sufferings, that when His glory is revealed, you may also be glad with exceeding joy.' - 1 Peter 4:12–13"
        ]
    },
    {
        'tag': 'guilty',
        'patterns': ['guilty', 'regret', 'remorse',],
        'responses': []
    },
    {
        'tag': 'afraid',
        'patterns': ['afraid', 'scared', 'fear', 'fearful', ],
        'responses': [
            "'Beloved, do not think it strange concerning the fiery trial which is to try you, as though some strange thing happened to you; but rejoice to the extent that you partake of Christ’s sufferings, that when His glory is revealed, you may also be glad with exceeding joy.' - 1 Peter 4:12–13"
        ]
    },

]


vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

tags = []
patterns = []

for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def regex_patterns(input_text):
    re_patterns = {
        'afraid': r'\b(afraid|scared|fear|fearful|)\b',
        'sad': r'\b(sad|sorrow|depressed|depression\sadness|)\b',
        'joy': r'\b(joy|happy|glad|joyful)\b,',
        'guilty': r'\b(guilty|regret|remorse|)\b',
        'code': r'\b(code|mantra|jedi code|)\b',
        'council': r'\b(council|High Council|jedi high council|)\b',
        'padawan': r'\b(padawan|)\b',
        'jedi': r'\b(jedi|Jedi Order|Jedi Master|who are the jedi|)\b',
    }

    for tag, pattern in re_patterns.items():
        if re.search(pattern, input_text, re.IGNORECASE):
            return tag
    return None

def chatbot(input_text):
    matched_tag = regex_patterns(input_text)
    if matched_tag:
        print(matched_tag)
        return get_response_by_tag(matched_tag)
    
    input_text_vectorized = vectorizer.transform([input_text])
    predicted_tag = clf.predict(input_text_vectorized)[0]
    print(predicted_tag)
    return get_response_by_tag(predicted_tag)

def get_response_by_tag(tag):
    for intent in intents:
        if intent['tag'] == tag:
            responses = intent['responses']
            return random.choice(responses)

def get_predetermined_response(input_text, responses):
    predetermined_responses = {
        'greeting': ['Hi there', 'Hello'],
        'goodbye': ['Goodbye', 'See you later'],
        'thanks': ['You\'re welcome', 'No problem'],
        'jedi': [
            'The Jedi Order was a noble monastic and nontheistic religious order united in their devotion to the light side of the Force.',

                 ],
        'code': [
            'The Jedi Code was a set of rules on tenets in the Jedi Order. The code evolved over the course of centuries an applied to all members of the Order. Among the precepts of the Code was a rule forbidding Jedi from training more than one Padawan at any given time. The Code also embodied the philosophical ideals of the Order, such as discipline, self control, a introspection, and was developed to help the Jedi maintain their devotion to the light side of the Force by rejecting the temptations of the dark side.',

        ],
        'padawans': [

        ],
        'council': [
            'The Jedi High Council, simply known as the Jedi Council, was a body of twelve Jedi Masters that governed the Jedi Order. Headquatered in the Jedi Grand Temple on Coruscant, the High Council worked with the Glactic Senate to maintain peace and justice in the Glactic Republic.',

        ],
    }

    for tag, predetermined_responses in predetermined_responses.items():
        if tag == clf.predict(input_text)[0]:
            return random.choice(predetermined_responses)
    return None

counter = 0

def main():
    global counter
    st.title("Chatbot")
    st.write("Welcome to the chatbot. Please type a message and press Enter to start the conversation,")

    counter += 1
    user_input = st.text_input("You: ", key=f"user_input_{counter}")

    if user_input:
        response = chatbot(user_input)
        st.text_area("Chatbot: ", value=response, height=100, max_chars=None, key=f"chatbot_response_{counter}")

        if response.lower() in ['goodbye', 'bye']:
            st.write("Thank you for chatting with me. Have a great day!")
            st.stop()


if __name__ == '__main__':
    main()