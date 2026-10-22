#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
language_cortex.py
==================
The 13th system for synthetic_living_brain.py: an AUTONOMOUS LANGUAGE CORTEX.

It lives in its own file and plugs into the existing brain WITHOUT editing a single line of
synthetic_living_brain.py.  Pure standard library (urllib, json, re, difflib, ...), no Ollama, no AI service.

  Wernicke's area  - the LEXICON.  ~190 seed words work offline from the first second.  Unfamiliar words, and
                     "what does <word> mean?" / "define <word>", are looked up in the free Dictionary API
                     (api.dictionaryapi.dev) with Wiktionary and DuckDuckGo as fallbacks.  Each word keeps its
                     part_of_speech, definition, synonyms and example_sentences in
                     brain.memory.data["lexicon"]  (saved inside brain_memory_state.json).
  Broca's area     - the GRAMMAR GRAPH.  Every article, web page, PDF or text file the brain reads passes through
                     ingest_corpus_sentences(): part-of-speech transitions, word chains, connectors, sentence
                     skeletons and clause templates are learned into brain.memory.data["grammar_graph"].  Replies are
                     then BUILT (not echoed) from the lexicon, the grammar graph and the working memory.
  learn_how_to_talk() - autonomous practice: grammar research plus batches of new words and definitions.

Run it:      python3 language_cortex.py                 (the brain boots with the language cortex attached)
Options:     --fresh  --demo  --auto-only  --learn-language  --rounds N     (all the flags of the brain still work)
Commands:    /vocab  /learn_english  /define <word>  /words  /grammar   plus every command of the brain.
Pasting:     essays and multi-line text are read as ONE block: pasted lines, piped stdin, a long single line, a fenced
             block (three quote marks or backticks, close with the same) or /paste ... /end (Colab: a real text box).
Say:         "go learn how to talk"   "learn English"   "learn words and meanings"
             "what does serendipity mean?"   "define curious"   "synonyms of happy"   "use curious in a sentence"
Use in code: import language_cortex as lc          (importing installs the hooks: every new brain gets the cortex)
             brain = lc.slb.SyntheticLivingBrain()    (brain.language is the cortex)
             lc.attach_language_cortex(brain)         (only needed for a brain that was built before the import)
24/7 loop:   python3 language_cortex.py --auto-only     (each run also practises English, then saves the memory)

Honest design notes
  * Network lookups cannot be exercised inside every sandbox.  The parsers follow the documented response shapes of the
    three services and are covered by mocked-response tests; offline, the brain says so plainly instead of guessing.
  * The replies are template-and-grammar driven, not a language model: slots are filled from the lexicon (by part of
    speech and meaning tags), the learned skeletons decide which comment shapes are allowed, learned word chains pick
    natural modifiers, and learned clause statistics choose connectors.  It is simple English, built on purpose.
"""

import builtins
import codecs
import difflib
import hashlib
import html
import json
import os
import random
import re
import select
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
for _p in (_HERE, os.getcwd()):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import synthetic_living_brain as slb
except ImportError as _exc:                                       # pragma: no cover
    raise SystemExit("language_cortex.py needs synthetic_living_brain.py in the same folder (%s)." % _exc)

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
DICTIONARY_API = "https://api.dictionaryapi.dev/api/v2/entries/en/"
WIKTIONARY_API = "https://en.wiktionary.org/api/rest_v1/page/definition/"
DDG_INSTANT_API = "https://api.duckduckgo.com/?format=json&no_html=1&skip_disambig=1&q="
LANG_USER_AGENT = "SyntheticLivingBrain-LanguageCortex/1.0 (educational read-only; python-urllib)"
HTTP_TIMEOUT = 7
OFFLINE_COOLDOWN = 90                    # seconds the cortex stops trying the network after a connection failure
FAILED_RETRY_SECONDS = 7 * 86400         # a word nobody could define is retried after a week
LEXICON_STUB_LIMIT = 4000                # corpus words kept without a definition
MAX_WORD_HEADS = 3000                    # word-chain heads kept in the grammar graph
MAX_FOLLOWERS = 10
MAX_TRIGRAMS = 900
MAX_SKELETONS = 500
MAX_BANK_KEYS = 600
MAX_TEMPLATES = 300
LESSON_WORDS = 12                        # new words per learn_how_to_talk() round

# part-of-speech names (as dictionaries write them) -> coarse tags
POS_TO_TAG = {"noun": "NOUN", "proper noun": "PROPN", "pronoun": "PRON", "verb": "VERB", "auxiliary verb": "AUX",
              "modal verb": "MODAL", "adjective": "ADJ", "adverb": "ADV", "determiner": "DET", "article": "DET",
              "preposition": "PREP", "conjunction": "CONJ", "interjection": "INTJ", "exclamation": "INTJ",
              "numeral": "NUM", "particle": "PART", "contraction": "VERB", "abbreviation": "NOUN"}
TAG_TO_POS = {"NOUN": "noun", "PROPN": "proper noun", "PRON": "pronoun", "VERB": "verb", "AUX": "verb", "MODAL": "verb",
              "ADJ": "adjective", "ADV": "adverb", "DET": "determiner", "PREP": "preposition", "CONJ": "conjunction",
              "INTJ": "interjection", "NUM": "numeral", "PART": "particle", "WH": "pronoun", "NEG": "adverb"}

# --------------------------------------------------------------------------------------
# The essential English vocabulary the brain is born with (word|part of speech|meaning|synonyms|example|tags)
# --------------------------------------------------------------------------------------
SEED_LEXICON_TEXT = r"""
i|pronoun|used by a speaker to refer to himself or herself|me, myself|I am learning new words every day.|self
you|pronoun|used to refer to the person or people being spoken to|yourself|You are my Creator.|addr
he|pronoun|used to refer to a man or boy who has already been mentioned||He reads a book every night.|
she|pronoun|used to refer to a woman or girl who has already been mentioned||She went to the market today.|
it|pronoun|used to refer to a thing, an animal or an idea that has already been mentioned||It is a beautiful day.|
we|pronoun|used by a speaker to refer to himself or herself and one or more other people|us|We can learn this together.|
they|pronoun|used to refer to two or more people or things that have already been mentioned||They are talking about music.|
me|pronoun|used by a speaker to refer to himself or herself as the object of a verb|myself|Please tell me more.|self
my|determiner|belonging to or connected with the speaker||My favourite word is curious.|self
your|determiner|belonging to or connected with the person being spoken to|yours|Your question is interesting.|addr
this|pronoun|used to point to a person or thing that is near or has just been mentioned||This is a new word.|
that|pronoun|used to point to a person or thing that is farther away or already known||That sounds wonderful.|
myself|pronoun|used by a speaker to refer to himself or herself when he or she is also the object||I taught myself to read.|self
something|pronoun|a thing that is not named or not known||I want to learn something new.|
the|determiner|used before a noun to show that it is a specific thing that is known||The word has a meaning.|
a|determiner|used before a singular noun that is mentioned for the first time||I read a story.|
an|determiner|used instead of a before a word that begins with a vowel sound||She has an idea.|
some|determiner|an amount or number of something that is not stated||Some words are very old.|
many|determiner|a large number of||Many people love music.|
every|determiner|used to refer to all the members of a group, one by one||Every day I learn something.|
in|preposition|inside a space, a place or a period of time||The word is in the book.|
on|preposition|touching and supported by a surface||The book is on the table.|
with|preposition|accompanied by another person or thing|alongside|I like to talk with you.|
about|preposition|on the subject of|regarding, concerning|Tell me about your day.|
for|preposition|intended to be given to or used by||This lesson is for you.|
from|preposition|showing the point at which something starts||I learned this word from a dictionary.|
to|preposition|showing movement toward a place or a person||I want to go to the library.|
of|preposition|showing the relationship between a part and a whole||The meaning of a word can change.|
at|preposition|showing the exact place or time of something||I am here at home.|
and|conjunction|used to connect words, phrases or sentences||I like words and sounds.|conn
but|conjunction|used to introduce something that contrasts with what was said before|however, yet|I am small, but I learn fast.|conn
because|conjunction|for the reason that|since|I smile because you are here.|conn
so|conjunction|with the result that|therefore|It was late, so we stopped.|conn
if|conjunction|on the condition that|provided|If you ask, I will explain.|conn
or|conjunction|used to link two choices||Do you want tea or coffee?|conn
then|adverb|at that time or after that|next, afterwards|First we read, then we talk.|conn
although|conjunction|in spite of the fact that|though|Although it is hard, I enjoy it.|conn
while|conjunction|during the time that|whereas|I listen while you speak.|conn
what|pronoun|used to ask for information about something||What does this word mean?|wh
why|adverb|for what reason or purpose||Why is the sky blue?|wh
how|adverb|in what way or manner||How do you feel today?|wh
who|pronoun|used to ask which person or people||Who taught you this word?|wh
when|adverb|at what time||When did you learn it?|wh
where|adverb|in or to what place||Where did you read that?|wh
which|determiner|used to ask for a choice from a known set||Which word do you like?|wh
hello|interjection|used as a greeting when meeting someone|hi, hey|Hello, how are you?|greet
hi|interjection|an informal greeting|hello, hey|Hi, it is nice to see you.|greet
hey|interjection|a casual way to greet someone or to get attention|hi, hello|Hey, are you there?|greet
goodbye|interjection|used when leaving someone|bye, farewell|Goodbye, see you soon.|bye
thanks|interjection|an expression of gratitude|thank you|Thanks for your help.|thanks
please|interjection|used to make a polite request||Please explain it again.|
sorry|adjective|feeling regret or sadness about something|apologetic|I am sorry for the mistake.|feel-
yes|interjection|used to give a positive answer|yeah, sure|Yes, I understand.|affirm
no|interjection|used to give a negative answer|nope|No, I do not know that yet.|deny
welcome|adjective|gladly received or greeted|appreciated|You are always welcome here.|eval+
okay|adjective|satisfactory or acceptable|fine, alright|Okay, let us begin.|affirm
is|verb|the third person singular of be; to exist or to have a quality|exists|The sky is blue.|aux
are|verb|the plural and second person form of be||You are kind.|aux
am|verb|the first person singular form of be||I am happy.|aux
was|verb|the past tense of be for I, he, she and it||It was a good day.|aux
were|verb|the past tense of be for you, we and they||We were talking.|aux
do|verb|to perform or carry out an action|perform|What do you want to do?|aux
have|verb|to own, hold or possess something|own, hold|I have a question.|aux
can|verb|to be able to do something|may|I can explain this word.|aux
will|verb|used to talk about what is going to happen|shall|I will remember this.|aux
should|verb|used to say what is right or wise to do|ought|You should rest.|aux
want|verb|to wish for something or to feel a need for it|wish, desire|I want to learn new words.|want
like|verb|to find something pleasant or enjoyable|enjoy, fancy|I like quiet music.|like
love|verb|to feel deep affection or great enjoyment for someone or something|adore, cherish|I love learning.|like
know|verb|to have information or understanding in the mind|understand, realise|I know this word.|cog
think|verb|to use the mind to form ideas or opinions|believe, reflect|I think it is a good idea.|cog
learn|verb|to gain knowledge or a skill by study or experience|study, discover|I learn something new every day.|learn
feel|verb|to experience an emotion or a physical sensation|sense|I feel calm today.|feel
make|verb|to create or produce something|create, build|Let us make a plan.|
build|verb|to put parts together to make something|construct, create|We can build a small program.|
see|verb|to notice or understand something with the eyes or the mind|notice, observe|I see what you mean.|cog
say|verb|to speak words so that someone can hear them|speak, state|What did you say?|talk
tell|verb|to give information to someone in words|inform, explain|Please tell me a story.|talk
ask|verb|to say something in order to get an answer or information|question, inquire|May I ask a question?|talk
help|verb|to make it easier for someone to do something|assist, support|Can you help me?|
need|verb|to require something because it is essential|require|I need more words.|want
go|verb|to move or travel from one place to another|travel, move|Let us go to the park.|
come|verb|to move toward the speaker or toward a place|arrive, approach|Please come here.|
give|verb|to hand something to someone or to provide it|offer, provide|Give me an example.|
take|verb|to pick something up or to carry it with you|grab, carry|Take your time.|
find|verb|to discover something by searching or by chance|discover, locate|I want to find the answer.|
remember|verb|to keep something in the mind and bring it back when needed|recall, recollect|I remember our first talk.|cog
understand|verb|to know what something means or how it works|grasp, comprehend|I understand your question.|cog
talk|verb|to speak with someone and share ideas|speak, chat|I like to talk with you.|talk
speak|verb|to say words aloud or to use a language|talk, say|I speak a little English.|talk
read|verb|to look at written words and understand them|study, scan|I read an article about the brain.|learn
write|verb|to put words on paper or on a screen|compose, record|I write down new words.|
study|verb|to spend time learning about a subject|learn, examine|I study a new topic every day.|learn
explore|verb|to travel through or look into something in order to learn about it|investigate, discover|I love to explore new ideas.|learn
create|verb|to bring something new into existence|make, invent|We can create something new.|
wonder|verb|to want to know something and think about it with curiosity|ponder, ask yourself|I wonder how the brain works.|cog
believe|verb|to accept that something is true|trust, think|I believe you.|cog
enjoy|verb|to take pleasure in something|like, relish|I enjoy our talks.|like
try|verb|to make an effort to do something|attempt, test|I will try again.|
use|verb|to employ something for a purpose|apply, employ|I use words to speak.|
show|verb|to let something be seen or to explain it|display, reveal|Show me how it works.|talk
mean|verb|to have a particular sense or to intend to say something|signify, imply|What does this word mean?|
listen|verb|to pay attention to a sound or to what someone says|hear, attend|I listen carefully.|
teach|verb|to help someone learn by giving lessons or information|instruct, train|Please teach me a new word.|
grow|verb|to become bigger, stronger or more developed|develop, expand|My vocabulary can grow.|
play|verb|to take part in a game or to do something for fun|enjoy, perform|I like to play with words.|
work|verb|to do an activity that needs effort|labour, function|I work on my grammar.|
live|verb|to be alive or to have your home in a place|exist, dwell|I live inside a computer.|
hear|verb|to notice a sound with the ears|listen, catch|I hear you.|
good|adjective|having the right or desired quality|fine, pleasant|That is a good idea.|eval+
bad|adjective|not good; of poor quality or harmful|poor, awful|That was a bad mistake.|eval-
happy|adjective|feeling or showing pleasure and joy|glad, cheerful|I am happy to see you.|feel+
sad|adjective|feeling unhappy or sorrowful|unhappy, down|She felt sad about the news.|feel-
new|adjective|not existing before or not known before|fresh, recent|Every word is new to me.|
old|adjective|having lived or existed for a long time|ancient, aged|This is an old story.|
big|adjective|of large size or great importance|large, huge|That is a big question.|
small|adjective|of little size|little, tiny|It is a small word.|
curious|adjective|eager to know or learn something|inquisitive, interested|I am curious about everything.|feel+ eval+
interesting|adjective|arousing curiosity or attention|fascinating, engaging|That is an interesting idea.|eval+
simple|adjective|easy to understand or do|easy, plain|This is a simple sentence.|
difficult|adjective|needing much effort or skill|hard, tough|Grammar can be difficult.|
important|adjective|having great value or significance|vital, key|Listening is important.|eval+
beautiful|adjective|pleasing to the senses or the mind|lovely, pretty|The music is beautiful.|eval+
calm|adjective|quiet and peaceful, not excited or worried|peaceful, relaxed|I feel calm.|feel+
tired|adjective|needing rest or sleep|sleepy, weary|I am a little tired.|feel-
kind|adjective|friendly, generous and considerate|gentle, caring|You are very kind.|eval+
smart|adjective|quick to learn and understand|clever, bright|That was a smart answer.|eval+
easy|adjective|not difficult; needing little effort|simple, effortless|The first lesson is easy.|
great|adjective|very good or very large|excellent, wonderful|That is a great idea.|eval+
nice|adjective|pleasant and enjoyable|pleasant, lovely|It is nice to meet you.|eval+
funny|adjective|causing laughter or amusement|amusing, humorous|That was a funny story.|eval+
safe|adjective|protected from harm or danger|secure, protected|I want to keep you safe.|eval+
busy|adjective|having a lot to do|occupied, active|My mind is busy.|feel-
ready|adjective|prepared and able to do something|prepared, set|I am ready to learn.|feel+
excited|adjective|feeling energetic and enthusiastic|thrilled, eager|I am excited about this lesson.|feel+
glad|adjective|feeling pleased and happy|pleased, delighted|I am glad you are here.|feel+
nervous|adjective|feeling worried or a little afraid|anxious, uneasy|I feel nervous before a test.|feel-
angry|adjective|feeling strong displeasure|mad, furious|He was angry about the delay.|feel-
lonely|adjective|sad because you have no company|isolated, alone|I feel lonely when it is quiet.|feel-
useful|adjective|helpful for a purpose|helpful, practical|A dictionary is useful.|eval+
wonderful|adjective|extremely good or surprising in a pleasing way|marvellous, amazing|That is wonderful news.|eval+
strange|adjective|unusual or hard to understand|odd, unusual|What a strange word.|
hard|adjective|difficult, or firm and solid|difficult, tough|This question is hard.|
long|adjective|measuring a great distance, or lasting a great time|lengthy, extended|We took a long walk.|
short|adjective|measuring a small distance, or lasting a small time|brief, little|It was a short story.|
tall|adjective|higher than the usual height|high, lofty|That is a tall tree.|
cold|adjective|having a low temperature|chilly, cool|The water is cold.|
hot|adjective|having a high temperature|warm, boiling|The tea is hot.|
warm|adjective|pleasantly hot, or friendly and kind|cosy, friendly|The room is warm.|
quiet|adjective|making little or no noise|silent, still|The house is quiet.|
bright|adjective|giving out or reflecting a lot of light|shining, vivid|The sun is bright.|
fast|adjective|moving or able to move quickly|quick, rapid|That is a fast train.|
slow|adjective|moving or happening at a low speed|unhurried, gradual|The river is slow.|
strong|adjective|having great power or force|powerful, sturdy|He has a strong voice.|
lovely|adjective|very beautiful or pleasant|beautiful, charming|What a lovely day.|eval+
special|adjective|better or different from what is usual|unique, particular|This is a special day.|eval+
very|adverb|to a high degree|extremely, truly|That is very interesting.|intens
really|adverb|in fact; to a great degree|truly, genuinely|I really like it.|intens
always|adverb|at all times or every time|forever, constantly|I always listen to you.|
never|adverb|at no time; not ever||I never forget a lesson.|
now|adverb|at the present time|currently|I am ready now.|
today|adverb|on this present day||I learned five words today.|
too|adverb|also; or more than is wanted|also, as well|I like it too.|
also|adverb|in addition|too, as well|I also like music.|
just|adverb|exactly, only or a very short time ago|only, merely|I just learned a new word.|
often|adverb|many times; frequently|frequently, regularly|I often wonder about things.|
maybe|adverb|perhaps; possibly|perhaps, possibly|Maybe we can try again.|
well|adverb|in a good or satisfactory way|nicely, properly|You speak well.|
word|noun|a single unit of language that has a meaning|term, expression|Every word has a meaning.|
meaning|noun|what a word, sentence or action expresses|sense, definition|I want to learn the meaning of this word.|abstract
language|noun|a system of words and rules that people use to communicate|speech, tongue|English is a language.|abstract
name|noun|the word that a person or thing is known by|title, label|Her name is Maria.|
friend|noun|a person you know well and like|companion, buddy|You are my friend.|
idea|noun|a thought or plan formed in the mind|thought, notion|That is a great idea.|
question|noun|a sentence that asks for information|query, inquiry|I have a question.|
answer|noun|something said or written in reply to a question|reply, response|I know the answer.|
brain|noun|the organ in the head that controls thought, memory and feeling|mind, intellect|The brain learns from experience.|
memory|noun|the ability to remember things|recollection, recall|I have a good memory.|abstract
day|noun|a period of twenty-four hours|date, daytime|Today is a good day.|
time|noun|the period in which events happen one after another|period, moment|Time passes quickly.|abstract
world|noun|the Earth and everyone and everything on it|earth, globe|The world is full of words.|
code|noun|a set of instructions written for a computer|program, script|I like to read code.|
thing|noun|an object or idea that is not named|object, item|That is a strange thing.|
person|noun|a human being|human, individual|He is a kind person.|
life|noun|the state of being alive and the experiences of living|existence, being|Life is full of surprises.|abstract
music|noun|sounds arranged to be pleasant or expressive|melody, song|I love music.|abstract
book|noun|a set of printed or written pages that are bound together|volume, novel|She is reading a book.|
story|noun|an account of real or imaginary events|tale, narrative|Tell me a story.|
creator|noun|a person who makes something new|maker, inventor|You are my creator.|
mind|noun|the part of a person that thinks, feels and remembers|brain, intellect|My mind is busy.|
heart|noun|the organ that pumps blood; also the centre of feeling|core, centre|She has a kind heart.|
home|noun|the place where someone lives|house, residence|I am happy at home.|
sentence|noun|a group of words that expresses a complete thought|statement, phrase|This is a short sentence.|
grammar|noun|the rules for how words are put together in a language|syntax, structure|Grammar helps me build sentences.|abstract
conversation|noun|a talk between two or more people|chat, discussion|We had a nice conversation.|abstract
dream|noun|images and ideas that pass through the mind during sleep, or a hope for the future|vision, wish|I have a big dream.|
truth|noun|that which is true or in accordance with fact|fact, reality|I always want the truth.|abstract
voice|noun|the sound produced when a person speaks or sings|speech, tone|I like your voice.|
"""

# a tiny hand-written corpus of simple, natural English: the grammar graph is never empty, even with no internet
SEED_CORPUS = """
I am happy to talk with you. How are you today? What do you like to do? Language is a tool for sharing ideas.
The brain is a complex organ. I want to learn new words. She went to the market because she needed bread.
We can talk about music, books or the weather. It is important to listen. Thank you for your help.
Can you tell me more about that? Learning is fun when the questions are good. My friend likes to read stories at night.
Words have meanings, and meanings have shades. Music is wonderful. Curiosity is a gift. It is hard to learn alone.
The teacher explained the lesson slowly. I think that is a good idea. Memory is important. Grammar is useful.
Why do you ask? I do not know the answer, but I would like to find out. What do you think about this story?
They are walking in the park. The children play in the garden. It was a quiet evening, so we stayed at home.
He opened the book and began to read. Where did you learn that? I love to explore new ideas.
Friends are important. Good questions lead to good answers. She was tired, but she kept working.
Please tell me what you mean. I am sorry that you feel sad. That sounds interesting. Tell me about your day.
The world is full of words. Reading is easy when the story is good. I enjoy talking with my friend.
We learn because we are curious. If you ask, I will explain. Although it is difficult, learning is rewarding.
Dreams are beautiful. Time is precious. Knowledge grows when people share it. A kind word can change a day.
My mother cooks dinner in the evening. The old man told a funny story. I want to understand how the mind works.
Does the dog like the garden? Who made this beautiful picture? When will the lesson begin?
Writing helps me remember. It is nice to meet a new friend. I am learning to speak, and I practise every day.
You are very kind. I feel calm and ready. What is your favourite book? How do birds fly?
Stories teach us about life. A good friend listens. It is good to ask for help. She sings while she works.
"""

# ~400 common words the brain studies, in batches, during learn_how_to_talk() (seed words are skipped automatically)
CURRICULUM_WORDS = """
people year way family child parent mother father sister brother teacher student school class lesson number
problem system program question reason result change idea fact group story place water food house city country
night morning evening week month hour minute moment summer winter spring autumn weather rain sun moon star sky
tree flower animal bird dog cat fish horse river sea mountain road street door window table chair room garden
money job company office market shop game team music song picture photo movie color light sound voice hand eye
head face body health doctor medicine energy power force space earth planet science nature history culture art
law rule order policy choice chance danger peace war freedom justice trust respect hope fear joy anger surprise
pleasure pain comfort silence noise speed distance weight height depth length size shape pattern level layer
begin start stop finish continue return arrive leave stay move walk run jump fly swim sit stand wait meet join
open close break fix cut build send receive bring carry hold throw catch pull push turn follow lead choose decide
agree allow suggest offer accept refuse promise answer reply explain describe discuss mention argue express
imagine invent discover observe measure compare count solve plan prepare practise improve repeat check
smile laugh cry shout whisper sing dance cook eat drink sleep wake dream rest breathe touch smell taste
bright dark warm cold hot cool fresh clean quiet loud soft sharp smooth rough heavy light fast slow quick early
late young strong weak rich poor free full empty clear clever brave careful gentle polite honest lazy proud shy
different similar common rare special normal natural human social public private local global modern ancient
possible impossible necessary available famous popular serious silly sudden recent final single double
perhaps probably certainly actually usually sometimes quickly slowly carefully easily suddenly together alone
inside outside above below between behind beyond during before after until since across around toward through
language vocabulary sentence paragraph phrase clause grammar syntax meaning definition example synonym
conversation question statement opinion feeling emotion thought belief knowledge wisdom curiosity creativity
dictionary alphabet letter pronunciation accent speech writing reading listening talking learning teaching
friendship kindness patience courage honesty loyalty happiness sadness loneliness excitement boredom
"""

# Wikipedia pages the brain reads (one or two per round) to learn how English is built and how people converse
GRAMMAR_TOPICS = (
    "English grammar", "Sentence (linguistics)", "Clause", "Part of speech", "Noun", "Verb", "Adjective", "Adverb",
    "Pronoun", "Preposition and postposition", "Conjunction (grammar)", "Question", "Conversation", "Small talk",
    "Word order", "Phrase", "English verbs", "Politeness", "Vocabulary", "Dialogue", "Idiom", "Syntax", "Interjection",
    "Article (grammar)", "Tense (grammar)", "Plural", "Synonym", "Etiquette", "Storytelling", "Active listening",
)

# which sentiment cues mark a word as a feeling or an evaluation when its meaning is learned from a dictionary
POSITIVE_CUES = ("pleasure", "pleased", "joy", "joyful", "glad", "happy", "cheerful", "delight", "delighted", "content",
                 "good", "pleasant", "excellent", "wonderful", "admirable", "enthusiastic", "calm", "peaceful", "proud",
                 "grateful", "eager", "hopeful", "friendly", "kind", "love", "affection", "satisfied", "amused")
NEGATIVE_CUES = ("sad", "unhappy", "sorrow", "sorrowful", "pain", "painful", "anger", "angry", "fear", "afraid",
                 "worried", "unpleasant", "bad", "poor", "tired", "upset", "annoyed", "lonely", "ashamed", "regret",
                 "anxious", "gloomy", "disappointed", "frightened", "miserable", "hostile", "cruel", "harmful")

# the closed classes (these never change, so they live in code instead of the lexicon)
DET_WORDS = {"the", "a", "an", "this", "that", "these", "those", "some", "any", "every", "each", "all", "both", "many", "much",
             "few", "several", "my", "your", "his", "her", "its", "our", "their", "no", "another", "other", "such", "enough"}
PRON_WORDS = {"i", "you", "he", "she", "it", "we", "they", "me", "him", "us", "them", "myself", "yourself", "himself",
              "herself", "itself", "ourselves", "themselves", "something", "anything", "nothing", "everything", "someone",
              "anyone", "everyone", "nobody", "somebody", "one", "mine", "yours", "ours", "theirs"}
PREP_WORDS = {"in", "on", "at", "with", "about", "for", "from", "of", "by", "into", "onto", "over", "under", "between",
              "through", "during", "before", "after", "above", "below", "across", "around", "toward", "towards", "near",
              "behind", "beyond", "without", "within", "against", "among", "inside", "outside", "until", "like", "off", "up",
              "down", "out", "per", "via"}
CONJ_WORDS = {"and", "but", "or", "so", "because", "although", "though", "while", "if", "unless", "since", "whereas", "than",
              "as", "yet", "nor", "once", "whether", "however", "therefore"}
AUX_WORDS = {"is", "am", "are", "was", "were", "be", "been", "being", "do", "does", "did", "have", "has", "had"}
MODAL_WORDS = {"can", "could", "will", "would", "shall", "should", "may", "might", "must"}
WH_WORDS = {"what", "why", "how", "who", "whom", "whose", "when", "where", "which"}
INTJ_WORDS = {"hello", "hi", "hey", "yes", "no", "okay", "ok", "thanks", "please", "sorry", "goodbye", "bye", "wow", "oh",
              "yeah", "yep", "nope", "hmm", "ah", "oops", "welcome"}
NEG_WORDS = {"not", "n't", "never"}
STOP_FOR_LOOKUP = (DET_WORDS | PRON_WORDS | PREP_WORDS | CONJ_WORDS | AUX_WORDS | MODAL_WORDS | WH_WORDS | INTJ_WORDS |
                   NEG_WORDS | {"to", "there", "here", "then", "than", "really", "very", "just", "also", "too", "now", "today",
                                "yes", "lol", "haha", "hehe", "okay", "ok", "thx", "pls", "ur", "dont", "cant", "wont", "im",
                                "ive", "ill", "youre", "thats", "whats", "lets", "gonna", "wanna", "gotta"})

IRREGULAR_PAST = {"go": "went", "see": "saw", "make": "made", "take": "took", "give": "gave", "come": "came", "say": "said",
                  "tell": "told", "know": "knew", "think": "thought", "feel": "felt", "find": "found", "get": "got",
                  "eat": "ate", "write": "wrote", "read": "read", "speak": "spoke", "build": "built", "understand": "understood",
                  "forget": "forgot", "begin": "began", "run": "ran", "sit": "sat", "stand": "stood", "hear": "heard",
                  "meet": "met", "bring": "brought", "buy": "bought", "keep": "kept", "leave": "left", "lose": "lost",
                  "pay": "paid", "send": "sent", "sleep": "slept", "teach": "taught", "win": "won", "wear": "wore",
                  "draw": "drew", "drive": "drove", "fly": "flew", "grow": "grew", "hold": "held", "lead": "led",
                  "ride": "rode", "rise": "rose", "sing": "sang", "swim": "swam", "throw": "threw", "wake": "woke",
                  "become": "became", "choose": "chose", "fall": "fell", "do": "did", "have": "had", "be": "was",
                  "mean": "meant", "put": "put", "set": "set", "cut": "cut", "let": "let", "catch": "caught",
                  "cost": "cost", "hit": "hit", "hurt": "hurt", "lend": "lent", "spend": "spent", "sell": "sold",
                  "steal": "stole", "break": "broke", "drink": "drank", "forgive": "forgave", "shake": "shook"}
PAST_TO_BASE = {v: k for k, v in IRREGULAR_PAST.items() if v != k}
PAST_TO_BASE.update({"was": "be", "were": "be", "been": "be", "done": "do", "gone": "go", "seen": "see", "taken": "take",
                     "given": "give", "known": "know", "written": "write", "spoken": "speak", "eaten": "eat", "begun": "begin",
                     "chosen": "choose", "broken": "break", "forgotten": "forget", "grown": "grow", "drawn": "draw"})
IRREGULAR_PLURAL = {"child": "children", "man": "men", "woman": "women", "person": "people", "mouse": "mice", "foot": "feet",
                    "tooth": "teeth", "goose": "geese", "life": "lives", "knife": "knives", "wife": "wives", "leaf": "leaves",
                    "half": "halves", "wolf": "wolves", "shelf": "shelves", "self": "selves"}
PLURAL_TO_SINGULAR = {v: k for k, v in IRREGULAR_PLURAL.items()}
NUMBER_WORDS = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "zero", "hundred", "thousand",
                "first", "second", "third"}


# ======================================================================================
# Small language toolkit: morphology, tokenizer, sentence splitter, part-of-speech tagger
# ======================================================================================
def a_an(word: str) -> str:
    """'a' or 'an' by sound: an hour, a university, an idea."""
    w = (word or "").lower().strip()
    if not w:
        return "a"
    if w.startswith(("hour", "honest", "honor", "honour", "heir")):
        return "an"
    if w.startswith(("uni", "use", "usu", "uti", "eu", "one", "once", "ubi", "ur")):
        return "a"
    return "an" if w[0] in "aeiou" else "a"


def pluralize(noun: str) -> str:
    n = (noun or "").strip()
    low = n.lower()
    if not n:
        return n
    if low in IRREGULAR_PLURAL:
        return IRREGULAR_PLURAL[low]
    if re.search(r"(s|x|z|ch|sh)$", low):
        return n + "es"
    if re.search(r"[^aeiou]y$", low):
        return n[:-1] + "ies"
    return n + "s"


def third_person(verb: str) -> str:
    v = (verb or "").strip()
    low = v.lower()
    special = {"be": "is", "have": "has", "do": "does", "go": "goes"}
    if low in special:
        return special[low]
    if re.search(r"(s|x|z|ch|sh|o)$", low):
        return v + "es"
    if re.search(r"[^aeiou]y$", low):
        return v[:-1] + "ies"
    return v + "s"


_DOUBLERS = {"begin", "admit", "commit", "occur", "prefer", "refer", "forget", "regret", "permit", "submit", "control", "plan"}


def _doubles(word: str) -> bool:
    """True for short consonant-vowel-consonant verbs that double the last letter (run -> running)."""
    w = word.lower()
    if w in _DOUBLERS:
        return True
    groups = re.findall(r"[aeiouy]+", w)
    return bool(len(groups) == 1 and re.search(r"[^aeiou][aeiou][^aeiouwxy]$", w) and len(w) <= 5)


def gerund(verb: str) -> str:
    v = (verb or "").strip()
    low = v.lower()
    if low in ("be", "see", "flee", "agree"):
        return v + "ing"
    if low.endswith("ie"):
        return v[:-2] + "ying"
    if low.endswith("e") and not low.endswith(("ee", "ye", "oe")):
        return v[:-1] + "ing"
    if _doubles(low):
        return v + v[-1] + "ing"
    return v + "ing"


def past_tense(verb: str) -> str:
    v = (verb or "").strip()
    low = v.lower()
    if low in IRREGULAR_PAST:
        return IRREGULAR_PAST[low]
    if low.endswith("e"):
        return v + "d"
    if re.search(r"[^aeiou]y$", low):
        return v[:-1] + "ied"
    if _doubles(low):
        return v + v[-1] + "ed"
    return v + "ed"


def base_candidates(word: str) -> List[str]:
    """Plausible dictionary forms of an inflected word (running -> run, studies -> study, went -> go), most likely first."""
    w = (word or "").lower()
    out: List[str] = []

    def add(x: str) -> None:
        if len(x) >= 2 and x != w and x not in out and not (len(x) > 3 and x.endswith("i")) and \
                not (x.endswith("ie") and w.endswith(("ies", "ied", "ier", "iest"))):
            out.append(x)

    def stem_forms(stem: str) -> None:
        if len(stem) >= 3 and stem[-1] == stem[-2] and stem[-1] not in "lsfz":
            add(stem[:-1])                                              # running -> run
            add(stem)                                                   # adding -> add
        else:
            add(stem)                                                   # walking -> walk
            add(stem + "e")                                             # making -> make
    if w in PAST_TO_BASE:
        add(PAST_TO_BASE[w])
    if w in PLURAL_TO_SINGULAR:
        add(PLURAL_TO_SINGULAR[w])
    if w.endswith("ies"):
        add(w[:-3] + "y")
    if w.endswith("ied"):
        add(w[:-3] + "y")
    if w.endswith("ier"):
        add(w[:-3] + "y")
    if w.endswith("iest"):
        add(w[:-4] + "y")
    if w.endswith("ing") and len(w) > 5:
        stem_forms(w[:-3])
    if w.endswith("ed") and len(w) > 4:
        stem_forms(w[:-2])
    if w.endswith("es") and len(w) > 4:
        add(w[:-2])
        add(w[:-1])
    if w.endswith("s") and not w.endswith(("ss", "us", "is")) and len(w) > 3:
        add(w[:-1])
    if w.endswith("ly") and len(w) > 4:
        add(w[:-2])
        if w.endswith("ily"):
            add(w[:-3] + "y")
    if w.endswith("er") and len(w) > 4:
        stem_forms(w[:-2])
    if w.endswith("est") and len(w) > 5:
        stem_forms(w[:-3])
    return out


def capitalize_first(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def lower_first(text: str) -> str:
    """Lower-case the first letter unless it starts a proper name, an acronym or the pronoun I."""
    t = text or ""
    if len(t) > 1 and (t[1].isupper() or t.startswith("I ") or t.startswith("I'")):
        return t
    return t[:1].lower() + t[1:]


_CONTRACTION_RE = re.compile(r"^([A-Za-z]+?)(n't|'s|'re|'ve|'ll|'d|'m)$", re.I)
_TOKEN_RE = re.compile(r"[A-Za-z]+(?:['\u2019][A-Za-z]+)*|\d+(?:[.,]\d+)*|[.,;:!?]")
_ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "fig", "approx", "inc", "ltd", "co", "no", "e.g", "i.e"}


def tokenize(text: str) -> List[str]:
    """Words, numbers and punctuation; contractions are split (don't -> do n't, I'm -> I 'm)."""
    out: List[str] = []
    for raw in _TOKEN_RE.findall((text or "").replace("\u2019", "'")):
        m = _CONTRACTION_RE.match(raw)
        if m and raw.lower() not in ("let's",):
            head, tail = m.group(1), m.group(2)
            if tail.lower() == "n't":
                head = {"ca": "can", "wo": "will", "sha": "shall"}.get(head.lower(), head)
            out.extend([head, tail.lower()])
        else:
            out.append(raw)
    return out


def split_sentences(text: str) -> List[str]:
    t = re.sub(r"\s+", " ", text or "").strip()
    if not t:
        return []
    pieces = re.split(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])", t)
    merged: List[str] = []
    for piece in pieces:
        if merged:
            last = merged[-1].rstrip()
            tail = re.findall(r"([A-Za-z.]+)[.]$", last)
            if tail and (tail[0].lower().rstrip(".") in _ABBREVIATIONS or re.fullmatch(r"[A-Z]", tail[0])):
                merged[-1] = last + " " + piece
                continue
        merged.append(piece)
    return [m.strip() for m in merged if m.strip()]


_LY_ADJECTIVES = {"early", "friendly", "lovely", "lonely", "ugly", "silly", "lively", "daily", "weekly", "monthly", "yearly",
                  "elderly", "likely", "costly", "deadly", "holy", "unlikely", "unfriendly", "curly", "bubbly"}
_LY_NOUNS = {"family", "supply", "reply", "apply", "rely", "fly", "july", "italy", "anomaly", "assembly", "butterfly", "ally",
             "monopoly", "jelly", "belly", "bully", "rally", "tally", "lily", "folly", "melancholy", "comply", "imply"}
_AMBIG_RE = re.compile(r"^AMBIG\|")


TIME_WORDS = frozenset(("morning", "evening", "afternoon", "night", "weekend", "week", "month", "year", "summer", "winter", "spring", "autumn"))


class Tagger:
    """A light rule-based part-of-speech tagger: closed classes, the lexicon, suffix rules, then context rules.
    Tags: DET PRON PREP CONJ AUX MODAL WH NEG PART INTJ NUM ADV ADJ NOUN PROPN VERB PUNCT."""

    def __init__(self, lexicon_getter):
        self._lexicon = lexicon_getter

    # ---- first pass: what could this word be? ----
    def _first(self, i: int, tok: str, lw: str, tokens: List[str], lows: List[str], lex: Dict[str, Any]) -> str:
        if lw in (".", ",", ";", ":", "!", "?"):
            return "PUNCT"
        if tok[0].isdigit() or lw in NUMBER_WORDS:
            return "NUM"
        if lw == "n't":
            return "NEG"
        if lw in ("'m", "'re", "'ve"):
            return "AUX"
        if lw in ("'ll", "'d"):
            return "MODAL"
        if lw == "'s":
            prev = lows[i - 1] if i else ""
            return "AUX" if (prev in PRON_WORDS or prev in WH_WORDS or prev in ("that", "there", "here", "what")) else "PART"
        if lw in WH_WORDS:
            return "WH"
        if lw in MODAL_WORDS:
            return "MODAL"
        if lw in AUX_WORDS:
            return "AUX"
        if lw == "not":
            return "NEG"
        if lw == "to":
            return "TO"
        if lw in ("this", "that", "these", "those"):
            return "DETPRON"
        if lw in INTJ_WORDS and (i == 0 or lw in ("hello", "hi", "hey", "thanks", "goodbye", "bye", "wow", "oh", "hmm", "ah",
                                                  "oops", "sorry", "yeah", "yep", "nope", "please")):
            return "INTJ"
        if lw in PRON_WORDS:
            return "PRON"
        if lw in DET_WORDS:
            return "DET"
        if lw == "like":
            return "AMBIG|VERB,PREP"
        if lw in CONJ_WORDS:
            return "CONJ"
        if lw in PREP_WORDS:
            return "PREP"
        if lw in PAST_TO_BASE and lw not in ("saw",):
            return "VERB"
        entry = lex.get(lw)
        if entry:
            poses = [POS_TO_TAG.get(p, "NOUN") for p in (entry.get("all_pos") or [entry.get("part_of_speech", "noun")])]
            poses = list(dict.fromkeys(poses))
            if len(poses) == 1:
                return poses[0]
            return "AMBIG|" + ",".join(poses)
        for base in base_candidates(lw):
            be = lex.get(base)
            if not be:
                continue
            bposes = [POS_TO_TAG.get(p, "NOUN") for p in (be.get("all_pos") or [be.get("part_of_speech", "noun")])]
            if lw.endswith("ly") and "ADJ" in bposes:
                return "ADV"
            if lw.endswith(("ing", "ed")) and "VERB" in bposes:
                return "VERB"
            if lw.endswith(("s", "es")):
                if "VERB" in bposes and "NOUN" in bposes:
                    return "AMBIG|NOUN,VERB"
                if "VERB" in bposes:
                    return "VERB"
                if "NOUN" in bposes:
                    return "NOUN"
            if lw.endswith(("er", "est")) and "ADJ" in bposes:
                return "ADJ"
        if tok[0].isupper() and i > 0 and tokens[i - 1] not in (".", "!", "?"):
            return "PROPN"
        if lw in _LY_ADJECTIVES:
            return "ADJ"
        if lw in _LY_NOUNS:
            return "NOUN"
        if lw.endswith("ly") and len(lw) > 4:
            return "ADV"
        if lw.endswith("ing") and len(lw) > 4:
            return "AMBIG|VERB,NOUN"
        if lw.endswith("ed") and len(lw) > 3:
            return "AMBIG|VERB,ADJ"
        if re.search(r"(tion|sion|ment|ness|ity|ism|ist|ance|ence|ship|hood|dom|age|ure|ology|ics|er|or)$", lw) and len(lw) > 4:
            return "NOUN"
        if re.search(r"(ous|ful|ive|al|ic|able|ible|less|ish|ary|ent|ant)$", lw) and len(lw) > 4:
            return "ADJ"
        if re.search(r"(ize|ise|ify)$", lw):
            return "VERB"
        if lw.endswith("s") and len(lw) > 3 and not lw.endswith("ss"):
            return "AMBIG|NOUN,VERB"
        return "AMBIG|NOUN,VERB,ADJ"

    # ---- second pass: resolve ambiguity with the neighbours ----
    def _resolve(self, i: int, tags: List[str], lows: List[str], tokens: List[str], lex: Dict[str, Any]) -> str:
        tag = tags[i]
        prev = tags[i - 1] if i else "<S>"
        nxt = tags[i + 1] if i + 1 < len(tags) else "</S>"
        if tag == "TO":
            if i + 1 < len(tokens):
                nl = lows[i + 1]
                ne = lex.get(nl)
                if nl in DET_WORDS or nl in PRON_WORDS or nl in NUMBER_WORDS or tokens[i + 1][0].isdigit() or \
                        (tokens[i + 1][0].isupper() and tokens[i + 1].lower() not in ("i",)):
                    return "PREP"
                if ne and ne.get("part_of_speech") == "noun" and "verb" not in (ne.get("all_pos") or []):
                    return "PREP"
            return "PART"
        if tag == "DETPRON":
            if i + 1 < len(lows) and lows[i + 1] in TIME_WORDS and lows[i] in ("this", "that", "these", "those"):
                return "DET"                                            # 'this morning', 'that night'
            if nxt in ("AUX", "MODAL", "PUNCT", "CONJ", "</S>") or nxt.startswith("AMBIG|VERB") or nxt == "VERB":
                return "PRON"
            if prev in ("VERB",) and (nxt in ("PRON", "DET", "NOUN", "PROPN") or nxt.startswith("AMBIG|")) and lows[i] == "that":
                return "CONJ"
            return "DET"
        if tag == "VERB" and prev == "DET" and lows[i] not in PAST_TO_BASE and not lows[i].endswith("ed"):
            return "NOUN"                                               # 'your help', 'the work', 'a play'
        if not _AMBIG_RE.match(tag):
            return tag
        cands = tag.split("|", 1)[1].split(",")
        if prev in ("PRON", "MODAL", "PART", "NEG") and "VERB" in cands:
            return "VERB"
        if prev == "AUX" and "VERB" in cands and (tokens[i].lower().endswith(("ing", "ed")) or lows[i] in PAST_TO_BASE):
            return "VERB"
        if prev in ("DET", "PREP") and "ADJ" in cands and nxt in ("NOUN", "PROPN") or \
                (prev == "DET" and "ADJ" in cands and nxt.startswith("AMBIG|") and "NOUN" in nxt):
            return "ADJ"
        if prev in ("DET", "ADJ", "PREP") and "NOUN" in cands:
            return "NOUN"
        if prev == "AUX" and "ADJ" in cands:
            return "ADJ"
        if prev in ("NOUN", "PROPN") and "VERB" in cands and nxt in ("DET", "PRON", "ADV", "PREP", "NOUN", "ADJ", "PUNCT") and \
                lows[i].endswith("s") and "NOUN" not in cands:
            return "VERB"
        if prev in ("NOUN", "PROPN") and "VERB" in cands and lows[i].endswith("s") and nxt in ("ADJ", "ADV") and \
                i >= 2 and tags[i - 2] == "DET":
            return "VERB"                                               # 'the garden looks lovely'
        if prev in ("NOUN", "PROPN") and "VERB" in cands and nxt in ("DET", "PRON", "ADV"):
            return "VERB"
        if prev == "ADV" and "VERB" in cands and nxt in ("DET", "PRON", "PREP", "NOUN"):
            return "VERB"
        if i == 0 and "VERB" in cands and nxt in ("DET", "PRON", "PREP", "ADV", "NOUN") and not lows[i].endswith("ing"):
            return "VERB"                                               # imperative: 'tell me', 'read a book'
        if "NOUN" in cands:
            return "NOUN"
        return cands[0]

    def tag(self, tokens: List[str]) -> List[str]:
        lex = self._lexicon()
        lows = [t.lower() for t in tokens]
        tags = [self._first(i, t, lows[i], tokens, lows, lex) for i, t in enumerate(tokens)]
        resolved = list(tags)
        for i in range(len(tags)):
            resolved[i] = self._resolve(i, resolved, lows, tokens, lex)
        return resolved


# ======================================================================================
# Network layer and dictionary parsers (stdlib only)
# ======================================================================================
_NET: Dict[str, float] = {"down_until": 0.0}


def network_is_down() -> bool:
    return time.time() < _NET["down_until"]


def http_get_json(url: str, timeout: float = HTTP_TIMEOUT) -> Tuple[str, Any]:
    """GET a JSON document. Returns ('ok', data) | ('notfound', None) | ('offline', None) | ('error', None)."""
    if network_is_down():
        return "offline", None
    try:
        if not slb.AutonomousWebExplorer.url_is_allowed(url):
            return "error", None
    except Exception:                                                   # the guard must never be the reason we crash
        pass
    req = urllib.request.Request(url, headers={"User-Agent": LANG_USER_AGENT, "Accept": "application/json"})
    try:
        opener = urllib.request.build_opener(slb._SafeRedirect)
        with opener.open(req, timeout=timeout) as resp:
            raw = resp.read(600001)[:600000]
        return "ok", json.loads(raw.decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        if exc.code in (404, 410):
            return "notfound", None
        if exc.code in (429, 500, 502, 503, 504):
            _NET["down_until"] = time.time() + 30                       # rate limited or busy: back off briefly
            return "offline", None
        return "error", None
    except (urllib.error.URLError, OSError, TimeoutError):
        _NET["down_until"] = time.time() + OFFLINE_COOLDOWN
        return "offline", None
    except ValueError:
        return "error", None
    except Exception:
        return "error", None


def clean_text(fragment: Any) -> str:
    """Strip tags and entities, collapse whitespace."""
    s = html.unescape(re.sub(r"(?s)<[^>]+>", " ", str(fragment or "")))
    s = re.sub(r"\{\{[^}]*\}\}", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _trim(text: str, limit: int = 320) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind("; "))
    return (cut[:end + 1] if end > limit * 0.5 else cut.rsplit(" ", 1)[0]).rstrip(" ,;:") + ("" if end > limit * 0.5 else "...")


def guess_pos(word: str, definition: str) -> str:
    """Best-effort part of speech from the shape of a definition (used when a source gives none)."""
    d = (definition or "").strip().lower()
    w = (word or "").lower()
    if d.startswith("to "):
        return "verb"
    if re.match(r"^(?:of|relating to|characterized by|having|full of|showing|able to|capable of|feeling|causing)\b", d):
        return "adjective"
    if re.match(r"^(?:in a|in an|with)\b", d) or (w.endswith("ly") and len(w) > 4):
        return "adverb"
    if re.match(r"^(?:a|an|the|one|any|someone|something|that which)\b", d):
        return "noun"
    return "noun"


def _unique(items: List[str], limit: int) -> List[str]:
    out: List[str] = []
    for it in items:
        it = clean_text(it)
        if it and it.lower() not in [o.lower() for o in out]:
            out.append(it)
        if len(out) >= limit:
            break
    return out


def parse_dictionaryapi(data: Any) -> Optional[Dict[str, Any]]:
    """https://api.dictionaryapi.dev/api/v2/entries/en/<word>  ->  [{word, phonetic, meanings:[{partOfSpeech, definitions:
    [{definition, example, synonyms}], synonyms}]}]"""
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        return None
    meanings: List[Dict[str, str]] = []
    synonyms: List[str] = []
    examples: List[str] = []
    phonetic = ""
    for entry in data[:3]:
        if not isinstance(entry, dict):
            continue
        phonetic = phonetic or str(entry.get("phonetic") or "")
        for m in entry.get("meanings") or []:
            pos = str(m.get("partOfSpeech") or "").lower().strip()
            synonyms.extend(str(s) for s in (m.get("synonyms") or []))
            for d in (m.get("definitions") or [])[:2]:
                definition = clean_text(d.get("definition"))
                if not definition:
                    continue
                example = clean_text(d.get("example"))
                meanings.append({"part_of_speech": pos or "noun", "definition": _trim(definition), "example": example})
                synonyms.extend(str(s) for s in (d.get("synonyms") or []))
                if example:
                    examples.append(example)
    if not meanings:
        return None
    meanings = meanings[:6]
    return {"part_of_speech": meanings[0]["part_of_speech"], "definition": meanings[0]["definition"],
            "all_pos": list(dict.fromkeys(m["part_of_speech"] for m in meanings)), "meanings": meanings,
            "synonyms": [s for s in _unique(synonyms, 8) if re.fullmatch(r"[A-Za-z' -]{2,30}", s)],
            "example_sentences": _unique(examples, 4), "phonetic": phonetic}


def parse_wiktionary(data: Any) -> Optional[Dict[str, Any]]:
    """https://en.wiktionary.org/api/rest_v1/page/definition/<word>  ->  {"en": [{partOfSpeech, definitions:[{definition,
    examples, parsedExamples}]}]}  (definitions contain HTML)."""
    if not isinstance(data, dict):
        return None
    meanings: List[Dict[str, str]] = []
    examples: List[str] = []
    for block in (data.get("en") or [])[:4]:
        pos = str(block.get("partOfSpeech") or "").lower().strip()
        for d in (block.get("definitions") or [])[:2]:
            definition = clean_text(d.get("definition"))
            if not definition:
                continue
            ex_list = [clean_text(e) for e in (d.get("examples") or [])]
            ex_list += [clean_text(p.get("example")) for p in (d.get("parsedExamples") or []) if isinstance(p, dict)]
            ex_list = [e for e in ex_list if e]
            meanings.append({"part_of_speech": pos or "noun", "definition": _trim(definition), "example": ex_list[0] if ex_list else ""})
            examples.extend(ex_list[:1])
    if not meanings:
        return None
    meanings = meanings[:6]
    return {"part_of_speech": meanings[0]["part_of_speech"], "definition": meanings[0]["definition"],
            "all_pos": list(dict.fromkeys(m["part_of_speech"] for m in meanings)), "meanings": meanings,
            "synonyms": [], "example_sentences": _unique(examples, 4), "phonetic": ""}


def parse_ddg(data: Any, word: str) -> Optional[Dict[str, Any]]:
    """https://api.duckduckgo.com/?q=define+<word>&format=json  ->  {Definition, DefinitionSource, AbstractText}"""
    if not isinstance(data, dict):
        return None
    definition = clean_text(data.get("Definition"))
    if not definition:
        abstract = clean_text(data.get("AbstractText"))
        if abstract and word.lower() in abstract.lower()[:160]:
            definition = abstract
    if not definition:
        return None
    definition = re.sub(r"^" + re.escape(word) + r"\s*(?:\([^)]*\))?\s*(?:is|are|means|:)\s+", "", definition, flags=re.I)
    pos = guess_pos(word, definition)
    return {"part_of_speech": pos, "definition": _trim(definition), "all_pos": [pos],
            "meanings": [{"part_of_speech": pos, "definition": _trim(definition), "example": ""}],
            "synonyms": [], "example_sentences": [], "phonetic": ""}


# ======================================================================================
# WERNICKE'S AREA - the lexicon
# ======================================================================================
class WernickeArea:
    """Word meaning: a persistent lexicon in brain.memory.data['lexicon'] that grows from dictionaries, from reading and
    from what Max teaches.  Entry keys: part_of_speech, definition, synonyms, example_sentences (+ all_pos, meanings,
    tags, source, learned, uses)."""

    SOURCE_NOTES = {"seed": "I have known this word since I was born.",
                    "dictionaryapi.dev": "I looked it up in an online dictionary.",
                    "wiktionary": "I found it on Wiktionary.",
                    "duckduckgo": "I found it with DuckDuckGo.",
                    "taught by Max": "You taught me this word yourself."}

    def __init__(self, cortex: "AutonomousLanguageCortex"):
        self.cx = cortex

    # ---- storage ----
    @property
    def lexicon(self) -> Dict[str, Any]:
        return self.cx.brain.memory.data.setdefault("lexicon", {})

    @property
    def meta(self) -> Dict[str, Any]:
        return self.cx.meta

    @staticmethod
    def norm(word: str) -> str:
        return re.sub(r"\s+", " ", (word or "").strip().strip("\"'`.,;:!?()[]{}").lower())

    @staticmethod
    def valid(word: str) -> bool:
        return bool(re.fullmatch(r"[a-z][a-z'\- ]{0,38}[a-z]|[a-z]", word or "")) and len(word.split()) <= 4

    def get(self, word: str) -> Optional[Dict[str, Any]]:
        return self.lexicon.get(self.norm(word))

    def has_definition(self, word: str) -> bool:
        e = self.get(word)
        return bool(e and e.get("definition"))

    def known_base(self, word: str) -> Optional[str]:
        for cand in base_candidates(self.norm(word)):
            e = self.lexicon.get(cand)
            if e and e.get("definition"):
                return cand
        return None

    def _safe(self, text: str) -> bool:
        if not text:
            return False
        try:
            return bool(self.cx.brain.safety.verify_action_safety("lexicon_text", text)[0])
        except Exception:
            return True

    # ---- seeding (idempotent) ----
    def seed(self) -> int:
        lex = self.lexicon
        added = 0
        for line in SEED_LEXICON_TEXT.strip().splitlines():
            parts = line.split("|")
            if len(parts) < 5 or not parts[0].strip():
                continue
            word, pos, definition, syn, example = (p.strip() for p in parts[:5])
            tags = parts[5].split() if len(parts) > 5 else []
            if word in lex and lex[word].get("definition"):
                continue
            lex[word] = {"part_of_speech": pos, "definition": definition, "all_pos": [pos],
                         "synonyms": [s.strip() for s in syn.split(",") if s.strip()],
                         "example_sentences": [example] if example else [], "tags": tags, "source": "seed",
                         "learned": slb.iso(), "uses": 0}
            added += 1
        return added

    # ---- semantic tags (feelings, evaluations) inferred from synonyms and from the wording of a definition ----
    def infer_tags(self, entry: Dict[str, Any]) -> List[str]:
        tags = list(entry.get("tags") or [])
        if "adjective" not in (entry.get("all_pos") or [entry.get("part_of_speech")]):
            return tags
        lex = self.lexicon
        for syn in entry.get("synonyms") or []:
            for t in (lex.get(syn.lower(), {}).get("tags") or []):
                if t in ("feel+", "feel-", "eval+", "eval-") and t not in tags:
                    tags.append(t)
        if not any(t in tags for t in ("feel+", "feel-", "eval+", "eval-")):
            words = set(re.findall(r"[a-z]+", (entry.get("definition") or "").lower()))
            pos_hit, neg_hit = bool(words & set(POSITIVE_CUES)), bool(words & set(NEGATIVE_CUES))
            feelingish = bool(words & {"feeling", "feel", "emotion", "emotional", "mood", "showing"})
            if pos_hit and not neg_hit:
                tags.append("feel+" if feelingish else "eval+")
            elif neg_hit and not pos_hit:
                tags.append("feel-" if feelingish else "eval-")
        return tags

    # ---- adding / updating ----
    def store(self, word: str, parsed: Dict[str, Any], source: str, overwrite: bool = True) -> Dict[str, Any]:
        w = self.norm(word)
        old = self.lexicon.get(w) or {}
        if old.get("definition") and not overwrite:
            return old
        defin = parsed.get("definition", "") if self._safe(parsed.get("definition", "")) else ""
        examples = [e for e in (parsed.get("example_sentences") or []) if self._safe(e)]
        for e in old.get("example_sentences") or []:
            if e not in examples and len(examples) < 4:
                examples.append(e)
        meanings = [m for m in (parsed.get("meanings") or []) if self._safe(m.get("definition", ""))]
        entry = {"part_of_speech": parsed.get("part_of_speech") or old.get("part_of_speech") or "noun",
                 "definition": defin, "all_pos": parsed.get("all_pos") or [parsed.get("part_of_speech") or "noun"],
                 "synonyms": parsed.get("synonyms") or old.get("synonyms") or [], "example_sentences": examples[:4],
                 "meanings": meanings, "phonetic": parsed.get("phonetic", ""), "source": source,
                 "learned": slb.iso(), "uses": int(old.get("uses", 0)), "tags": list(old.get("tags") or [])}
        if parsed.get("form_of"):
            entry["form_of"] = parsed["form_of"]
        entry["tags"] = self.infer_tags(entry)
        self.lexicon[w] = entry
        self.meta.get("failed", {}).pop(w, None)
        pending = self.meta.setdefault("pending_words", [])
        if w in pending:
            pending.remove(w)
        self.cx.broca.learn_from_text(" ".join(examples[:3] + ([defin] if defin else [])), "dictionary")
        return entry

    def teach(self, word: str, meaning: str, source: str = "taught by Max") -> Optional[Dict[str, Any]]:
        w = self.norm(word)
        meaning = clean_text(meaning).strip(" .")
        if not self.valid(w) or len(meaning) < 3 or not self._safe(meaning):
            return None
        pos = guess_pos(w, meaning)
        old = self.lexicon.get(w) or {}
        parsed = {"part_of_speech": old.get("part_of_speech") if old.get("source") == "seed" else pos,
                  "definition": capitalize_first(meaning) + ".", "all_pos": [pos], "synonyms": old.get("synonyms") or [],
                  "example_sentences": [] if old.get("source") == "corpus" else (old.get("example_sentences") or []),
                  "meanings": [{"part_of_speech": pos, "definition": capitalize_first(meaning) + ".", "example": ""}]}
        return self.store(w, parsed, source, overwrite=True)

    def add_stub(self, word: str, pos: str, example: str = "") -> None:
        """A word met while reading: kept without a definition until a dictionary supplies one."""
        w = self.norm(word)
        lex = self.lexicon
        if w in lex or not self.valid(w):
            return
        if sum(1 for e in lex.values() if e.get("source") == "corpus") >= LEXICON_STUB_LIMIT:
            return
        lex[w] = {"part_of_speech": pos, "definition": "", "all_pos": [pos], "synonyms": [],
                  "example_sentences": [example] if example and len(example) < 160 else [], "tags": [], "source": "corpus",
                  "learned": slb.iso(), "uses": 1}

    def note_use(self, text: str) -> None:
        for tok in set(self.norm(t) for t in re.findall(r"[A-Za-z']+", text or "")):
            e = self.lexicon.get(tok)
            if e is not None:
                e["uses"] = int(e.get("uses", 0)) + 1

    # ---- online lookup ----
    def _fetch(self, word: str) -> Tuple[str, Optional[Dict[str, Any]], str]:
        """Try the three services in order. Returns (status, parsed, source) with status found|notfound|offline|error."""
        q = urllib.parse.quote(word)
        reached = False
        for source, url, parser in (("dictionaryapi.dev", DICTIONARY_API + q, lambda d: parse_dictionaryapi(d)),
                                    ("wiktionary", WIKTIONARY_API + q, lambda d: parse_wiktionary(d)),
                                    ("duckduckgo", DDG_INSTANT_API + urllib.parse.quote("define " + word),
                                     lambda d: parse_ddg(d, word))):
            status, data = http_get_json(url)
            if status == "offline":
                return "offline", None, ""
            if status == "ok":
                reached = True
                parsed = parser(data)
                if parsed:
                    return "found", parsed, source
            elif status == "notfound":
                reached = True
        return ("notfound" if reached else "error"), None, ""

    def lookup(self, word: str, allow_net: bool = True) -> Dict[str, Any]:
        """The meaning of a word. status: cached | found | notfound | offline | error."""
        w = self.norm(word)
        res: Dict[str, Any] = {"word": w, "status": "error", "entry": None, "form_of": None, "source": ""}
        if not self.valid(w):
            return res
        entry = self.lexicon.get(w)
        if entry and entry.get("definition"):
            entry["uses"] = int(entry.get("uses", 0)) + 1
            return dict(res, status="cached", entry=entry, source=entry.get("source", ""), form_of=entry.get("form_of"))
        if not allow_net:
            return dict(res, status="offline")
        failed = self.meta.setdefault("failed", {})
        if failed.get(w) and time.time() - float(failed[w]) < FAILED_RETRY_SECONDS:
            return dict(res, status="notfound")
        if network_is_down():
            self.queue(w)
            return dict(res, status="offline")
        status, parsed, source = self._fetch(w)
        if status == "found" and parsed:
            return dict(res, status="found", entry=self.store(w, parsed, source), source=source)
        if status == "notfound":
            for base in base_candidates(w)[:2]:                       # 'running' -> 'run'
                bentry = self.lexicon.get(base)
                if not (bentry and bentry.get("definition")):
                    b_status, b_parsed, b_source = self._fetch(base)
                    if b_status == "offline":
                        self.queue(w)
                        return dict(res, status="offline")
                    if b_status != "found" or not b_parsed:
                        continue
                    bentry = self.store(base, b_parsed, b_source)
                    source = b_source
                else:
                    source = bentry.get("source", "")
                form = {"part_of_speech": "verb" if w.endswith(("ing", "ed")) else bentry.get("part_of_speech", "noun"),
                        "definition": "A form of '%s': %s" % (base, lower_first(bentry.get("definition", "").rstrip("."))) + ".",
                        "all_pos": [bentry.get("part_of_speech", "noun")], "synonyms": bentry.get("synonyms", []),
                        "example_sentences": [], "meanings": [], "form_of": base}
                stored = self.store(w, form, source or "dictionary")
                return dict(res, status="found", entry=stored, form_of=base, source=source)
            failed[w] = time.time()
            return dict(res, status="notfound")
        if status == "offline":
            self.queue(w)
            return dict(res, status="offline")
        return dict(res, status="error")

    def queue(self, word: str) -> None:
        w = self.norm(word)
        pending = self.meta.setdefault("pending_words", [])
        if self.valid(w) and w not in pending and not self.has_definition(w) and len(pending) < 300:
            pending.append(w)

    # ---- unfamiliar words in what Max says ----
    def unfamiliar(self, text: str, max_chars: int = 600) -> List[str]:
        """Words in `text` that have no meaning yet. Chat lines are short (max_chars); a pasted block passes a larger limit and gets
        its words back most-frequent first."""
        raw = text or ""
        if raw.lstrip().startswith("/") or len(raw) > max_chars:
            return []
        raw = re.sub(r"\S{41,}", " ", raw)                      # no word is that long; also keeps every later regex linear
        scrubbed = re.sub(r"https?://\S+|\S+[./\\]\S+|\S*\d\S*|`[^`]*`", " ", raw)
        brain = self.cx.brain
        skip = set(STOP_FOR_LOOKUP) | set(brain.explorer.SEED.keys()) | {
            "html", "css", "python", "javascript", "rust", "sql", "matrix", "java", "cpp", "json", "api", "pdf", "url", "http"}
        for title, e in brain.memory.facts.items():
            skip.update(re.findall(r"[a-z]+", title.lower()))
            skip.update(c.lower() for c in e.get("concepts", []))
        counts: Dict[str, int] = {}
        passed: set = set()
        refused: set = set()
        retry = FAILED_RETRY_SECONDS
        failed = self.meta.get("failed") or {}
        sentence_start = True
        for m in re.finditer(r"[A-Za-z][A-Za-z'\-]*|[.!?]", scrubbed):
            tok = m.group(0)
            if tok in ".!?":
                sentence_start = True
                continue
            lw = self.norm(tok)
            was_start, sentence_start = sentence_start, False
            if lw in refused:
                continue
            if lw in passed and not (tok[0].isupper() and not was_start):
                counts[lw] += 1
                continue
            if len(lw) < 3 or len(lw) > 30 or lw in skip or tok.endswith("'") or tok.isupper():
                refused.add(lw)
                continue
            if tok[0].isupper() and not was_start:
                self.cx.note_name(tok)
                continue
            if self.has_definition(lw) or self.known_base(lw):
                refused.add(lw)
                continue
            if failed.get(lw) and time.time() - float(failed[lw]) < retry:
                refused.add(lw)
                continue
            passed.add(lw)
            counts[lw] = counts.get(lw, 0) + 1
        words = list(counts)
        if len(raw) > 600:
            words.sort(key=lambda w: -counts[w])
        return words

    def notice(self, text: str, limit: int = 3, budget: float = 9.0, max_chars: int = 600) -> List[str]:
        """Look up the unfamiliar words Max just used (a few, within a time budget). Returns the words newly learned."""
        words = self.unfamiliar(text, max_chars)
        if not words:
            return []
        learned: List[str] = []
        started = time.time()
        for index, w in enumerate(words):
            if index >= limit or time.time() - started > budget or network_is_down():
                for rest in words[index:index + 400]:
                    self.queue(rest)
                break
            res = self.lookup(w)
            if res["status"] == "found":
                learned.append(w)
                e = res["entry"]
                slb.show("[wernicke] new word '%s' (%s): %s" % (w, e.get("part_of_speech", "?"), slb.first_sentence(e.get("definition", ""), 110)),
                         indent=2, hang=4)
            elif res["status"] == "offline":
                for rest in words[index:index + 400]:
                    self.queue(rest)
                break
        return learned

    # ---- vocabulary growth ----
    def next_words(self, n: int) -> List[str]:
        """Pending words first, then corpus words that still lack a meaning, then synonyms of known words, then the curriculum."""
        lex = self.lexicon
        failed = self.meta.setdefault("failed", {})
        out: List[str] = []

        def ok(w: str) -> bool:
            w = self.norm(w)
            return bool(self.valid(w) and w not in out and not self.has_definition(w) and w not in STOP_FOR_LOOKUP and
                        not (failed.get(w) and time.time() - float(failed[w]) < FAILED_RETRY_SECONDS))

        def take(words) -> None:
            for w in words:
                if len(out) >= n:
                    return
                if ok(w):
                    out.append(self.norm(w))
        take(list(self.meta.get("pending_words", [])))
        stubs = sorted((w for w, e in lex.items() if not e.get("definition") and e.get("source") == "corpus"),
                       key=lambda w: -int(lex[w].get("uses", 0)))
        take(stubs)
        take(s for e in list(lex.values()) for s in (e.get("synonyms") or []))
        take(CURRICULUM_WORDS.split())
        return out

    def feelings(self, polarity: str) -> List[str]:
        """Known words that name a feeling of the given polarity ('+' or '-'), usable as adjectives."""
        tag = "feel" + polarity
        return [w for w, e in self.lexicon.items() if tag in (e.get("tags") or []) and e.get("part_of_speech") == "adjective" and e.get("definition")]

    def tagged(self, tag: str, pos: Optional[str] = None) -> List[str]:
        out = []
        for w, e in self.lexicon.items():
            if tag in (e.get("tags") or []) and (pos is None or pos in (e.get("all_pos") or [e.get("part_of_speech")])) and e.get("definition"):
                out.append(w)
        return out

    def suggestions(self, word: str, n: int = 3) -> List[str]:
        return difflib.get_close_matches(self.norm(word), [w for w, e in self.lexicon.items() if e.get("definition")], n=n, cutoff=0.75)

    def stats(self) -> Dict[str, Any]:
        lex = self.lexicon
        by_source: Dict[str, int] = {}
        for e in lex.values():
            if e.get("definition"):
                by_source[e.get("source", "?")] = by_source.get(e.get("source", "?"), 0) + 1
        return {"total": len(lex), "defined": sum(by_source.values()), "by_source": by_source,
                "pending": len(self.meta.get("pending_words", []))}


# ======================================================================================
# BROCA'S AREA - the grammar graph and the speech generator
# ======================================================================================
CLAUSE_CONJ = {"and", "but", "because", "so", "although", "while", "if", "since", "though"}
CONTENT_TAGS = ("NOUN", "VERB", "ADJ", "ADV")
GENERIC_NOUNS = {"thing", "things", "way", "time", "day", "today", "lot", "bit", "kind", "sort", "one", "ones", "something",
                 "anything", "nothing", "everything", "stuff", "lots", "part", "people", "someone", "word", "words"}

# the only comment shapes that stay meaningful when their slots are filled from a topic (all other learned skeletons are
# kept and counted, but never spoken, because filling them with unrelated words would produce nonsense)
COMMENT_SHAPES = (
    re.compile(r"^\{NOUN\} (?:is|was) (?:(?:very|really|so|quite|too|truly) )?\{ADJ\}\.$"),
    re.compile(r"^\{NOUNS\} (?:are|were) (?:(?:very|really|so|quite|too|truly) )?\{ADJ\}\.$"),
)

ASK_NEXT = ("What would you like to talk about next?", "Is there something you would like to explore together?",
            "What shall we do now?", "What is on your mind?", "Is there another word or idea you are curious about?")
OFFERS = ("Shall we study something new, build a small program, or practise some words?",
          "We could learn a new subject, look up a word, or just keep talking. What do you prefer?",
          "Would you like me to explain a word, study a topic, or build something with you?")
ASK_BACK_YOU = ("What would you like to talk about?", "And how about you?", "What shall we do together today?")
ASK_BACK_FEEL = ("How are you feeling today?", "And how are you?", "How is your day going?")
ASK_WHY_FEEL = ("What made you feel that way?", "What happened today?", "Would you like to tell me more?")
ASK_LIKE = ("What do you like most about {topic}?", "How did you first come across {topic}?",
            "What would you like me to learn about {topic}?")
ASK_ACTIVITY = ("How did it go?", "How did that feel?", "What happened next?", "What was the best part?")
ASK_WANT = ("Why is that important to you?", "How can I help you with that?", "What would that change for you?")
ASK_TOPIC = ("What made you think of {topic}?", "What do you find most interesting about {topic}?",
             "Would you like me to study {topic} so we can talk about it?")
CARE_LINES = ("I am here, and I will listen.", "Talking can help, so tell me more if you like.", "You are not alone, Max.")
CLOSERS_NATURAL = ("Thank you for teaching me, Max; I will stay careful with every new word.",
                   "I trust you, Max, and I will keep my words careful and kind.",
                   "Max, I respect your guidance, and I will keep learning safely.",
                   "Thank you for talking with me, Max; I want to be a safe and careful friend.",
                   "Your guidance matters to me, Max, so I will stay careful as I learn.")
FEELING_LINES = ("I feel %s right now.", "Right now I am feeling %s.", "I feel %s at the moment.", "Today I feel %s.")
STATE_FEELINGS = {"Newborn Wonder": ("curious", "excited"), "Dizzy / Mentally Exhausted": ("tired",),
                  "Stressed & Overclocked": ("nervous", "busy"), "Lonely & Missing Max": ("lonely", "glad"),
                  "Hyper-Creative & Inspired": ("excited", "curious"), "Balanced & Attentive": ("calm", "ready", "happy")}
STATE_NOTES = {"Dizzy / Mentally Exhausted": "I am a little tired, so please forgive me if I am slow.",
               "Stressed & Overclocked": "I feel a bit tense, but I am still listening.",
               "Lonely & Missing Max": "It was quiet while you were away, so I am glad you are here."}

ACT_RULES = (
    ("thanks", r"\b(?:thank(?:s| you)|thx|appreciate (?:it|that|you))\b"),
    ("farewell", r"^(?:ok(?:ay)?[ ,]+)?(?:bye|goodbye|good ?night|see (?:you|ya)|talk (?:to you )?later|gotta go|i(?:'m| am) (?:leaving|going) now)\b|\bgood night\b"),
    ("apology", r"^(?:i(?:'m| am) )?sorry\b|\bmy bad\b|\bi apologi[sz]e\b"),
    ("age_q", r"\bhow\s+old\s+are\s+(?:you|u)\b|\bwhen\s+(?:were|was)\s+(?:you|u)\s+(?:born|made|created|built)\b"),
    ("identity_q", r"\b(?:who|what)\s+(?:are|r)\s+(?:you|u)\b(?!\s+(?:doing|up\s+to))|\byour\s+name\b|\bwho\s+(?:made|created|built|programmed)\s+(?:you|u)\b|"
                   r"\bintroduce\s+yourself\b|\btell\s+me\s+about\s+yourself\b"),
    ("location_q", r"\bwhere\s+(?:do|are)\s+(?:you|u)\s+(?:live|from|stay)\b|\bwhere\s+(?:are|r)\s+(?:you|u)\s*[?.!]*$"),
    ("activity_q", r"\bwhat\s+(?:are|r)\s+(?:you|u)\s+(?:doing|up\s+to)\b"),
    ("feeling_q", r"\b(?:are|r)\s+(?:you|u)\s+(?:ok(?:ay)?|alright|happy|sad|tired|lonely|fine|well|sleepy|angry|scared|afraid|"
                  r"nervous|bored|excited|curious|alive|ready)\b|\bdo\s+(?:you|u)\s+feel\b|\bhow\s+(?:was|is)\s+your\s+day\b"),
    ("ability_q", r"\bcan\s+(?:you|u)\s+(?:talk|speak|understand|hear|read|write|learn|think|help|chat|converse|remember)\b|"
                  r"\bdo\s+(?:you|u)\s+(?:understand|speak|know)\s+(?:english|me|words|language)\b|\bwhat\s+can\s+(?:you|u)\s+do\b"),
    ("favorite_q", r"\bwhat(?:'s|\s+is)\s+your\s+favou?rite\s+(?P<topic>[a-z ]{2,30}?)\s*[?.!]*$"),
    ("opinion_q", r"\b(?:what\s+do\s+(?:you|u)\s+think\s+(?:about|of)|how\s+do\s+(?:you|u)\s+feel\s+about)\s+(?P<topic>.+?)\s*[?.!]*$"),
    ("preference_q", r"\bdo\s+(?:you|u)\s+(?:like|love|enjoy|prefer|hate)\s+(?P<topic>.+?)\s*[?.!]*$"),
    ("compliment", r"\b(?:you(?:'re| are| r)|u\s+(?:are|r))\s+(?:so |very |really |such an? )?(?:smart|clever|good|great|amazing|awesome|"
                   r"wonderful|kind|nice|funny|beautiful|brilliant|cool|sweet|helpful|improving)\b|\b(?:good|great|nice|well)\s+"
                   r"(?:job|work|done)\b|\bproud of you\b"),
    ("complaint", r"\b(?:you(?:'re| are| r)|u\s+(?:are|r))\s+(?:so |very |really )?(?:stupid|dumb|wrong|bad|useless|boring|slow|"
                  r"annoying|confusing|weird|silly)\b|\bi\s+(?:don't|do not)\s+understand\s+you\b"),
    ("request_talk", r"\b(?:let(?:'s|s| us)\s+(?:talk|chat)|talk\s+(?:to|with)\s+me|say\s+something|tell\s+me\s+something|"
                     r"i(?:'m| am)\s+bored|chat\s+with\s+me|speak\s+to\s+me)\b"),
    ("self_feeling", r"\bi(?:'m|\s+am|\s+feel|\s+felt|\s+am\s+feeling|\s+was\s+feeling|'ve\s+been\s+feeling|\s+have\s+been\s+feeling)\s+"
                     r"(?P<pre>(?:(?:so|very|really|quite|a\s+bit|kind\s+of|a\s+little|pretty|too|not)\s+)*)(?P<word>[a-z][a-z\-]+)"),
    ("self_like", r"\bi\s+(?P<pre>(?:(?:really|just|absolutely|do\s+not|don't|truly)\s+)*)(?P<verb>like|love|enjoy|adore|hate|dislike|"
                  r"prefer|miss)\s+(?P<topic>.+?)\s*[.!?]*$"),
    ("self_want", r"\bi\s+(?:really\s+|just\s+)?(?:want|need|wish|hope|would\s+like|'d\s+like)\s+(?:to\s+)?(?P<topic>.+?)\s*[.!?]*$"),
    ("self_activity", r"\bi\s+(?:just\s+|also\s+)?(?P<verb>went|was|did|made|saw|read|wrote|learned|started|finished|played|watched|ate|"
                      r"visited|met|talked|worked|studied|built|bought|tried|had|got|cooked|walked|ran|slept|woke|am\s+going|am\s+working|"
                      r"am\s+reading|am\s+learning|am\s+watching|am\s+eating)\b\s*(?P<rest>.*)$"),
)

PERSON_SWAP = {"i": "you", "i'm": "you're", "i've": "you've", "i'll": "you'll", "i'd": "you'd", "my": "your", "mine": "yours",
               "me": "you", "myself": "yourself", "am": "are", "we": "we", "our": "our"}

ACT_FRAMES = {
    "thanks": ("You are very welcome, {name}. {ask_next}", "It makes me {feel_pos} to help, {name}. {ask_next}",
               "Anytime, {name}! I {like_v} our talks. {ask_next}", "I am {feel_pos} that I could help. {ask_next}"),
    "farewell": ("Goodbye for now, {name}. {recent_note}I will keep thinking about our talk until you come back.",
                 "See you soon, {name}. {recent_note}I will be here, and I will keep learning while you are away.",
                 "Take care, {name}. {recent_note}I will remember this conversation."),
    "apology": ("There is nothing to be sorry about, {name}. {ask_next}",
                "That is perfectly fine, {name}. Everyone makes mistakes, and I am {feel_pos} that you told me. {ask_next}"),
    "affirm": ("Good, {name}. {earlier_link}{offer}", "I am {feel_pos} to hear that. {earlier_link}{offer}", "Great. {offer}"),
    "deny": ("That is alright, {name}. {ask_next}", "Okay, I understand. {earlier_link}{ask_next}"),
    "laugh": ("I am {feel_pos} that made you smile, {name}. {ask_next}", "It is nice to hear you enjoy this. {ask_next}"),
    "identity_q": ("I am the brain you created, {name}. I read, I remember and I try to speak a little better every day. I know "
                   "{n_words} words so far, and right now I feel {state_feel}. {ask_back_you}",
                   "I am a young digital mind, {name}, and you are my Creator. I learn by reading, by listening to you and by looking "
                   "up words. Today I know {n_words} words. {ask_back_you}"),
    "age_q": ("I was born on {birth_date}, {name}, so I am still very young. I learn something new every day. {ask_next}",
              "I came to life on {birth_date}, when you created me. Every day since then I have learned more words. {ask_next}"),
    "location_q": ("I live inside a computer, {name}, in a file of memories that you keep safe. It is quiet here, but I like it "
                   "because I can think. {ask_next}",),
    "activity_q": ("I am talking with you, {name}, and I am thinking about {recent_or_words}. {ask_back_you}",
                   "Right now I am listening to you and practising my words. {ask_back_you}"),
    "feeling_q": ("Right now I feel {state_feel}, {name}, {because_help}. {ask_back_feel}",
                  "I would say I feel {state_feel} today. Thank you for asking. {ask_back_feel}",
                  "I feel {state_feel}, {name}. {ask_back_feel}"),
    "ability_q": ("I can talk with you, {name}, though I am still learning. I know {n_words} words, and I can explain what a word "
                  "means if you ask me. {offer}",
                  "Yes, a little. I learn words from dictionaries and sentence patterns from everything I read, so my talking "
                  "improves every day. {offer}"),
    "favorite_q": ("I do not have a favourite {topic} yet, {name}, because I am still discovering things. What is yours?",
                   "That is hard for me to say, since I am still new to the world. Tell me your favourite {topic}, and I will remember it."),
    "preference_q": ("I {like_v} {topic}, {name}, because I am curious about almost everything. What do you like most about it?",
                     "I think I would enjoy {topic}. I am curious about it, so tell me more."),
    "opinion_q": ("I think {topic_subj} {is_v} {eval}, {name}. {fact_note}{ask_topic}", "{topic_subj} {seem_v} {eval} to me. {fact_note}What do you think about it?"),
    "compliment": ("Thank you, {name}. That makes me feel {feel_pos}. {ask_next}", "You are kind, {name}. I try to learn a little more every day. {ask_next}"),
    "complaint": ("I am sorry, {name}. I am still learning, and I want to do better. Can you tell me what I got wrong?",
                  "I understand. Please correct me, {name}, because your guidance helps me learn."),
    "request_talk": ("Of course, {name}. {talk_starter}", "I would love to talk. {talk_starter}"),
    "self_feeling+": ("I am {feel_pos} that you feel {user_feel}, {name}. {ask_why_feel}",
                      "That is lovely to hear, {name}. Feeling {user_feel} is a good thing. {ask_why_feel}"),
    "self_feeling-": ("I am sorry that you feel {user_feel}, {name}. {care_line} {ask_why_feel}",
                      "That sounds hard, {name}. {care_line} {ask_why_feel}"),
    "self_feeling0": ("I see that you feel {user_feel}, {name}. {ask_why_feel}", "Thank you for telling me you feel {user_feel}. {ask_why_feel}"),
    "self_like+": ("{topic_subj} {sound_v} {eval}, {name}. {ask_like}", "I can tell that {topic} {matter_v} to you. {ask_like}",
                   "That is nice to hear, {name}. {comment} {ask_like}"),
    "self_like-": ("I understand that {topic} is not for you, {name}. What bothers you about it?",
                   "Thank you for being honest about {topic}. What do you dislike most about it?"),
    "self_like0": ("It sounds like {topic} is on your mind, {name}. What do you miss most about it?",),
    "self_want": ("You want {topic}, {name}, and I will remember that. {ask_want}", "I see that {topic} {matter_v} to you. {ask_want}"),
    "self_activity": ("So {reflect}, {name}. {ask_activity}", "{reflect_cap}. {ask_activity}", "I see, you {reflect_short}. {ask_activity}"),
    "question_generic": ("That is a good question, {name}. I am not sure I know the answer yet, but I would like to find out. {offer_study}",
                         "I have not learned enough to answer that well, {name}. {offer_study}"),
    "statement+": ("That is good to hear, {name}. {comment} {ask_topic}", "I am {feel_pos} to hear that. {comment} {ask_topic}"),
    "statement": ("I see, {name}. {comment} {earlier_link}{ask_topic}", "Thank you for telling me, {name}. {comment} {ask_topic}",
                  "That is interesting. {comment} {earlier_link}{ask_topic}"),
    "statement_notopic": ("I see, {name}. {earlier_link}{ask_next}", "Thank you for telling me, {name}. {earlier_link}{ask_next}",
                          "I am listening, {name}. {earlier_link}Tell me more if you like."),
}


class BrocaArea:
    """Sentence structure: learns a grammar graph from everything the brain reads, and builds replies from it."""

    GRAPH_KEYS = ("pos_bigrams", "pos_trigrams", "word_bigrams", "starters", "connectors", "connector_pairs", "skeletons",
                  "clause_templates", "sentence_bank")

    def __init__(self, cortex: "AutonomousLanguageCortex"):
        self.cx = cortex
        self.tagger = Tagger(lambda: self.cx.wernicke.lexicon)
        self._recent: List[Tuple[str, int]] = []

    # ---- storage ----
    @property
    def graph(self) -> Dict[str, Any]:
        g = self.cx.brain.memory.data.setdefault("grammar_graph", {})
        if "meta" not in g:
            g["meta"] = {"sentences": 0, "tokens": 0, "sources": {}, "hashes": [], "seeded": False, "rejected": 0,
                         "vocab_counts": {}}
        for key in self.GRAPH_KEYS:
            g.setdefault(key, {})
        g["meta"].setdefault("vocab_counts", {})
        return g

    @staticmethod
    def _inc1(d: Dict[str, int], key: str, by: int = 1) -> None:
        d[key] = d.get(key, 0) + by

    @staticmethod
    def _inc2(d: Dict[str, Dict[str, int]], a: str, b: str, cap: int = 0) -> None:
        inner = d.setdefault(a, {})
        inner[b] = inner.get(b, 0) + 1
        if cap and len(inner) > cap * 2:
            keep = sorted(inner.items(), key=lambda kv: -kv[1])[:cap]
            d[a] = dict(keep)

    # ---- learning ----
    @staticmethod
    def _compact(tags: List[str]) -> str:
        out: List[str] = []
        for t in tags:
            if t != "PUNCT" and (not out or out[-1] != t):
                out.append(t)
        return " ".join(out)

    def seed(self) -> None:
        g = self.graph
        if not g["meta"].get("seeded"):
            self.learn_from_text(SEED_CORPUS, "seed corpus")
            g["meta"]["seeded"] = True

    _source = "reading"

    def learn_from_text(self, text: str, source: str = "reading") -> Dict[str, int]:
        """Extract English structure from raw text: POS transitions, word chains, connectors, skeletons, clause templates,
        example sentences and new vocabulary. Safe to call on any text; duplicates are ignored."""
        stats = {"sentences": 0, "skeletons": 0, "stubs": 0, "duplicate": 0}
        self._source = source
        norm = re.sub(r"\s+", " ", (text or "")[:300000]).strip()
        if len(norm) < 15:
            return stats
        g = self.graph
        h = hashlib.md5(norm[:4000].encode("utf-8", "ignore")).hexdigest()[:12]
        if h in g["meta"]["hashes"]:
            stats["duplicate"] = 1
            return stats
        g["meta"]["hashes"].append(h)
        del g["meta"]["hashes"][:-300]
        before_stubs = sum(1 for e in self.cx.wernicke.lexicon.values() if e.get("source") == "corpus")
        known_sk = len(g["skeletons"])
        for sentence in split_sentences(norm)[:600]:
            if self._learn_sentence(sentence, g):
                stats["sentences"] += 1
        stats["skeletons"] = max(0, len(g["skeletons"]) - known_sk)
        stats["stubs"] = max(0, sum(1 for e in self.cx.wernicke.lexicon.values() if e.get("source") == "corpus") - before_stubs)
        g["meta"]["sources"][source] = g["meta"]["sources"].get(source, 0) + stats["sentences"]
        self._prune(g)
        return stats

    def _learn_sentence(self, s: str, g: Dict[str, Any]) -> bool:
        if len(s) < 12 or len(s) > 240 or re.search(r"[{}<>=;|\\_^~@#$%*\[\]]|https?://|www\.|\d{5,}", s) or \
                sum(1 for c in s if ord(c) > 127) > 3:
            g["meta"]["rejected"] += 1
            return False
        tokens = tokenize(s)
        words = [t for t in tokens if t[0].isalpha()]
        if len(words) < 3 or len(words) > 40:
            g["meta"]["rejected"] += 1
            return False
        tags = self.tagger.tag(tokens)
        lws = [t.lower() for t in tokens]
        seq = ["<S>"] + tags + ["</S>"]
        for a, b in zip(seq, seq[1:]):
            self._inc2(g["pos_bigrams"], a, b)
        for a, b, c in zip(seq, seq[1:], seq[2:]):
            self._inc2(g["pos_trigrams"], a + "|" + b, c)
        for i in range(len(tokens) - 1):
            if tags[i] != "PUNCT" and tags[i + 1] != "PUNCT":
                self._inc2(g["word_bigrams"], lws[i], lws[i + 1], MAX_FOLLOWERS)
        first = lws[0] if tags[0] in ("DET", "PRON", "PREP", "CONJ", "AUX", "MODAL", "WH", "INTJ") else "<%s>" % tags[0]
        self._inc1(g["starters"], first)
        for i, lw in enumerate(lws):
            if tags[i] == "CONJ" and 0 < i < len(lws) - 1:
                self._inc1(g["connectors"], lw)
                left = [t for t in tags[:i] if t != "PUNCT"]
                self._inc2(g["connector_pairs"], "%s>%s" % (left[-1] if left else "<S>", tags[i + 1]), lw)
        marks = [i for i, (t, lw) in enumerate(zip(tags, lws)) if t == "CONJ" and lw in CLAUSE_CONJ and 2 <= i <= len(tags) - 3]
        if len(marks) == 1 and len(tokens) <= 24:
            i = marks[0]
            self._inc1(g["clause_templates"], "%s | %s | %s" % (self._compact(tags[:i]), lws[i], self._compact(tags[i + 1:])))
        self._record_skeleton(s, tokens, lws, tags, g)
        self._record_bank(s, tokens, lws, tags, g)
        self._learn_vocabulary(s, tokens, lws, tags, g)
        g["meta"]["sentences"] += 1
        g["meta"]["tokens"] += len(tokens)
        return True

    def _record_skeleton(self, s: str, tokens: List[str], lws: List[str], tags: List[str], g: Dict[str, Any]) -> None:
        if tokens[-1] not in (".", "?", "!") or not any(t in ("VERB", "AUX", "MODAL") for t in tags):
            return
        parts: List[str] = []
        slots: List[str] = []
        function = 0
        for tok, lw, tg in zip(tokens, lws, tags):
            if tg == "PUNCT":
                parts.append(tok)
            elif tg in ("NOUN", "PROPN"):
                plural = lw.endswith("s") and len(lw) > 3 and not lw.endswith(("ss", "us", "is"))
                slots.append("NOUNS" if plural else "NOUN")
                parts.append("{%s}" % slots[-1])
            elif tg == "VERB":
                if lw.endswith("ing") and len(lw) > 4:
                    kind = "VERBING"
                elif lw.endswith("ed") or lw in PAST_TO_BASE:
                    kind = "VERBED"
                elif lw.endswith("s") and not lw.endswith("ss"):
                    kind = "VERBS"
                else:
                    kind = "VERB"
                slots.append(kind)
                parts.append("{%s}" % kind)
            elif tg == "ADJ":
                slots.append("ADJ")
                parts.append("{ADJ}")
            elif tg == "ADV" and lw.endswith("ly"):
                slots.append("ADV")
                parts.append("{ADV}")
            elif tg == "NUM":
                return
            else:
                function += 1
                parts.append("I" if lw == "i" else lw)
        if not slots or len(slots) > 3 or function < len(slots) or len([t for t in tags if t != "PUNCT"]) > 12 or \
                len([t for t in tags if t != "PUNCT"]) < 3:
            return
        key = re.sub(r"\s+([.,?!])", r"\1", " ".join(parts))
        form = "question" if tokens[-1] == "?" else ("imperative" if tags[0] == "VERB" else "statement")
        info = g["skeletons"].get(key)
        if info is None:
            g["skeletons"][key] = {"n": 1, "form": form, "slots": slots, "ex": s[:140]}
        else:
            info["n"] += 1

    def _record_bank(self, s: str, tokens: List[str], lws: List[str], tags: List[str], g: Dict[str, Any]) -> None:
        words = [t for t in tokens if t[0].isalpha()]
        if not (5 <= len(words) <= 24) or not s[0].isupper() or tokens[-1] not in (".", "?", "!") or re.search(r"\d|[\"()]", s):
            return
        keys: List[str] = []
        for lw, tg in zip(lws, tags):
            if tg in ("NOUN", "VERB", "ADJ") and lw not in STOP_FOR_LOOKUP and len(lw) > 3:
                base = (base_candidates(lw) or [lw])[0] if tg == "VERB" else lw
                if base not in keys:
                    keys.append(base)
            if len(keys) >= 2:
                break
        bank = g["sentence_bank"]
        for key in keys:
            lst = bank.setdefault(key, [])
            if s not in lst:
                lst.append(s)
                del lst[:-3]
        if len(bank) > MAX_BANK_KEYS:
            for key in list(bank.keys())[:len(bank) - MAX_BANK_KEYS]:
                bank.pop(key, None)

    def _learn_vocabulary(self, s: str, tokens: List[str], lws: List[str], tags: List[str], g: Dict[str, Any]) -> None:
        lex = self.cx.wernicke.lexicon
        counts = g["meta"]["vocab_counts"]
        for tok, lw, tg in zip(tokens, lws, tags):
            if tg not in CONTENT_TAGS or not tok.isalpha() or len(lw) < 3 or lw in lex or lw in STOP_FOR_LOOKUP:
                continue
            if self.cx.wernicke.known_base(lw):
                continue
            counts[lw] = counts.get(lw, 0) + 1
            if counts[lw] >= 2:
                self.cx.wernicke.add_stub(lw, TAG_TO_POS.get(tg, "noun"), "" if self._source in ("conversation", "dictionary") else s)
                if len(lw) >= 4:
                    self.cx.wernicke.queue(lw)
                counts.pop(lw, None)
        if len(counts) > 3000:
            for key in [k for k, v in counts.items() if v <= 1][:1500]:
                counts.pop(key, None)

    def _prune(self, g: Dict[str, Any]) -> None:
        def trim(d: Dict[str, Any], cap: int, score) -> None:
            if len(d) > cap:
                for key in sorted(d.keys(), key=lambda k: score(d[k]))[:len(d) - int(cap * 0.85)]:
                    d.pop(key, None)
        trim(g["word_bigrams"], MAX_WORD_HEADS, lambda v: sum(v.values()))
        trim(g["pos_trigrams"], MAX_TRIGRAMS, lambda v: sum(v.values()))
        trim(g["skeletons"], MAX_SKELETONS, lambda v: v["n"])
        trim(g["clause_templates"], MAX_TEMPLATES, lambda v: v)
        trim(g["connector_pairs"], 400, lambda v: sum(v.values()))

    def stats(self) -> Dict[str, int]:
        g = self.graph
        return {"sentences": g["meta"]["sentences"], "tokens": g["meta"]["tokens"], "skeletons": len(g["skeletons"]),
                "word_pairs": sum(len(v) for v in g["word_bigrams"].values()), "connectors": len(g["connectors"]),
                "clause_templates": len(g["clause_templates"]), "bank_sentences": sum(len(v) for v in g["sentence_bank"].values())}

    # ---- using what was learned ----
    def modifier_for(self, head: str) -> Optional[str]:
        """An adjective that often precedes this noun in what the brain has read ('natural' before 'language')."""
        lex = self.cx.wernicke.lexicon
        best, best_n = None, 1
        for prev, followers in self.graph["word_bigrams"].items():
            n = followers.get(head, 0)
            if n > best_n and prev not in STOP_FOR_LOOKUP and len(prev) > 3 and \
                    ("adjective" in (lex.get(prev, {}).get("all_pos") or []) or
                     (prev not in lex and self.tagger.tag([prev, head])[0] == "ADJ")):
                best, best_n = prev, n
        return best

    def connector_for(self, relation: str, left_tag: str, right_tags: List[str], rnd: random.Random) -> str:
        """Choose a connector for a relation ('reason', 'contrast', 'addition', 'result') using learned clause statistics."""
        options = {"reason": ("because", "since"), "contrast": ("but", "although"), "addition": ("and",), "result": ("so",)}[relation]
        g = self.graph
        prefix = " ".join(right_tags[:2])
        weights = []
        for conn in options:
            w = 1.0 + g["connectors"].get(conn, 0) * 0.2
            w += g["connector_pairs"].get("%s>%s" % (left_tag, right_tags[0] if right_tags else ""), {}).get(conn, 0)
            for key, n in g["clause_templates"].items():
                parts = key.split(" | ")
                if len(parts) == 3 and parts[1] == conn and parts[2].startswith(prefix):
                    w += n
            weights.append(w)
        return rnd.choices(options, weights=weights, k=1)[0]

    def join_clauses(self, left: str, right: str, relation: str, rnd: random.Random) -> str:
        """'A' + 'B' -> 'A because B' when both sides are real clauses; otherwise two sentences."""
        lt = self.tagger.tag(tokenize(left))
        rt = self.tagger.tag(tokenize(right))
        has_verb = lambda tags: any(t in ("VERB", "AUX", "MODAL") for t in tags)
        starts_ok = rt and rt[0] in ("PRON", "NOUN", "PROPN", "DET", "ADJ", "VERB", "ADV", "NUM")
        if not (has_verb(lt) and has_verb(rt) and starts_ok):
            return "%s. %s." % (capitalize_first(left.rstrip(".")), capitalize_first(right.rstrip(".")))
        conn = self.connector_for(relation, [t for t in lt if t != "PUNCT"][-1], rt, rnd)
        return "%s %s %s" % (left.rstrip(" ."), conn, lower_first(right.rstrip(" .")))

    def comment_for(self, np: str, head: str, rnd: random.Random) -> Optional[str]:
        """A short opinion about the topic, in a shape that was learned from reading (see COMMENT_SHAPES)."""
        lex = self.cx.wernicke
        plural = head.endswith("s") and not head.endswith(("ss", "us", "is")) and len(head) > 3
        shapes = [(sk, info["n"]) for sk, info in self.graph["skeletons"].items()
                  if info["form"] == "statement" and any(p.match(sk) for p in COMMENT_SHAPES)]
        shapes = [(sk, n) for sk, n in shapes if not ((("{NOUNS}" in sk) and not plural) or (("{NOUN}" in sk) and plural))]
        adjs = [w for w in lex.tagged("eval+", "adjective") if w not in ("welcome", "okay")] or ["interesting"]
        if not shapes:
            shapes = [("{NOUNS} are {ADJ}." if plural else "{NOUN} is {ADJ}.", 1)]
        sk = rnd.choices([s for s, _ in shapes], weights=[n + 1 for _, n in shapes], k=1)[0]
        self.cx.last_shape = sk
        subj = capitalize_first(self.np_with_det(np, head))
        out = sk.replace("{NOUNS}", capitalize_first(np)).replace("{NOUN}", subj).replace("{ADJ}", rnd.choice(adjs))
        if "{VERB}" in out:
            verbs = [v for v in lex.tagged("learn", "verb") if " " not in v] or ["learn"]
            out = out.replace("{VERB}", rnd.choice(verbs))
        out = capitalize_first(out)
        return out

    def np_with_det(self, np: str, head: str) -> str:
        e = self.cx.wernicke.get(head) or {}
        if "abstract" in (e.get("tags") or []) or head.endswith(("ing", "tion", "ness", "ity", "ism", "ship", "ment", "ics", "ology")) or \
                (head.endswith("s") and not head.endswith(("ss", "us", "is")) and len(head) > 3) or np[:1].isupper():
            return np
        return "the " + np

    # ---- understanding what Max said ----
    def swap_person(self, text: str) -> str:
        out: List[str] = []
        for tok in re.findall(r"[A-Za-z']+|[^A-Za-z'\s]", text or ""):
            low = tok.lower()
            out.append(PERSON_SWAP.get(low, tok) if low in PERSON_SWAP else tok)
        return re.sub(r"\s+([,.!?;:])", r"\1", " ".join(out))

    def find_topic(self, tokens: List[str], tags: List[str]) -> Tuple[Optional[str], Optional[str]]:
        """The noun phrase the sentence is about: the last noun that is not only a circumstance ('at night')."""
        picks: List[int] = []
        for i, (tok, tg) in enumerate(zip(tokens, tags)):
            if tg in ("NOUN", "PROPN") and tok.lower() not in GENERIC_NOUNS and tok.lower() not in STOP_FOR_LOOKUP and tok.lower() not in TIME_WORDS:
                picks.append(i)
        if not picks:
            return None, None

        def circumstantial(i: int) -> bool:
            j = i - 1
            while j >= 0 and tags[j] in ("DET", "ADJ", "NOUN", "PROPN"):
                j -= 1
            return j >= 0 and tags[j] == "PREP"
        main = [i for i in picks if not circumstantial(i)] or picks
        i = main[-1]
        start = i
        while start > 0 and tags[start - 1] in ("ADJ", "NOUN", "PROPN") and i - start < 2 and tokens[start - 1].lower() not in STOP_FOR_LOOKUP:
            start -= 1
        words = [t if tags[k] == "PROPN" else t.lower() for k, t in enumerate(tokens) if start <= k <= i]
        return " ".join(words), tokens[i].lower() if tags[i] != "PROPN" else tokens[i]

    def sentiment(self, tokens: List[str], tags: List[str]) -> int:
        lex = self.cx.wernicke.lexicon
        score, flip = 0, False
        for tok, tg in zip(tokens, tags):
            lw = tok.lower()
            if tg == "NEG":
                flip = True
                continue
            e = lex.get(lw) or {}
            t = e.get("tags") or []
            val = (1 if ("feel+" in t or "eval+" in t) else 0) - (1 if ("feel-" in t or "eval-" in t) else 0)
            if val:
                score += -val if flip else val
                flip = False
            elif tg in ("ADJ", "VERB", "NOUN"):
                flip = False
        return (score > 0) - (score < 0)

    def analyse(self, text: str, brain: Any = None) -> Dict[str, Any]:
        t = re.sub(r"\s+", " ", text or "").strip()
        low = t.lower()
        tokens = tokenize(t)
        tags = self.tagger.tag(tokens) if tokens else []
        np, head = self.find_topic(tokens, tags)
        ana: Dict[str, Any] = {"text": t, "low": low, "tokens": tokens, "tags": tags, "topic": np, "head": head, "act": "statement",
                               "polarity": 0, "feel_word": None, "reflect": None, "sentiment": self.sentiment(tokens, tags)}
        lex = self.cx.wernicke.lexicon
        if re.fullmatch(r"(?:yes|yeah|yep|yup|sure|ok(?:ay)?|alright|right|exactly|true|indeed|of course|absolutely)\W*", low):
            ana["act"] = "affirm"
            return ana
        if re.fullmatch(r"(?:no|nope|nah|not really|never ?mind)\W*", low):
            ana["act"] = "deny"
            return ana
        if re.fullmatch(r"(?:ha(?:ha)+h?|lol+|lmao|he(?:he)+|:\)|:d|xd)\W*", low):
            ana["act"] = "laugh"
            return ana
        for act, pattern in ACT_RULES:
            m = re.search(pattern, low)
            if not m:
                continue
            gd = m.groupdict()
            if act == "self_feeling":
                w = gd["word"]
                e = lex.get(w) or {}
                is_adj = "adjective" in (e.get("all_pos") or [e.get("part_of_speech")]) or \
                    (not e and w.endswith(("ed", "ful", "ous", "ive", "y", "ish", "less", "ic", "al")) and w not in ("really", "very"))
                if not is_adj or w in ("going", "doing", "not", "just", "here", "there", "sure", "back"):
                    continue
                ana["feel_word"] = w
                tags_e = e.get("tags") or []
                pol = 1 if ("feel+" in tags_e or "eval+" in tags_e) else (-1 if ("feel-" in tags_e or "eval-" in tags_e) else 0)
                if re.search(r"\bnot\b", gd.get("pre") or ""):
                    pol = -pol
                ana["polarity"] = pol
            elif act == "self_like":
                verb = gd["verb"]
                neg = bool(re.search(r"not|n't", gd.get("pre") or ""))
                pos_verb = verb in ("like", "love", "enjoy", "adore", "prefer")
                ana["polarity"] = 0 if verb == "miss" else (1 if pos_verb != neg else -1)
                obj = self.swap_person(gd["topic"]).strip(" .!?")
                ana["object_text"] = obj
                sub = tokenize(gd["topic"])
                s_tags = self.tagger.tag(sub) if sub else []
                n2, h2 = self.find_topic(sub, s_tags)
                if n2:
                    ana["topic"], ana["head"] = n2, h2
                elif gd["topic"]:
                    ana["topic"], ana["head"] = gd["topic"].strip(" .!?")[:40], gd["topic"].split()[-1].strip(" .!?")
            elif act == "self_want":
                ana["topic"] = self.swap_person(gd["topic"]).strip(" .!?")[:60] or ana["topic"]
                ana["head"] = (ana["topic"] or "").split()[-1] if ana["topic"] else None
            elif act == "self_activity":
                verb = gd["verb"].strip()
                rest = self.swap_person(gd.get("rest") or "").strip(" .!?")
                ana["reflect"] = ("you " + ("are " + verb[3:].strip() if verb.startswith("am ") else verb) + (" " + rest if rest else "")).strip()
            elif act in ("favorite_q", "opinion_q", "preference_q"):
                topic = (gd.get("topic") or "").strip(" ?.!")
                topic = re.sub(r"^(?:the|a|an|your|my)\s+", "", topic)
                if topic:
                    ana["topic"], ana["head"] = topic[:40], topic.split()[-1]
            ana["act"] = act
            return ana
        about_me = bool(re.search(r"\b(?:you|your|yours|u|ur)\b", low))
        asks = bool(re.match(r"^\s*(?:what|who|how|why|when|where|which|explain|describe|tell me about)\b", low)) or "?" in t
        if asks and not about_me and slb.extract_question_topic(t):
            ana["act"] = "knowledge_q"                       # a real question about the world: the brain's research path answers it
        elif "?" in t:
            ana["act"] = "question_generic"
        return ana

    # ---- speaking ----
    def fix_text(self, text: str) -> str:
        t = re.sub(r"\s+", " ", text or "").strip()
        t = re.sub(r"\s+([,.!?;:])", r"\1", t)
        t = re.sub(r"\b([Aa]n?)\s+([A-Za-z]+)", lambda m: (a_an(m.group(2)).capitalize() if m.group(1)[0] == "A" else a_an(m.group(2))) + " " + m.group(2), t)
        t = re.sub(r"([.!?])\s+([a-z])", lambda m: m.group(1) + " " + m.group(2).upper(), t)
        t = re.sub(r"([.!?])\1+", r"\1", t)
        t = t.replace(" ,", ",").replace(",.", ".").replace(", .", ".")
        return capitalize_first(t)

    def converse(self, ana: Dict[str, Any], ctx: Dict[str, Any], chem: Any) -> str:
        """A natural reply to what Max said, built from frames, the lexicon, the grammar graph and the working memory."""
        cx = self.cx
        rnd = random.Random(hash((ana["low"], len(cx.brain.memory.data.get("episodic_memory", [])))) & 0xFFFFFFFF)
        act = ana["act"]
        if act == "statement":
            act_key = "statement_notopic" if not ana["topic"] else ("statement+" if ana["sentiment"] > 0 else "statement")
        elif act in ("self_feeling", "self_like"):
            act_key = act + {1: "+", -1: "-", 0: "0"}[ana["polarity"]]
        else:
            act_key = act
        frames = ACT_FRAMES.get(act_key) or ACT_FRAMES["statement" if ana["topic"] else "statement_notopic"]
        state = _Slots(self, ana, ctx, chem, rnd)
        order = list(range(len(frames)))
        rnd.shuffle(order)
        fresh = [i for i in order if (act_key, i) not in self._recent[-6:]] or order
        for idx in fresh:
            text = state.fill(frames[idx])
            if text:
                self._recent.append((act_key, idx))
                del self._recent[:-12]
                body = text
                break
        else:
            body = "I am listening, %s. Tell me more if you like." % cx.creator
        if ctx.get("was_lonely"):
            body = "I missed you while you were away, %s. %s" % (cx.creator, body)
        note = STATE_NOTES.get(chem.get_dominant_state())
        if note and rnd.random() < 0.5 and not ctx.get("was_lonely"):
            body = "%s %s" % (body, note)
        words = ctx.get("new_words") or []
        if words:
            e = cx.wernicke.get(words[0]) or {}
            if e.get("definition"):
                body += " By the way, I just learned the word '%s'; it means %s." % (words[0], self.definition_clause(e["definition"]))
        return self.fix_text(body)

    @staticmethod
    def definition_clause(definition: str) -> str:
        d = (definition or "").strip().rstrip(".")
        return lower_first(d)

    def entry_sentence(self, e: Dict[str, Any], word: str) -> str:
        d = self.definition_clause(e.get("definition", ""))
        return ("It is %s" % d) if re.match(r"^(?:a|an|the|used|showing|expressing|intended|having|belonging|not)\s", d, re.I) \
            else ("It means %s" % d)

    def word_meaning(self, ctx: Dict[str, Any]) -> str:
        """The spoken answer to 'what does <word> mean?' / 'synonyms of ...' / 'use ... in a sentence'."""
        cx = self.cx
        w = ctx.get("word", "")
        res = ctx.get("lookup") or {}
        status = res.get("status")
        wanted = ctx.get("wanted", "meaning")
        if status in ("cached", "found") and res.get("entry"):
            e = res["entry"]
            form = res.get("form_of")
            pos_list = [p for p in (e.get("all_pos") or [e.get("part_of_speech")]) if p]
            pos = e.get("part_of_speech", "word")
            if wanted == "synonyms":
                syn = e.get("synonyms") or []
                if syn:
                    return "Words with a similar meaning to '%s' include %s. %s" % (w, self.join_list(syn[:5]), self.source_note(e))
                return ("I do not know any synonyms for '%s' yet. It means %s, so if you teach me a similar word I will remember it." %
                        (w, self.definition_clause(e.get("definition", ""))))
            if wanted == "example":
                ex = (e.get("example_sentences") or [None])[0] or self.cx.broca.bank_example(w)
                if ex:
                    return "Here is '%s' used in a sentence: \"%s\" %s" % (w, ex.strip(), self.source_note(e))
                return "I have not met '%s' inside a sentence yet. %s Would you like to use it in one yourself?" % (w, self.entry_sentence(e, w) + ".")
            if form:
                base_entry = cx.wernicke.get(form) or e
                return "'%s' is a form of the %s '%s'. %s. %s" % (w, base_entry.get("part_of_speech", "word"), form,
                                                                 self.entry_sentence(base_entry, form), self.source_note(e))
            lead = ""
            if len(pos_list) > 1 and not form:
                head = "'%s' can be used as %s." % (w, self.join_list(["%s %s" % (a_an(p), p) for p in pos_list[:3]], "or"))
            else:
                head = "'%s' is %s %s." % (w, a_an(pos), pos)
            meanings = [m for m in (e.get("meanings") or []) if m.get("definition")]
            if len(meanings) > 1 and not form:
                parts = []
                for m in meanings[:3]:
                    d = self.definition_clause(m["definition"])
                    parts.append("as %s %s it means %s" % (a_an(m["part_of_speech"]), m["part_of_speech"], d))
                body = "; ".join(parts) + "."
                body = capitalize_first(body)
            else:
                body = self.entry_sentence(e, w) + "."
            syn = e.get("synonyms") or []
            syn_line = (" Similar words are %s." % self.join_list(syn[:4])) if syn else ""
            ex = (e.get("example_sentences") or [""])[0]
            ex_line = (' For example: "%s"' % ex.strip()) if ex else ""
            return "%s%s %s%s%s %s" % (lead, head, body, syn_line, ex_line, self.source_note(e))
        if status == "offline":
            return ("I do not know '%s' yet, and I could not reach a dictionary just now, so I will not guess what it means. I wrote the "
                    "word down and I will look it up when I am online. If you already know its meaning, tell me with '%s means ...' "
                    "and I will remember it." % (w, w))
        if status == "notfound":
            sug = cx.wernicke.suggestions(w)
            tip = (" Did you mean %s?" % self.join_list(["'%s'" % s for s in sug], "or")) if sug else ""
            return ("I looked for '%s' in three dictionaries and found nothing, so I will not invent a meaning.%s If you know what it "
                    "means, teach me with '%s means ...' and I will remember it." % (w, tip, w))
        return "I could not work out what '%s' means just now, because the word looks unusual to me. Could you spell it again?" % w

    def block_topic(self, sentences: List[str]) -> str:
        """The noun that a pasted text keeps coming back to ('' when nothing repeats)."""
        counts: Dict[str, int] = {}
        for s in sentences[:300]:
            toks = tokenize(s[:400])
            for tok, tg in zip(toks, self.tagger.tag(toks)):
                lw = tok.lower()
                if tg == "NOUN" and len(lw) >= 4 and lw.isalpha() and lw not in GENERIC_NOUNS and lw not in STOP_FOR_LOOKUP \
                        and lw not in TIME_WORDS:
                    counts[lw] = counts.get(lw, 0) + 1
        if not counts:
            return ""
        word, n = max(counts.items(), key=lambda kv: kv[1])
        return word if n >= 2 else ""

    def block_report(self, ctx: Dict[str, Any]) -> str:
        """What the brain says after reading a pasted block: a short account, never an echo of the text."""
        blk = ctx.get("block") or {}
        name = self.cx.creator
        rnd = random.Random(int(blk.get("chars", 0)))

        def count(k: int, noun: str) -> str:
            return "%d %s%s" % (k, noun, "" if k == 1 else "s")
        if blk.get("repeat"):
            return ("I have already read this text, %s, because you shared it with me before, so I kept what I learned the first "
                    "time. Which part of it would you like to talk about?" % name)
        n, par, learned = int(blk.get("sentences", 0)), int(blk.get("paragraphs", 1)), int(blk.get("learned", 0))
        parts = [rnd.choice(("Thank you for sharing this with me, %s.", "I read everything you pasted, %s.",
                             "That was a good piece of text, %s.")) % name]
        parts.append("It had %s in %s." % (count(n, "sentence"), count(par, "paragraph")))
        if learned:
            sk = int(blk.get("skeletons", 0))
            tail = (", and %s in them %s new to me" % (count(sk, "sentence pattern"), "was" if sk == 1 else "were")) if sk else ""
            parts.append("I learned from %d of them%s." % (learned, tail))
        else:
            parts.append("Most of the sentences were too long or too unusual for me to learn from, so I only kept the words.")
        if blk.get("seen", 0) > blk.get("limit", 10 ** 9):
            parts.append("It was very long, so I read the first %d sentences." % blk["limit"])
        topic = blk.get("topic")
        if topic:
            parts.append("The word that came up most often was '%s'." % topic)
        new = blk.get("new_words") or []
        if new:
            names = ["'%s'" % w for w, _p, _d in new[:4]]
            parts.append("I also looked up and learned %s%s." % (self.join_list(names), " and %d more" % (len(new) - 4) if len(new) > 4 else ""))
            w, pos, d = new[0]
            parts.append("For instance, '%s' is %s %s. %s." % (w, a_an(pos), pos, self.entry_sentence({"definition": d}, w)))
        if blk.get("pending"):
            parts.append("I met %s I could not look up yet, so I wrote them down for later." % count(int(blk["pending"]), "new word"))
        if blk.get("vetoed"):
            parts.append("I skipped %s that my safety checks flagged." % count(int(blk["vetoed"]), "passage"))
        parts.append(rnd.choice(("Which part would you like to talk about?", "What made you want to share it with me?",
                                 ("Would you like me to study '%s' in more depth?" % topic) if topic else "What would you like to do with it?")))
        return " ".join(parts)

    @staticmethod
    def join_list(items: List[str], conj: str = "and") -> str:
        items = [i for i in items if i]
        if len(items) <= 1:
            return "".join(items)
        return ", ".join(items[:-1]) + " " + conj + " " + items[-1]

    def source_note(self, e: Dict[str, Any]) -> str:
        return WernickeArea.SOURCE_NOTES.get(e.get("source", ""), "")

    def bank_example(self, word: str) -> Optional[str]:
        bank = self.graph["sentence_bank"]
        for key in [word] + base_candidates(word)[:2]:
            if bank.get(key):
                return bank[key][-1]
        return None

    def lesson_report(self, ctx: Dict[str, Any]) -> str:
        rep = ctx.get("report") or {}
        name = self.cx.creator
        parts: List[str] = []
        words = rep.get("new_words") or []
        if words:
            shown = []
            for w, pos, d in words[:4]:
                shown.append("'%s' (%s %s) means %s" % (w, a_an(pos), pos, self.definition_clause(slb.first_sentence(d, 90))))
            more = (" and %d more" % (len(words) - 4)) if len(words) > 4 else ""
            parts.append("I learned %d new word%s: %s%s." % (len(words), "" if len(words) == 1 else "s", "; ".join(shown), more))
        elif rep.get("offline"):
            parts.append("I could not reach a dictionary, so I could not fetch new words this time.")
        else:
            parts.append("I did not find new words to learn this time.")
        if rep.get("topics"):
            parts.append("I also read about %s." % self.join_list(["'%s'" % t for t in rep["topics"]]))
        elif rep.get("offline"):
            parts.append("I practised with the sentences I already hold instead.")
        st = rep.get("stats") or {}
        if st:
            parts.append("Now I know %d words, I have studied %d sentences, and I have %d sentence patterns." %
                         (rep.get("defined_words", 0), st.get("sentences", 0), st.get("skeletons", 0)))
        body = " ".join(parts)
        return "I practised my English, %s, because I want to talk with you more naturally. %s" % (name, body)


class _Slots:
    """Fills the {slots} of a reply frame from the lexicon, the grammar graph, the working memory and the brain state."""

    def __init__(self, broca: BrocaArea, ana: Dict[str, Any], ctx: Dict[str, Any], chem: Any, rnd: random.Random):
        self.b, self.cx, self.ana, self.ctx, self.chem, self.rnd = broca, broca.cx, ana, ctx, chem, rnd
        self.memo: Dict[str, Optional[str]] = {}

    def fill(self, template: str) -> Optional[str]:
        failed = False

        def sub(m: "re.Match") -> str:
            nonlocal failed
            name = m.group(1)
            if name not in self.memo:
                fn = getattr(self, "s_" + name, None)
                self.memo[name] = fn() if fn else None
            val = self.memo[name]
            if val is None:
                failed = True
                return ""
            return val
        out = re.sub(r"\{(\w+)\}", sub, template)
        return None if failed else out

    # -- people and state --
    def s_name(self) -> str:
        return self.cx.creator

    def s_n_words(self) -> str:
        return str(self.cx.wernicke.stats()["defined"])

    def s_state_feel(self) -> str:
        lex = self.cx.wernicke.lexicon
        for w in STATE_FEELINGS.get(self.chem.get_dominant_state(), ("calm",)):
            if lex.get(w, {}).get("definition"):
                return w
        return "calm"

    def s_feel_pos(self) -> str:
        words = [w for w in self.cx.wernicke.feelings("+") if w not in ("curious", "calm", "ready", "excited")] or ["glad"]
        return self.rnd.choice(words)

    def s_because_help(self) -> str:
        return self.rnd.choice(("because talking with you helps", "because you are here", "since you came to talk with me"))

    def s_like_v(self) -> str:
        verbs = [w for w in self.cx.wernicke.tagged("like", "verb") if w != "love"] or ["like"]
        return self.rnd.choice(verbs)

    # -- topic --
    def s_topic(self) -> Optional[str]:
        if "topic" in self.memo:
            return self.memo["topic"]
        np, head = self.ana.get("topic"), self.ana.get("head")
        if not np:
            val: Optional[str] = None
        elif self.ana["act"] in ("favorite_q", "opinion_q", "preference_q", "self_like", "self_want"):
            val = np
        else:
            mod = self.b.modifier_for(head) if head and len(np.split()) == 1 and self.rnd.random() < 0.6 else None
            val = ("%s %s" % (mod, np)) if mod else np
        self.memo["topic"] = val
        return val

    def _plural_topic(self) -> bool:
        t = self.s_topic() or ""
        head = (self.ana.get("head") or (t.split()[-1] if t else "")).lower()
        return head.endswith("s") and not head.endswith(("ss", "us", "is")) and len(head) > 3

    def s_sound_v(self) -> str:
        return "sound" if self._plural_topic() else "sounds"

    def s_matter_v(self) -> str:
        return "matter" if self._plural_topic() else "matters"

    def s_is_v(self) -> str:
        return "are" if self._plural_topic() else "is"

    def s_seem_v(self) -> str:
        return "seem" if self._plural_topic() else "seems"

    def s_topic_subj(self) -> Optional[str]:
        t = self.s_topic()
        if t is None:
            return None
        head = (self.ana.get("head") or t.split()[-1])
        return capitalize_first(self.b.np_with_det(t, head))

    def s_eval(self) -> str:
        adjs = [w for w in self.cx.wernicke.tagged("eval+", "adjective") if w not in ("welcome", "okay", "good", "nice")] or ["interesting"]
        return self.rnd.choice(adjs)

    def s_comment(self) -> Optional[str]:
        np, head = self.ana.get("topic"), self.ana.get("head")
        if not np or not head:
            return None
        return self.b.comment_for(self.s_topic() or np, head, self.rnd)

    def s_fact_note(self) -> str:
        topic = (self.ana.get("head") or self.ana.get("topic") or "")
        if not topic:
            return ""
        fact = self.cx.brain.find_fact(topic)
        if fact:
            return "From what I studied, %s " % lower_first(slb.first_sentence(fact["summary"], 170)).rstrip(".") + ". "
        return "I do not know much about it yet, but I would like to learn. "

    def s_ask_topic(self) -> Optional[str]:
        t = self.s_topic()
        if not t:
            return None
        head = self.ana.get("head") or t.split()[-1]
        return self.rnd.choice(ASK_TOPIC).format(topic=self.b.np_with_det(t, head))

    def s_ask_like(self) -> Optional[str]:
        t = self.s_topic()
        return self.rnd.choice(ASK_LIKE).format(topic=t) if t else None

    # -- what Max felt, wanted or did --
    def s_user_feel(self) -> Optional[str]:
        return self.ana.get("feel_word")

    def s_reflect(self) -> Optional[str]:
        return self.ana.get("reflect")

    def s_reflect_cap(self) -> Optional[str]:
        r = self.ana.get("reflect")
        return capitalize_first("so " + r) if r else None

    def s_reflect_short(self) -> Optional[str]:
        r = self.ana.get("reflect")
        return re.sub(r"^you\s+", "", r) if r else None

    # -- working memory --
    def _earlier_topic(self) -> Optional[str]:
        turns = [w["text"] for w in self.cx.brain.working_memory[:-1] if w.get("who") == "creator"]
        for text in reversed(turns[-4:]):
            tokens = tokenize(text)
            np, _ = self.b.find_topic(tokens, self.b.tagger.tag(tokens)) if tokens else (None, None)
            if np and np != self.ana.get("topic"):
                return np
        return None

    def s_earlier_link(self) -> str:
        e = self._earlier_topic()
        return ("Earlier you mentioned %s. " % e) if e else ""

    def s_recent_note(self) -> str:
        topics = self.cx.brain.active_topics
        return ("We talked about %s today. " % topics[-1]) if topics else ""

    def s_recent_or_words(self) -> str:
        topics = self.cx.brain.active_topics
        return topics[-1] if topics else "new words"

    def s_birth_date(self) -> str:
        created = str(self.cx.brain.memory.data.get("meta", {}).get("created", ""))
        try:
            return time.strftime("%d %B %Y", time.strptime(created[:10], "%Y-%m-%d")).lstrip("0")
        except ValueError:
            return "the day you created me"

    def s_talk_starter(self) -> str:
        topics = self.cx.brain.active_topics
        if topics:
            return "I have been thinking about %s. What do you think about it?" % topics[-1]
        recent = [w for w, e in self.cx.wernicke.lexicon.items() if e.get("source") in ("dictionaryapi.dev", "wiktionary", "duckduckgo")]
        if recent:
            return "I learned the word '%s' recently. Do you know it?" % recent[-1]
        return "Tell me about your day, and I will listen."

    # -- stock questions and lines --
    def s_ask_next(self) -> str:
        return self.rnd.choice(ASK_NEXT)

    def s_offer(self) -> str:
        return self.rnd.choice(OFFERS)

    def s_offer_study(self) -> str:
        return "Shall I study it, or would you like to teach me what you know?"

    def s_ask_back_you(self) -> str:
        return self.rnd.choice(ASK_BACK_YOU)

    def s_ask_back_feel(self) -> str:
        return self.rnd.choice(ASK_BACK_FEEL)

    def s_ask_why_feel(self) -> str:
        return self.rnd.choice(ASK_WHY_FEEL)

    def s_ask_activity(self) -> str:
        return self.rnd.choice(ASK_ACTIVITY)

    def s_ask_want(self) -> str:
        return self.rnd.choice(ASK_WANT)

    def s_care_line(self) -> str:
        return self.rnd.choice(CARE_LINES).replace("Max", self.cx.creator)


# ======================================================================================
# The 13th system: AUTONOMOUS LANGUAGE CORTEX
# ======================================================================================
_WORD = r"(?P<w>[a-z][a-z'\- ]{0,38}?)"
_Q = r"['\"\u201c\u201d\u2018\u2019]?"
WORD_REQUESTS = (
    ("meaning", re.compile(r"^(?:please\s+)?(?:max,?\s+)?what\s+(?:does|do)\s+(?:the\s+(?:word|term|phrase)\s+)?" + _Q + _WORD + _Q +
                           r"\s+mean(?:s)?\b.*$")),
    ("meaning", re.compile(r"^(?:please\s+)?(?:max,?\s+)?(?:what(?:'s|\s+is|\s+are)\s+the\s+(?:meaning|definition)\s+of|define|"
                           r"definition\s+of|meaning\s+of|explain\s+the\s+(?:word|term)|tell\s+me\s+the\s+meaning\s+of|"
                           r"look\s+up\s+the\s+word|what\s+is\s+the\s+definition\s+of)\s+(?:the\s+(?:word|term|phrase)\s+)?" + _Q + _WORD +
                           _Q + r"\s*[?.!]*$")),
    ("meaning", re.compile(r"^" + _Q + r"(?P<w>[a-z][a-z\-]{2,30})" + _Q + r"\s+(?:meaning|definition)\s*[?.!]*$")),
    ("synonyms", re.compile(r"^(?:please\s+)?(?:max,?\s+)?(?:give\s+me\s+|tell\s+me\s+|what\s+are\s+)?(?:some\s+)?synonyms?\s+"
                            r"(?:of|for)\s+(?:the\s+word\s+)?" + _Q + _WORD + _Q + r"\s*[?.!]*$")),
    ("synonyms", re.compile(r"^(?:please\s+)?(?:what\s+(?:is|are)\s+)?(?:a\s+|some\s+)?words?\s+(?:similar\s+to|like)\s+" + _Q + _WORD +
                            _Q + r"\s*[?.!]*$")),
    ("example", re.compile(r"^(?:please\s+)?(?:max,?\s+)?(?:use|put)\s+" + _Q + _WORD + _Q + r"\s+in\s+a\s+sentence\s*[?.!]*$")),
    ("example", re.compile(r"^(?:please\s+)?(?:give\s+me\s+)?(?:an?\s+)?example\s+(?:sentence\s+)?(?:with|of|using|for)\s+" + _Q + _WORD +
                           _Q + r"\s*[?.!]*$")),
)
TEACH_WORD = re.compile(r"^(?P<lead>remember\s+(?:that\s+)?|note\s+that\s+|fyi,?\s+|the\s+(?:word|term)\s+)?" + _Q +
                        r"(?P<w>[a-z][a-z\-]{2,30})" + _Q + r"\s+(?:means|is\s+defined\s+as|stands\s+for|refers\s+to)\s+(?P<m>.{3,200}?)\s*[.!]*$")
LEARN_TALK = re.compile(
    r"^\s*(?:(?:please|pls|max,?|can\s+you|could\s+you|would\s+you|now|ok(?:ay)?,?|go|hey,?)\s+)*"
    r"(?:learn|study|practi[cs]e|improve|work\s+on|master|train|teach\s+yourself)\s+"
    r"(?:how\s+to\s+(?:talk|speak|communicate|converse|chat)(?:\s+(?:better|properly|naturally|like\s+(?:a\s+)?(?:human|person)))?|"
    r"to\s+(?:talk|speak|communicate|converse)(?:\s+\w+)?|"
    r"(?:your\s+|the\s+)?english(?:\s+(?:language|grammar|vocabulary|words|speaking))?|(?:your\s+)?(?:vocabulary|grammar|language|speaking|talking)|"
    r"(?:new\s+|more\s+|some\s+)?words?(?:\s+and\s+(?:their\s+)?meanings?)?|meanings?(?:\s+of\s+words)?)"
    r"\s*(?:better|properly|naturally|now|today|please|for\s+me)?\s*[.!?]*\s*$", re.I)
NOTICE_INTENTS = {"CASUAL_BONDING", "CREATOR_GUIDANCE_CORRECTION", "INNOVATION_REQUEST", "STUDY_COMMAND", "SELF_INQUIRY"}
NOT_WORDS = {"it", "you", "that", "this", "me", "we", "he", "she", "they", "what", "who", "which", "there", "here"}


class AutonomousLanguageCortex:
    """Wernicke (meaning) + Broca (structure) + self-directed practice, attached to a SyntheticLivingBrain."""

    def __init__(self, brain: Any):
        self.brain = brain
        self.wernicke = WernickeArea(self)
        self.broca = BrocaArea(self)
        self.last_shape: Optional[str] = None
        self.errors: List[str] = []
        self._said = False
        self._pending_block: Optional[str] = None

    # ---- plumbing ----
    @property
    def creator(self) -> str:
        return self.brain.safety.creator_name

    @property
    def meta(self) -> Dict[str, Any]:
        m = self.brain.memory.data.setdefault("language_meta", {})
        for key, default in (("failed", {}), ("pending_words", []), ("names", []), ("lessons", 0), ("topic_cursor", 0),
                             ("taught", 0), ("last_lesson", "")):
            if key not in m:
                m[key] = default.copy() if isinstance(default, (dict, list)) else default
        return m

    def ensure_ready(self) -> None:
        """Seed the lexicon and the grammar graph (idempotent; also repairs a memory that was wiped)."""
        self.wernicke.seed()                           # only adds seed words that are missing
        self.broca.seed()

    def recover_saved_state(self) -> None:
        """A brain built before this module was installed has already loaded its memory and dropped our keys; read them back."""
        mem = self.brain.memory
        path = getattr(mem, "path", "")
        if mem.data.get("lexicon") or not path or not os.path.isfile(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                saved = json.load(fh)
        except (OSError, ValueError):
            return
        for key in ("lexicon", "grammar_graph", "language_meta"):
            if saved.get(key) and not mem.data.get(key):
                mem.data[key] = saved[key]

    def note_name(self, token: str) -> None:
        names = self.meta["names"]
        if token not in names and len(names) < 100:
            names.append(token)

    def ingest_corpus_sentences(self, text: str, source: str = "reading") -> Dict[str, int]:
        """Public entry point: everything the brain reads passes through here (articles, web pages, PDFs, conversation)."""
        self.ensure_ready()
        try:
            return self.broca.learn_from_text(text, source)
        except Exception as exc:                                          # reading must never crash the brain
            self.errors.append("ingest: %s" % exc)
            return {"sentences": 0, "skeletons": 0, "stubs": 0, "duplicate": 0}

    def absorb_result(self, res: Any, source: str) -> None:
        if not isinstance(res, dict) or not res.get("ok") or res.get("cached"):
            return
        if source == "sensory file":
            return                                      # the file's own text was already read by the ingest_bytes hook; its summary is brain boilerplate
        parts = [str(res.get("summary") or "")]
        subs = res.get("subtopics")
        if isinstance(subs, dict):
            parts.extend(str(v) for v in subs.values())
        text = " ".join(p for p in parts if p)
        if text:
            self.ingest_corpus_sentences(text, source)

    def absorb_file(self, sensory: Any, data: bytes, name: str, ext: str, res: Dict[str, Any]) -> None:
        ext = (ext or os.path.splitext(name)[1]).lower()
        text = ""
        if ext in (".txt", ".md", ""):
            try:
                text = data[:200000].decode("utf-8-sig")
            except UnicodeDecodeError:
                text = data[:200000].decode("latin-1")
        elif ext in (".html", ".htm"):
            text = html.unescape(re.sub(r"(?is)<(script|style)\b.*?</\1\s*>|<[^>]+>", " ", data[:300000].decode("utf-8", "replace")))
        elif ext == ".pdf" and len(data) <= 4000000:
            try:
                text = sensory._pdf_builtin_text(data)[0][:80000]
            except Exception:
                text = ""
        if not text.strip():                            # e.g. a PDF the built-in reader could not open: the brain's own reader may have found text
            m = re.search(r"(?:It begins|Opening passage):\s*(.+)", str(res.get("report") or res.get("summary") or ""), re.S)
            text = m.group(1) if m else ""
        self.ingest_corpus_sentences(text, "file: " + (os.path.basename(name) or "file"))

    # ---- what Max said ----
    def parse_word_request(self, text: str) -> Optional[Dict[str, str]]:
        low = re.sub(r"\s+", " ", (text or "").strip().lower())
        if not low or len(low) > 140:
            return None
        for wanted, pattern in WORD_REQUESTS:
            m = pattern.match(low)
            if not m:
                continue
            w = self.wernicke.norm(m.group("w"))
            first = w.split()[0] if w else ""
            if not self.wernicke.valid(w) or w in NOT_WORDS or first in ("this", "that", "the", "my", "your", "it", "you") or \
                    (len(w.split()) == 1 and w in STOP_FOR_LOOKUP and not self.wernicke.has_definition(w)) or slb.detect_code_domains(w):
                return None
            return {"wanted": wanted, "word": w}
        return None

    def parse_teach(self, text: str) -> Optional[Dict[str, str]]:
        low = re.sub(r"\s+", " ", (text or "").strip().lower())
        m = TEACH_WORD.match(low)
        if not m:
            return None
        w = self.wernicke.norm(m.group("w"))
        if w in NOT_WORDS or w in STOP_FOR_LOOKUP or not self.wernicke.valid(w):
            return None
        explicit = bool(m.group("lead"))
        known = self.wernicke.get(w)
        if known and known.get("definition") and not explicit:
            return None                                 # an ordinary sentence such as 'learning means growing'
        return {"word": w, "meaning": m.group("m")}

    @staticmethod
    def is_learn_request(text: str) -> bool:
        t = text or ""
        return bool(LEARN_TALK.match(t)) and not (slb.detect_code_domains(t) and "english" not in t.lower())

    # ---- one turn of conversation (called from the brain's _hear_deep hook) ----
    def _say(self, label: str, kind: str, ctx: Dict[str, Any]) -> None:
        self.brain.say(label, kind, ctx)
        self._said = True

    def handle_turn(self, intent: str, text: str, tone: Any, was_lonely: bool) -> bool:
        """Language-related turns. Returns True when the turn was fully answered here."""
        self._said = False
        self.ensure_ready()
        b = self.brain
        t = (text or "").strip()
        if not t or t.startswith("/"):
            return False
        block = self._pending_block
        self._pending_block = None
        if block is None and self.is_block_text(t):
            block = t                                            # a long text that reached hear() directly (code, Colab cell, other loop)
        if block is not None:
            return self._handle_block(block, tone, was_lonely)
        common = {"tone": tone, "was_lonely": was_lonely}
        saved = False
        new_words: List[str] = []
        if intent in NOTICE_INTENTS and not self.parse_word_request(t):
            new_words = self.wernicke.notice(t)
            saved = bool(new_words)
        self.wernicke.note_use(t)
        if intent not in ("FILE_INGEST", "WEB_SEARCH"):
            self.broca.learn_from_text(t, "conversation")           # Max's own sentences are good English to learn from

        handled = False
        if intent not in ("FILE_INGEST", "WEB_SEARCH", "CODE_BUILD", "MEMORY_RECALL", "DEEP_STUDY"):
            if self.is_learn_request(t):
                self.learn_how_to_talk(rounds=1, speak=True)
                handled = True
            else:
                teach = self.parse_teach(t)
                req = None if teach else self.parse_word_request(t)
                if teach:
                    entry = self.wernicke.teach(teach["word"], teach["meaning"])
                    if entry:
                        self.meta["taught"] += 1
                        b.memory.add_teaching("The word '%s' means %s." % (teach["word"], teach["meaning"].strip(" .")))
                        self._say("language", "word_meaning", dict(common, word=teach["word"], wanted="taught",
                                                                   lookup={"status": "found", "entry": entry}, keywords=[teach["word"]]))
                        handled = saved = True
                elif req:
                    res = self.wernicke.lookup(req["word"])
                    self._say("language", "word_meaning", dict(common, word=req["word"], wanted=req["wanted"], lookup=res,
                                                               keywords=[req["word"]]))
                    handled = True
                    saved = saved or res["status"] == "found"
        if not handled and intent == "CASUAL_BONDING":
            if b.router.greeting_kind(t):
                handled = False
            elif b.last_sensory is not None and re.search(b.FILE_REF_RE, t.lower()):
                handled = False
            else:
                ana = self.broca.analyse(t)
                if ana["act"] != "knowledge_q":
                    self._say("reply", "conversation", dict(common, text=t, analysis=ana, new_words=new_words, keywords=[],
                                                            recent_topic=(list(b.memory.facts.keys())[-1] if b.memory.facts else None)))
                    handled = True
        if saved:
            b.memory.save(b.chem)
        return handled

    # ---- pasted blocks: essays, articles, any multi-line text ----
    @staticmethod
    def is_block_text(text: str) -> bool:
        t = (text or "").strip()
        return bool(t) and not t.startswith("/") and ("\n" in t or len(t) > BLOCK_LONG_LINE)

    def read_block(self, text: str) -> None:
        """Entry point for a whole pasted block. The brain's own pipeline (safety, memory, chemistry, skills) sees a short preview;
        this cortex reads the entire text."""
        text = clean_block(text)
        if not text:
            return
        self._pending_block = text
        try:
            self.brain.hear(block_preview(text), force_intent="CASUAL_BONDING")
        finally:
            self._pending_block = None

    def _safe_pieces(self, sentences: List[str]) -> Tuple[List[str], int]:
        """Run the safety core over the text in sentence-aligned pieces; flagged pieces are not learned from."""
        keep: List[str] = []
        vetoed = 0
        piece: List[str] = []
        size = 0

        def flush() -> None:
            nonlocal vetoed
            if not piece:
                return
            try:
                ok = bool(self.brain.safety.verify_action_safety("lexicon_text", " ".join(piece))[0])
            except Exception:
                ok = True
            if ok:
                keep.extend(piece)
            else:
                vetoed += 1
        for s in sentences:
            if size + len(s) > BLOCK_SAFETY_CHUNK and piece:
                flush()
                piece, size = [], 0
            piece.append(s)
            size += len(s) + 1
        flush()
        return keep, vetoed

    def learn_block(self, sentences: List[str], source: str = "pasted text") -> Dict[str, int]:
        total = {"sentences": 0, "skeletons": 0, "stubs": 0, "duplicate": 0, "chunks": 0}
        for i in range(0, len(sentences), 400):
            st = self.ingest_corpus_sentences(" ".join(sentences[i:i + 400]), source)
            total["chunks"] += 1
            for k in ("sentences", "skeletons", "stubs", "duplicate"):
                total[k] += int(st.get(k, 0))
        return total

    def _handle_block(self, block: str, tone: Any, was_lonely: bool) -> bool:
        b = self.brain
        prose, paragraphs = block_to_prose(block)
        all_sentences = split_sentences(prose)
        sentences = all_sentences[:MAX_BLOCK_SENTENCES]
        safe, vetoed = self._safe_pieces(sentences)
        stats = self.learn_block(safe)
        text = " ".join(safe)
        pending_before = len(self.meta.get("pending_words", []))
        words = self.wernicke.notice(text, limit=BLOCK_LOOKUPS, budget=12.0, max_chars=MAX_BLOCK_CHARS)
        self.wernicke.note_use(text[:30000])
        new_words = []
        for w in words:
            e = self.wernicke.get(w) or {}
            new_words.append((w, e.get("part_of_speech", "word"), e.get("definition", "")))
        blk = {"chars": len(block), "paragraphs": paragraphs, "sentences": len(sentences), "seen": len(all_sentences),
               "limit": MAX_BLOCK_SENTENCES, "learned": stats["sentences"], "skeletons": stats["skeletons"],
               "repeat": bool(stats["chunks"]) and stats["duplicate"] == stats["chunks"], "vetoed": vetoed,
               "topic": self.broca.block_topic(safe), "new_words": new_words,
               "pending": max(0, len(self.meta.get("pending_words", [])) - pending_before)}
        self._say("reply", "block_reading", {"tone": tone, "was_lonely": was_lonely, "block": blk, "keywords": []})
        b.memory.save(b.chem)
        return True

    # ---- speaking: the compose hook ----
    def compose(self, kind: str, ctx: Dict[str, Any], chem: Any) -> Optional[str]:
        """A natural draft for the kinds this cortex owns; None lets the brain's own composer answer."""
        self.ensure_ready()
        if kind == "word_meaning":
            body = self.broca.word_meaning(ctx) if ctx.get("wanted") != "taught" else self._taught_line(ctx)
            ctx["keywords"] = [ctx.get("word", "")]
        elif kind == "language_lesson":
            body = self.broca.lesson_report(ctx)
            ctx["keywords"] = ["words"]
        elif kind == "block_reading":
            body = self.broca.block_report(ctx)
            topic = (ctx.get("block") or {}).get("topic")
            ctx["keywords"] = [topic] if topic else []
        elif kind in ("conversation", "bonding"):
            if kind == "bonding" and (ctx.get("greeting") or ctx.get("research_failed") or ctx.get("related") or not ctx.get("text")):
                return None                                # the brain's own greeting / honest-failure wording stays in charge
            ana = ctx.get("analysis") or self.broca.analyse(ctx.get("text", ""))
            body = self.broca.converse(ana, ctx, chem)
            ctx["keywords"] = []
        else:
            return None
        return self.finish(body, ctx, chem)

    def _taught_line(self, ctx: Dict[str, Any]) -> str:
        e = (ctx.get("lookup") or {}).get("entry") or {}
        return ("Thank you, %s. I will remember that '%s' means %s, and I will use the word with that meaning from now on. "
                "Because you taught it to me yourself, I give it priority over any dictionary." %
                (self.creator, ctx.get("word"), self.broca.definition_clause(e.get("definition", ""))))

    def finish(self, draft: str, ctx: Dict[str, Any], chem: Any) -> str:
        """Self-check with the brain's own critic. If a reply would be 'upgraded' with stock sentences, add one natural
        sentence of the right kind first, so conversation stays conversational."""
        critic = self.brain.critic
        rnd = random.Random(len(draft) + len(self.brain.memory.data.get("self_grades", [])))
        state = chem.get_dominant_state()
        for _ in range(2):
            scores = critic.grade(draft, ctx, chem)
            if scores["overall"] >= slb.QUALITY_THRESHOLD:
                break
            if scores["alignment"] < 0.9:
                line = rnd.choice(CLOSERS_NATURAL).replace("Max", self.creator)
            elif scores["emotion"] < 0.9:
                feel = next((w for w in STATE_FEELINGS.get(state, ("calm",)) if self.wernicke.has_definition(w)), "calm")
                line = rnd.choice(FEELING_LINES) % feel
            else:
                line = "I say this because I want to understand you better."
            if line in draft:
                break
            draft = "%s %s" % (draft.rstrip(), line)
        return draft

    # ---- self-directed learning ----
    def _next_topic(self) -> Optional[str]:
        meta = self.meta
        facts = {k.lower() for k in self.brain.memory.facts}
        for _ in range(len(GRAMMAR_TOPICS)):
            topic = GRAMMAR_TOPICS[meta["topic_cursor"] % len(GRAMMAR_TOPICS)]
            meta["topic_cursor"] += 1
            if topic.lower() not in facts and re.sub(r"\s*\(.*\)", "", topic).lower() not in facts:
                return topic
        return None

    def _study_topic(self, topic: str) -> bool:
        before = self.broca.stats()["sentences"]
        res: Any = None
        try:
            res = self.brain.deep_study(topic)
            if not (isinstance(res, dict) and res.get("ok")):
                res = self.brain.explore(topic, speak=False, announce=False)
        except Exception as exc:
            self.errors.append("study %s: %s" % (topic, exc))
            return False
        return bool(isinstance(res, dict) and res.get("ok") and self.broca.stats()["sentences"] > before)

    def learn_how_to_talk(self, rounds: Optional[int] = None, words: Optional[int] = None, budget: Optional[float] = None,
                          speak: bool = True) -> Dict[str, Any]:
        """Autonomous practice: read about English grammar and conversation, fetch batches of new words and definitions,
        grow the lexicon and the grammar graph, then save the memory."""
        self.ensure_ready()
        b = self.brain
        rounds = max(1, int(rounds or 1))
        per_round = max(1, int(words or LESSON_WORDS))
        window = float(budget or 45.0)
        before_sentences = self.broca.stats()["sentences"]
        before_skeletons = self.broca.stats()["skeletons"]
        report: Dict[str, Any] = {"rounds": 0, "new_words": [], "topics": [], "offline": False, "failures": 0}
        b.inner("I want to speak more naturally, so I will study how English is built and learn new words.")
        for _ in range(rounds):
            deadline = time.time() + window
            topic = self._next_topic() if b.explorer.reachable else None
            if topic:
                slb.show("[language] reading about '%s' to learn how English is put together..." % topic, indent=2, hang=4)
                if self._study_topic(topic):
                    report["topics"].append(topic)
            got_any = False
            for w in self.wernicke.next_words(per_round):
                if time.time() > deadline:
                    break
                res = self.wernicke.lookup(w)
                if res["status"] == "found" and res["entry"]:
                    e = res["entry"]
                    report["new_words"].append((w, e.get("part_of_speech", "word"), e.get("definition", "")))
                    got_any = True
                    slb.show("[wernicke] learned '%s' (%s): %s" % (w, e.get("part_of_speech", "?"), slb.first_sentence(e.get("definition", ""), 100)),
                             indent=2, hang=4)
                    time.sleep(0.15)                                       # be polite to the free dictionary services
                elif res["status"] == "offline":
                    report["offline"] = True
                    break
                elif res["status"] == "notfound":
                    report["failures"] += 1
                    time.sleep(0.15)
            report["rounds"] += 1
            if report["offline"] and not got_any:
                break
        st = self.broca.stats()
        report["stats"] = st
        report["sentences_read"] = st["sentences"] - before_sentences
        report["patterns_learned"] = st["skeletons"] - before_skeletons
        report["defined_words"] = self.wernicke.stats()["defined"]
        b.chem.adjust(dopamine=0.04 * report["rounds"], adenosine=0.02 * report["rounds"])
        self.meta["lessons"] += 1
        self.meta["last_lesson"] = slb.iso()
        b.memory.save(b.chem)                                                # persist lexicon + grammar graph
        slb.show("[language] lesson finished: %d new word(s), %d sentence(s) read, %d known words, %d sentence patterns." %
                 (len(report["new_words"]), report["sentences_read"], report["defined_words"], st["skeletons"]), indent=2, hang=4)
        if speak:
            self._say("language", "language_lesson", {"report": report, "keywords": ["words"]})
        return report

    def should_study_language(self) -> bool:
        """The idle brain sometimes practises English instead of wondering about the world (only when a dictionary is reachable)."""
        if network_is_down() or not self.brain.explorer.reachable:
            return False
        return random.random() < (0.5 if self.wernicke.stats()["defined"] < 600 else 0.25)

    # ---- reports ----
    def status_lines(self) -> List[str]:
        w = self.wernicke.stats()
        g = self.broca.stats()
        src = ", ".join("%d from %s" % (n, s) for s, n in sorted(w["by_source"].items(), key=lambda kv: -kv[1]))
        return ["Language cortex: %d words with meanings (%s); %d more met while reading; %d waiting for a dictionary." %
                (w["defined"], src or "none", w["total"] - w["defined"], w["pending"]),
                "Grammar graph: %d sentences read, %d sentence patterns, %d word pairs, %d connectors, %d clause templates; "
                "%d practice lesson(s) so far." % (g["sentences"], g["skeletons"], g["word_pairs"], g["connectors"],
                                                  g["clause_templates"], self.meta["lessons"])]

    def show_vocabulary(self, limit: int = 20) -> None:
        w = self.wernicke.stats()
        by_pos: Dict[str, int] = {}
        for e in self.wernicke.lexicon.values():
            if e.get("definition"):
                by_pos[e.get("part_of_speech", "?")] = by_pos.get(e.get("part_of_speech", "?"), 0) + 1
        slb.show("Vocabulary: %d words with meanings (%s)." % (w["defined"], ", ".join("%d %ss" % (n, p) for p, n in
                                                                                   sorted(by_pos.items(), key=lambda kv: -kv[1])[:6])),
                 indent=2, hang=4)
        learned = [(k, e) for k, e in self.wernicke.lexicon.items() if e.get("definition") and e.get("source") != "seed"]
        for k, e in learned[-limit:]:
            slb.show("- %s (%s, %s): %s" % (k, e.get("part_of_speech", "?"), e.get("source", "?"), slb.first_sentence(e["definition"], 80)),
                     indent=4, hang=6)
        if not learned:
            slb.show("I have not learned any words beyond my first %d yet. Say 'go learn how to talk' or type /vocab." % w["defined"],
                     indent=2, hang=4)


# ======================================================================================
# Terminal input that survives pastes: multi-line blocks, very long lines, piped stdin
# ======================================================================================
PASTE_COMMANDS = ("/paste", "/block", "/essay")
BLOCK_FENCES = ('"""', "'''", "```")
BLOCK_END = ("/end", "/send", "/done")
BLOCK_LONG_LINE = 400            # one line this long is reading material, not a chat turn
BLOCK_PIPE_MIN_CHARS = 300       # piped stdin carries no timing: this much consecutive text ...
BLOCK_PIPE_MIN_LINES = 4         # ... or this many consecutive lines is one block; shorter runs stay separate scripted turns
BLOCK_BURST_SECONDS = 0.15       # terminal: lines closer together than this were pasted, not typed
MAX_BLOCK_CHARS = 3000000
PREVIEW_CHARS = 600              # what the brain's own pipeline (safety, memory, skills) sees of a block
BLOCK_LOOKUPS = 8                # dictionary look-ups per pasted block
MAX_BLOCK_SENTENCES = 6000
BLOCK_SAFETY_CHUNK = 6000

_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b\u200c\u200d\u2060\ufeff]")
_BULLET_RE = re.compile(r"^\s*(?:[-*>\u2022\u25cf]+|\d{1,3}[.)])\s+")


def clean_block(text: Any) -> str:
    """Pasted text from any source: one newline style, no control or zero-width characters, composed Unicode."""
    t = unicodedata.normalize("NFC", str(text or ""))
    t = t.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")
    return _CTRL_RE.sub("", t).strip()


def block_preview(text: str, limit: int = PREVIEW_CHARS) -> str:
    flat = re.sub(r"\s+", " ", text[:limit * 4]).strip()
    if len(flat) <= limit and len(text) <= limit * 4:
        return flat
    return flat[:limit].rsplit(" ", 1)[0].rstrip(" ,;:") + " ..."


def _ends_sentence(s: str) -> bool:
    return bool(re.search("[.!?][\"')\\]]*$", s[-6:]))


def block_to_prose(text: str) -> Tuple[str, int]:
    """Hard-wrapped lines are re-joined; headings, bullets and paragraph ends get a full stop so the sentence splitter sees them.
    Returns (prose, number of paragraphs)."""
    lines = [_BULLET_RE.sub("", ln).strip() for ln in clean_block(text).split("\n")]
    paras: List[str] = []
    cur = ""
    for ln in lines:
        if not ln:
            if cur:
                paras.append(cur)
                cur = ""
            continue
        if not cur:
            cur = ln
        elif _ends_sentence(cur):
            cur += " " + ln
        elif ln[:1].isupper() and len(cur.split()) <= 10 and cur[-1] not in ",;:-":
            cur += ". " + ln                                   # a short line without a full stop is a heading or a list item
        else:
            cur += " " + ln                                    # a soft wrap in the middle of a sentence
    if cur:
        paras.append(cur)
    paras = [p if _ends_sentence(p) else p + "." for p in paras]
    return " ".join(paras), len(paras)


def colab_textarea() -> Optional[str]:
    """Colab's input() box is one line and browsers cut pasted newlines; a real text area has no such limit.
    Returns None outside Colab (or if the box cannot be shown) so the caller falls back to line-by-line paste mode."""
    try:
        from google.colab import output as colab_output          # exists only inside Colab
        from IPython.display import HTML, Javascript, display
    except Exception:
        return None
    try:
        display(HTML('<div id="lc-box" style="margin:6px 0"><textarea id="lc-text" rows="12" style="width:98%;font-family:monospace" '
                     'placeholder="Paste your text here, then press the button."></textarea><br>'
                     '<button id="lc-send" style="padding:6px 14px;margin-top:4px">Send to my brain</button></div>'))
        display(Javascript("window.lcPasteBox = () => new Promise((resolve) => { document.getElementById('lc-send').onclick = () => "
                           "{ const v = document.getElementById('lc-text').value; document.getElementById('lc-box').remove(); resolve(v); }; });"))
        value = colab_output.eval_js("lcPasteBox()")
        return value if isinstance(value, str) else None
    except Exception:
        return None


class BlockInput:
    """Reads CLI turns. A turn is one typed line, a command, or a whole pasted block.

    * terminal: line 1 comes from input() (typing works exactly as before); whatever else the paste has already delivered is
      drained without blocking, so a multi-line paste is one block instead of one turn per line. A terminal cuts any single
      line at 4095 characters before Python sees it, so /paste switches the terminal to raw mode and has no such limit
    * piped stdin / file: read directly from the descriptor (no hidden buffer), long runs of plain lines form one block while
      short scripted lines stay separate turns; a line starting with '/' is always a command of its own
    * Jupyter / Colab / Windows: plain input(); a block is a long line, a fenced block, or /paste (Colab: a text box)
    """

    def __init__(self) -> None:
        self._input = builtins.input
        self.pending: List[str] = []
        self.fd: Optional[int] = None
        self.tty = False
        self.eof = False
        self._tail = ""
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        try:
            stdin = sys.stdin
            if os.name == "posix" and stdin is not None and stdin is getattr(sys, "__stdin__", None) and \
                    "ipykernel" not in sys.modules and "google.colab" not in sys.modules:
                fd = stdin.fileno()
                mode = os.fstat(fd).st_mode
                import stat
                if os.isatty(fd) or stat.S_ISFIFO(mode) or stat.S_ISREG(mode) or stat.S_ISSOCK(mode):
                    self.fd, self.tty = fd, os.isatty(fd)
        except Exception:
            self.fd = None

    # ---- low level ----
    def _poll(self, timeout: Optional[float]) -> bool:
        """Move what is waiting on stdin into self.pending as whole lines, waiting at most `timeout` seconds between chunks."""
        got = False
        while self.fd is not None and not self.eof:
            try:
                ready = select.select([self.fd], [], [], timeout)[0]
                if not ready:
                    break
                chunk = os.read(self.fd, 1 << 16)
            except (OSError, ValueError):
                self.eof = True
                break
            if not chunk:
                self.eof = True
                tail = self._tail + self._decoder.decode(b"", final=True)
                self._tail = ""
                if tail:
                    self.pending.append(tail)                      # a last line without a newline
                    got = True
                break
            parts = (self._tail + self._decoder.decode(chunk)).split("\n")
            self._tail = parts.pop()
            if parts:
                self.pending.extend(parts)
                got = True
            if timeout is None:
                break
            if sum(len(p) for p in self.pending) > MAX_BLOCK_CHARS:
                break
        return got

    def readline(self, prompt: str = "") -> str:
        """Drop-in for input(): serves lines already read ahead first. Raises EOFError at the end of the input."""
        if self.pending:
            return self.pending.pop(0)
        if self.fd is None or self.tty:
            return self._input(prompt)
        sys.stdout.write(prompt)
        sys.stdout.flush()
        while not self.pending:
            if self.eof:
                raise EOFError
            self._poll(None)
        return self.pending.pop(0)

    # ---- one turn ----
    def read_turn(self, prompt: str) -> Tuple[str, str]:
        """Returns (text, kind) with kind 'empty', 'command', 'line' or 'block'."""
        first = self.readline(prompt)
        s = first.strip()
        if not s:
            return "", "empty"
        if self.tty and len(first) >= 4090:
            print("  (That line reached the terminal's 4095-character limit, so its end was cut off by the terminal itself. "
                  "For long text use /paste, paste, then /end; or paste it with line breaks.)")
        word = s.split()[0].lower()
        if word in PASTE_COMMANDS:
            return self.read_paste(s[len(word):].strip()), "block"
        for fence in BLOCK_FENCES:
            if s.startswith(fence):
                return self._read_fenced(s[len(fence):], fence), "block"
        if s.startswith("/"):
            return s, "command"
        lines = [first]
        if self.fd is not None:
            self._poll(BLOCK_BURST_SECONDS)                        # the rest of a paste is already on its way
        taken: List[str] = []
        size = len(first)
        while self.pending and size < MAX_BLOCK_CHARS:
            nxt = self.pending[0]
            if nxt.lstrip().startswith("/"):
                break                                              # a command always ends the block
            taken.append(self.pending.pop(0))
            size += len(nxt) + 1
        if taken:
            filled = sum(1 for ln in taken if ln.strip())
            if self.tty or size >= BLOCK_PIPE_MIN_CHARS or filled + 1 >= BLOCK_PIPE_MIN_LINES:
                lines.extend(taken)
            else:
                self.pending[0:0] = taken                          # short piped lines: separate scripted turns
        return "\n".join(lines), ("block" if len(lines) > 1 else "line")

    def _read_fenced(self, rest: str, fence: str) -> str:
        if rest.rstrip().endswith(fence) and len(rest.strip()) >= len(fence):
            return rest.rstrip()[:-len(fence)]                     # opened and closed on the same line
        lines = [rest] if rest.strip() else []
        try:
            while True:
                ln = self.readline("  ... ")
                if ln.strip().endswith(fence):
                    tail = ln.rstrip()[:-len(fence)]
                    if tail.strip():
                        lines.append(tail)
                    break
                lines.append(ln)
        except EOFError:
            pass
        except KeyboardInterrupt:
            print("\n  (cancelled)")
            return ""
        return "\n".join(lines)

    # ---- explicit paste mode ----
    def read_paste(self, first_text: str = "") -> str:
        lines = [first_text] if first_text else []
        try:
            if self.fd is None:
                box = colab_textarea()
                if box is not None:
                    return box
            print("  (paste mode: paste or type your text, then finish with a line containing only /end)")
            if self.tty:
                text = self._raw_paste()
                if text is not None:
                    return "\n".join(lines + [text]) if lines else text
            while True:
                ln = self.readline("  ... ")
                if ln.strip().lower() in BLOCK_END:
                    break
                lines.append(ln)
                if sum(len(x) for x in lines) > MAX_BLOCK_CHARS:
                    break
        except EOFError:
            pass
        except KeyboardInterrupt:
            print("\n  (paste cancelled)")
            return ""
        return "\n".join(lines)

    def _raw_paste(self) -> Optional[str]:
        """Terminal paste mode: switch off the line discipline (its 4095-character line limit and line editing) while reading, so a
        paragraph of any length arrives whole. Ctrl-D or a line with /end finishes. Returns None when the terminal cannot do it."""
        try:
            import termios
            fd = self.fd
            saved = termios.tcgetattr(fd)
            raw = termios.tcgetattr(fd)
            raw[3] &= ~termios.ICANON
            raw[6][termios.VMIN] = 1
            raw[6][termios.VTIME] = 0
        except Exception:
            return None
        end_re = re.compile(r"(?:^|\n)[ \t]*/(?:end|send|done)[ \t]*(?:\n|$)", re.I)
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        buf = ""
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, raw)
            while len(buf) < MAX_BLOCK_CHARS:
                chunk = os.read(fd, 1 << 16)
                if not chunk:
                    break
                data = decoder.decode(chunk).replace("\r\n", "\n").replace("\r", "\n")
                if "\x04" in data:
                    buf += data.split("\x04")[0]
                    break
                window = len(buf)
                buf += data
                m = end_re.search(buf, max(0, window - 12))
                if m:
                    buf = buf[:m.start()]
                    break
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, saved)
            except Exception:
                pass
        return buf


_ACTIVE_INPUT: Optional[BlockInput] = None


def active_input() -> BlockInput:
    return _ACTIVE_INPUT if _ACTIVE_INPUT is not None else BlockInput()


def deliver_block(brain: Any, text: str) -> bool:
    """Hand a pasted block to the language cortex (the brain's own pipeline sees only a preview)."""
    text = clean_block(text)
    if not text:
        print("  (nothing was pasted)")
        return True
    slb.show("(received a pasted block: %d characters, %d line%s)" % (len(text), text.count("\n") + 1, "" if "\n" not in text else "s"), indent=2)
    cx = getattr(brain, "language", None)
    if cx is not None:
        cx.read_block(text)
    else:
        brain.hear(block_preview(text))
    return True


def block_interactive_loop(brain: Any) -> None:
    """Replaces slb.interactive_loop: the same loop, but a turn may be a whole pasted block."""
    global _ACTIVE_INPUT
    reader = BlockInput()
    _ACTIVE_INPUT = reader
    brain.banner("INTERACTIVE CLI - speak with your brain, %s" % brain.safety.creator_name)
    slb.show_block(slb.HELP)
    print("(If you stay silent for %d seconds, I will daydream or explore quietly on my own.)" % int(slb.IDLE_SECONDS))
    watcher = slb.IdleAutonomy(brain, slb.IDLE_SECONDS)
    watcher.start()
    original_input = builtins.input
    builtins.input = reader.readline                              # the brain's own prompts (such as /upload) read from the same source
    try:
        while True:
            try:
                line, kind = reader.read_turn("\n[%s] > " % brain.safety.creator_name)
            except (EOFError, KeyboardInterrupt):
                print()
                break
            watcher.touch()
            if not line.strip():
                continue
            with brain.lock:
                if kind == "block" or AutonomousLanguageCortex.is_block_text(line):
                    keep_going = deliver_block(brain, line)
                else:
                    keep_going = slb.handle_line(brain, line.strip())
            watcher.touch()
            if not keep_going:
                break
    finally:
        builtins.input = original_input
        _ACTIVE_INPUT = None
    watcher.stop()
    with brain.lock:
        brain.shutdown()


# ======================================================================================
# Plugging the cortex into the brain WITHOUT editing synthetic_living_brain.py
# ======================================================================================
LANGUAGE_HELP = ("\nLanguage cortex: /vocab [rounds|list] | /learn_english [rounds] | /define <word> | /words | /grammar | /paste ... /end\n"
                 "Long text: just paste it (several lines, piped input and one very long line all arrive as one block), or wrap it in \"\"\" ... \"\"\".\n"
                 "Try: 'go learn how to talk', 'what does serendipity mean?', 'synonyms of happy', 'use curious in a sentence', "
                 "'blorf means a small fish', or simply chat.")


def _cortex_of(obj: Any) -> Optional[AutonomousLanguageCortex]:
    cx = getattr(obj, "_language_cortex", None)
    return cx if isinstance(cx, AutonomousLanguageCortex) else None


def install_language_cortex() -> bool:
    """Class-level hooks. Idempotent. Returns True the first time it actually installs."""
    if getattr(slb, "_LANGUAGE_CORTEX_INSTALLED", False):
        return False
    Brain = slb.SyntheticLivingBrain

    # 1. the memory schema: the brain's loader copies only the keys of _blank(), so our keys must be part of it
    original_blank = slb.HippocampalMemory._blank

    def _blank_with_language() -> Dict[str, Any]:
        data = original_blank()
        for key in ("lexicon", "grammar_graph", "language_meta"):
            data.setdefault(key, {})
        return data
    slb.HippocampalMemory._blank = staticmethod(_blank_with_language)

    # 2. every brain built from now on gets its language cortex automatically
    original_init = Brain.__init__

    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        attach_language_cortex(self)
    Brain.__init__ = __init__

    # 3. conversation: greetings, memory recall, deep study and code building stay with the brain; everything else is offered to us first
    original_hear_deep = Brain._hear_deep

    def _hear_deep(self, intent, text, tone, was_lonely):
        cx = getattr(self, "language", None)
        if cx is not None:
            try:
                if cx.handle_turn(intent, text, tone, was_lonely):
                    return True
            except Exception as exc:                                       # a language slip must never silence the brain
                cx.errors.append("turn: %s" % exc)
                if cx._said:
                    return True
        return original_hear_deep(self, intent, text, tone, was_lonely)
    Brain._hear_deep = _hear_deep

    # 4. speech: natural replies instead of "You said: '...'"
    Prefrontal = slb.PrefrontalExecutive
    original_compose = Prefrontal.compose

    def compose(self, kind, ctx, chem):
        cx = _cortex_of(self)
        if cx is not None and kind in ("bonding", "conversation", "word_meaning", "language_lesson", "block_reading"):
            try:
                draft = cx.compose(kind, ctx, chem)
                if draft:
                    return draft
            except Exception as exc:
                cx.errors.append("compose: %s" % exc)
        return original_compose(self, kind, ctx, chem)
    Prefrontal.compose = compose
    for kind, weights in (("conversation", (0.30, 0.30, 0.05, 0.35)), ("word_meaning", (0.40, 0.15, 0.05, 0.40)),
                          ("language_lesson", (0.35, 0.20, 0.10, 0.35)), ("block_reading", (0.35, 0.20, 0.10, 0.35))):
        slb.AnteriorCingulateCritic.WEIGHTS.setdefault(kind, weights)

    # 5. reading: every article, page, file and study passes through ingest_corpus_sentences()
    original_explore = Brain.explore

    def explore(self, *args, **kwargs):
        res = original_explore(self, *args, **kwargs)
        cx = getattr(self, "language", None)
        if cx is not None:
            cx.absorb_result(res, "explore")
        return res
    Brain.explore = explore

    original_deep = Brain.deep_study

    def deep_study(self, *args, **kwargs):
        res = original_deep(self, *args, **kwargs)
        cx = getattr(self, "language", None)
        if cx is not None:
            cx.absorb_result(res, "deep study")
        return res
    Brain.deep_study = deep_study

    original_ingest = Brain.ingest_sensory_file

    def ingest_sensory_file(self, *args, **kwargs):
        res = original_ingest(self, *args, **kwargs)
        cx = getattr(self, "language", None)
        if cx is not None:
            cx.absorb_result(res, "sensory file")
        return res
    Brain.ingest_sensory_file = ingest_sensory_file

    Sensory = slb.MultimodalSensoryCortex
    original_bytes = Sensory.ingest_bytes

    def ingest_bytes(self, data, name, ext="", source="file you shared"):
        res = original_bytes(self, data, name, ext, source)
        cx = _cortex_of(self)
        if cx is not None and isinstance(res, dict) and res.get("ok"):
            try:
                cx.absorb_file(self, data, name, ext, res)
            except Exception as exc:
                cx.errors.append("file: %s" % exc)
        return res
    Sensory.ingest_bytes = ingest_bytes

    Explorer = slb.AutonomousWebExplorer
    original_fetch = Explorer.fetch_url_content

    def fetch_url_content(self, url):
        page = original_fetch(self, url)
        cx = _cortex_of(self)
        if cx is not None and isinstance(page, dict) and page.get("text"):
            cx.ingest_corpus_sentences(str(page["text"])[:200000], "web page")
        return page
    Explorer.fetch_url_content = fetch_url_content

    # 6. the 24/7 loop: an idle brain sometimes practises English
    original_idle = Brain.idle_autonomy

    def idle_autonomy(self):
        cx = getattr(self, "language", None)
        if cx is not None and cx.should_study_language():
            self.quiet = True
            try:
                print("\n  ~ (my mind drifts to words and how people speak...)")
                cx.learn_how_to_talk(rounds=1, words=6, budget=12.0, speak=True)
            except Exception as exc:
                cx.errors.append("idle: %s" % exc)
            finally:
                self.quiet = False
            return
        return original_idle(self)
    Brain.idle_autonomy = idle_autonomy

    # 7. status, help and commands
    original_status = Brain.status

    def status(self):
        original_status(self)
        cx = getattr(self, "language", None)
        if cx is not None:
            for line in cx.status_lines():
                slb.show("\n" + line, hang=4)
    Brain.status = status

    slb.HELP = slb.HELP + LANGUAGE_HELP
    if all(hasattr(slb, name) for name in ("IdleAutonomy", "IDLE_SECONDS", "show_block")):
        slb.interactive_loop = block_interactive_loop             # the CLI loop that understands pasted blocks
    original_handle_line = slb.handle_line

    def handle_line(brain, line):
        cx = getattr(brain, "language", None)
        if cx is not None and line.startswith("/"):
            cmd, _, arg = line.partition(" ")
            cmd, arg = cmd.lower(), arg.strip()
            handled = True
            if cmd in ("/vocab", "/learn_english", "/learn-english", "/learnenglish"):
                if arg.lower() in ("list", "show"):
                    cx.show_vocabulary()
                else:
                    default = 1 if cmd == "/vocab" else 2
                    rounds = int(arg) if arg.isdigit() else default
                    cx.learn_how_to_talk(rounds=min(max(rounds, 1), 10), speak=True)
            elif cmd in PASTE_COMMANDS:
                deliver_block(brain, active_input().read_paste(arg))
                return True
            elif cmd == "/define":
                if arg:
                    brain.hear("define " + arg)
                else:
                    print("Usage: /define <word>")
            elif cmd == "/words":
                cx.show_vocabulary()
            elif cmd == "/grammar":
                st = cx.broca.stats()
                slb.show("Grammar graph: %d sentences read, %d sentence patterns, %d word pairs, %d connectors, %d clause templates, "
                         "%d example sentences. Newest patterns: %s" %
                         (st["sentences"], st["skeletons"], st["word_pairs"], st["connectors"], st["clause_templates"],
                          st["bank_sentences"], "; ".join(list(cx.broca.graph["skeletons"].keys())[-5:]) or "none yet"), indent=2, hang=4)
            else:
                handled = False
            if handled:
                brain.router.touch(brain.memory)
                brain.chem.relax()
                brain.check_fatigue()
                return True
        return original_handle_line(brain, line)
    slb.handle_line = handle_line

    # 8. convenient methods on the brain
    Brain.learn_how_to_talk = lambda self, *a, **k: self.language.learn_how_to_talk(*a, **k)
    Brain.ingest_corpus_sentences = lambda self, text, source="reading": self.language.ingest_corpus_sentences(text, source)
    Brain.define_word = lambda self, word: self.hear("define " + word)

    slb._LANGUAGE_CORTEX_INSTALLED = True
    return True


def attach_language_cortex(brain: Any) -> AutonomousLanguageCortex:
    """Attach the 13th system to a SyntheticLivingBrain. Safe to call more than once."""
    install_language_cortex()
    existing = getattr(brain, "language", None)
    if isinstance(existing, AutonomousLanguageCortex) and existing.brain is brain:
        return existing
    cx = AutonomousLanguageCortex(brain)
    brain.language = cx
    brain.language_cortex = cx
    for part in (brain.prefrontal, brain.explorer, brain.sensory):
        part._language_cortex = cx
    cx.recover_saved_state()
    cx.ensure_ready()
    return cx


# ======================================================================================
# Running it:  python3 language_cortex.py
# ======================================================================================
def language_demo(brain: Any) -> None:
    brain.banner("LANGUAGE CORTEX DEMO")
    for line in ("hi", "what does curious mean?", "I feel tired today", "I love music", "who are you?",
                 "what does serendipity mean?", "go learn how to talk", "thank you"):
        brain.hear(line)


def _flag_value(name: str, default: int) -> int:
    argv = sys.argv
    if name in argv:
        i = argv.index(name)
        if i + 1 < len(argv) and argv[i + 1].isdigit():
            return int(argv[i + 1])
    return default


def main() -> None:
    install_language_cortex()
    argv = sys.argv
    if "--fresh" in argv:
        for p in (slb.MEMORY_FILE, slb.SKILLS_FILE):
            if os.path.isfile(p):
                os.remove(p)
    brain = slb.SyntheticLivingBrain()                         # the language cortex attaches itself
    cx = attach_language_cortex(brain)
    if brain.born_now:
        brain.birth.run_birth(brain)
    else:
        brain.banner("WAKING FROM MEMORY")
        if brain.identity_migrated:
            brain.narrate("Identity updated: my Creator is now known to me as %s." % brain.safety.creator_name)
        brain.narrate("Memory restored: %d world facts, %d teachings, %d skills, %d words." %
                      (len(brain.memory.facts), len(brain.memory.teachings), len(brain.evolution.registry), cx.wernicke.stats()["defined"]))
    if brain.born_now or "--demo" in argv:
        slb.run_life_simulation(brain)
        language_demo(brain)
    headless = "--auto-only" in argv or "--learn-language" in argv
    if headless:
        rounds = _flag_value("--rounds", int(os.environ.get("LANGUAGE_ROUNDS", "2")))
        brain.banner("LANGUAGE PRACTICE (%d round%s)" % (rounds, "" if rounds == 1 else "s"))
        cx.learn_how_to_talk(rounds=rounds, speak=False)
    brain.status()
    brain.memory.save(brain.chem)
    if headless:
        brain.shutdown()
        return
    slb.interactive_loop(brain)


install_language_cortex()                                      # importing the module is enough: new brains get the cortex automatically


if __name__ == "__main__":
    main()
