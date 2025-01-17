from chatnoir_pyterrier import ChatNoirRetrieve, Feature
import json
import os


pwd = os.path.dirname(os.path.abspath(__file__))


def load_config(filename=pwd + "/../config.json"):
    with open(filename, "r") as f:
        config = json.load(f)
    return config


# Get the configuration settings
config = load_config()
CHATNOIR_TARGET_INDICES = config['CHATNOIR_TARGET_INDICES']

text1 = ' "Well, this is a debate about sexual education, and since that point seemed to be irrelevant to the \
      topic at hand, I didn\'t see much point in commenting."*facepalm*"What are you referring to as \'standard\' \
        sex education?"No comment... (Meaning "safe sex education")"Secondly, I must reiterate the fact that nothing, \
            not even surgery is 100% effective against pregnancy, but that does not mean that you should do it, \
                  as contraceptives (as asserted by the strong correlation in the previous round) does help prevent \
                    pregnancies [  1][2]."That still doesn\'t take away from the fact, abistinence is 100% effective \
                        against pregnancy and STD\'s, if practiced.  That being said, it needs to be taught in schools \
                            one way or the other.  Even if "safe sex" education is being taught in schools.  The \
                                government \\shouldn\'t have the right to only fund "safe sex" education and not \
                                    Abstinence programs."You\'ve contradicted yourself.  Should the person have no \
                                        risk, or not be infertile?  Secondly, many people, including myself, wish to \
                                            never be a father [  3].  Personally, I loathe children and understand the \
                                                huge amount of money it takes to raise a child."What I was saying \
                                                    is why take the rsik of being infertile, due to either \
                                                        contraceptives or STD\'s. '

text2 = (
    ' "Well, this is a debate about sexual education, and since that point seemed to be '
    'irrelevant to the topic at hand, I didn\'t see much point in commenting."*facepalm*"What '
    'are you referring to as \'standard\' sex education?"No comment... (Meaning "safe sex '
    'education")"Secondly, I must reiterate the fact that nothing, not even surgery is 100% '
    'effective against pregnancy, but that does not mean that you should do it, as '
    'contraceptives (as asserted by the strong correlation in the previous round) does help '
    'prevent pregnancies [  1][2]."That still doesn\'t take away from the fact, abistinence is '
    '100% effective against pregnancy and STD\'s, if practiced.  That being said, it needs to be '
    'taught in schools one way or the other.  Even if "safe sex" education is being taught in '
    'schools.  The government shouldn\'t have the right to only fund "safe sex" education and '
    'not Abstinence programs."You\'ve contradicted yourself.  Should the person have no risk, or '
    'not be infertile?  Secondly, many people, including myself, wish to never be a father [  3]. '
    'Personally, I loathe children and understand the huge amount of money it takes to raise a '
    'child."What I was saying is why take the rsik of being infertile, due to either '
    'contraceptives or STD\'s. '
)

chatnoir = ChatNoirRetrieve(index=CHATNOIR_TARGET_INDICES,
                            features=Feature.CONTENTS,
                            num_results=2000,
                            retrieval_system="bm25")
# results = chatnoir.search('What about us').loc[:, ['qid', 'docno']].head(20)
# print(results)

documentID = '9QQJQYBpWz-l-1PbkkEXjA'
warcID = 'b8117996-4581-43eb-b33a-348c5aa48a0a'
trecID = 'clueweb22-en0012-00-01587'
