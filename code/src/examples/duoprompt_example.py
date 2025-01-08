import autoqrels
import ir_datasets
import os
import json
import gzip

# Load the configuration settings
pwd = os.path.dirname(os.path.abspath(__file__))


def load_config(filename=pwd + "/../config.json"):
    with open(filename, "r") as f:
        config = json.load(f)
    return config


# Get the configuration settings
config = load_config()

DOCUMENT_DATASET_TARGET_NAME = config['DOCUMENT_DATASET_TARGET_NAME']
SOURCE_PATH = os.path.join(config['DATA_PATH'], config["DOCUMENT_DATASET_SOURCE_NAME"])
TARGET_PATH = os.path.join(SOURCE_PATH, config["DOCUMENT_DATASET_TARGET_NAME"])
DUOPROMPT_CACHE_PATH = os.path.join(TARGET_PATH, config['DUOPROMPT_CACHE_PATH'])

with gzip.open(DUOPROMPT_CACHE_PATH, 'wt') as fin:
    fin.write(json.dumps({}))

dataset = ir_datasets.load('argsme/2020-04-01/touche-2021-task-1')
duoprompt = autoqrels.oneshot.DuoPrompt(dataset=dataset, cache_path=DUOPROMPT_CACHE_PATH)


query = 'Do we need sex education in schools?'
rel_doc_text = 'Sex education should be taught in grade schools to students.  Puberty begins at the age of twelve for females and thirteen for males.  Puberty is a time of confusion, questioning, and self-examination, along with emotional and physical change.  Alot of children and teens are uncomfortable talking to their parents about sexual behaviors and questions and are left feeling helpless.  Sex education in grade schools allows a child to be able to freely express themselves emotionally, ask questions, gain answers, and be comfortable aroung peers who are experiencing the same difficult change.  The only thing different about sex education is that it is NOT a \"how-to\" class; it will NOT teach students how to have, perform, and enjoy sex, nor will it teach a child how to properly use contraceptives.  The class (program) is used to teach children about the difference in which their bodies is going through as well as teach them the bodily form, such as organs and etc.  It is better to have a child be taught about sex and the body by a mature and responsible teacher, rather than from the media.  Why sex education should be taught in grade schools.'
unk_doc_text = 'Thanks, SirCrona.  Biology I dont want to say this...  but right off the bat you say different sexes, implying we arent equal  [  1].  Although I do know what you mean, saying \"Men and women are equal... heres why men and women are different isnt too good for your argument, lol.  But back on point.  There are physical differences between the sexes, thanks for clarifying.  Youre right that a self-replicating animal, or one that reproduces asexually, would not pass on any real advantageous genes, and the only real form of change from the parent to offspring would be slight adaptations to the environment.  However my point is not that one sex has an advantage over the other or that one sex is better than the other - my point is that they are different/unequal.  If I have a red ball and green ball, they arent equal in colour, but theyre both useful for playing catch with.  Similar =/= equal.  Youre right that we both have nipples, however a womans nipples have a much different function than a mans.  She uses hers to produce milk to feed her offspring, whereas a man... well, he doesnt.'

result = duoprompt.infer_oneshot_text(query, rel_doc_text, [unk_doc_text])
print(result[0])
