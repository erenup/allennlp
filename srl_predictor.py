import argparse
import json
import collections
import spacy
import re

import unicodedata
from functools import partial
from multiprocessing import Pool
import re
import random
from copy import deepcopy
from multiprocessing import cpu_count

from allennlp.predictors.predictor import Predictor
from add_retrieved_text import add_retrieved_text
predictor_srl = Predictor.from_path(
    "https://s3-us-west-2.amazonaws.com/allennlp/models/bert-base-srl-2019.06.17.tar.gz", cuda_device=0)
srl = predictor_srl.predict(
    sentence="Did Uriah honestly think he could beat the game in under three hours?"
)
print(cpu_count())
print('srl test:')
print(srl)
spacy_nlp = spacy.load('en', parser=False)
spacy_result = spacy_nlp("Did Uriah honestly think he could beat the game in under three hours?")
print('spacy test')
for token in spacy_result:
    print(token.text, token.pos_)

# from allennlp.predictors.predictor import Predictor
# predictor_ctp = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz", cuda_device=0)
# ctp = predictor_ctp.predict(
#   sentence="If I bring 10 dollars tomorrow, can you buy me lunch?"
# )
# print(ctp)

from tqdm import tqdm

# analyse question and option word patterns:
# question:
# rather than, other than, what, which, best, more. most,
# option: a and b, a and and b and c

SRL_REGX = re.compile(r'.*?ARG0(.*?)V(.*?)ARG1(.*)')
SRL_REGX02 = re.compile(r'.*?ARG0(.*?)V(.*?)ARG2(.*)')
SRL_REGX12 = re.compile(r'.*?ARG1(.*?)V(.*?)ARG2(.*)')
MOD_VERBS = ['could', 'can', 'must', 'may', 'should', 'will']


def clean_spaces(text):
    """normalize spaces in a string."""
    text = re.sub(r'\s', ' ', text)
    return text


def init():
    """initialize spacy in each process"""
    global nlp
    nlp = spacy.load('en', parser=False)


def normalize_text(text):
    return unicodedata.normalize('NFD', text)


def token_question_option(example):
    question = nlp(clean_spaces(example['question']['stem']))
    options = [example['question']['choices'][i]['text'] for i in range(len(example['question']['choices']))]
    options = [nlp(option) for option in options]
    question_tokens = [w.text for w in question]
    question_pos = [w.pos_ for w in question]
    option_tokens = [[w.text for w in option] for option in options]
    option_pos = [[w.pos_ for w in option] for option in options]
    example['question_tokens'] = question_tokens
    example['question_pos'] = question_pos
    example['option_tokens'] = option_tokens
    example['option_pos'] = option_pos
    return example


def extract_srl(example):
    question = example['question']['stem']
    question_srl = predictor_srl.predict(sentence=question)
    options = example['question']['choices']
    options_srl = []
    for option in options:
        options_srl.append(predictor_srl.predict(sentence=option['text']))
    example['question']['stem_srl'] = question_srl
    example['question']['choices_srl'] = options_srl
    return example


def collect_structure_fusion_examples(example):
    # option adv:

    example['adv_type'] = 'srl_option'
    example['adv'] = False
    choices_srl = example['question']['choices_srl']
    options_pos = example['option_pos']
    choices = example['question']['choices']
    if len(choices) < 4:
        return example
    answer_choice = None
    adv_option = None
    verb_num = 0
    for i, choice_srl_choice_pos in enumerate(zip(choices_srl, options_pos)):
        choice_srl, choice_pos = choice_srl_choice_pos
        label = example['question']['choices'][i]['label']
        answerKey = example['answerKey']
        choice_text = choices[i]['text']
        if len(choice_srl['verbs']) == 0:
            continue
        else:
            verb_num += 1
            if label == answerKey:
                answer_choice = i
                verbs = choice_srl['verbs']
                words = choice_srl['words']
                for verb_dict in verbs:
                    verb = verb_dict['verb']
                    if verb in MOD_VERBS:
                        print('mode verbs')
                        continue
                    description = verb_dict['description']
                    tags = verb_dict['tags']
                    try:
                        # assert len(words) == len(choice_pos)
                        # words2pos = {word: pos for word, pos in zip(words, choice_pos)}
                        reg_results = SRL_REGX.match(description)
                        if reg_results:
                            print('match 01:', reg_results.groups())
                        if not reg_results:
                            reg_results = SRL_REGX02.match(description)
                            if reg_results:
                                print('match 02:', reg_results.groups())
                        if not reg_results:
                            reg_results = SRL_REGX12.match(description)
                            if reg_results:
                                print('match 12:', reg_results.groups())
                        if reg_results:
                            groups = reg_results.groups()
                            if len(groups) != 3:
                                print('groups len is not 3')
                                continue
                            arg_berfore = ' '.join(
                                [x for x in groups[0].replace('[', '').replace(']', '').split()[1:] if x])
                            verb_middel = [x for x in groups[1].replace('[', '').replace(']', '').split()[1:] if x]
                            arg_after = ' '.join(
                                [x for x in groups[2].replace('[', '').replace(']', '').split()[1:] if x])
                            words = ' '.join([x for x in words if x])
                            if arg_berfore in words and arg_after in words:
                                regex = re.compile(r'(%s)(.*)(%s)' % (arg_berfore, arg_after.strip('\.')))
                                adv_option = regex.sub(r'\3\2\1', words)

                            else:
                                print('cannot find before and after in original string')
                    except:
                        print('coding errors!')

    adv_num = 0
    if verb_num == 0:
        for i, choice in enumerate(choices):
            if answerKey == choice['label']:
                answer_choice = i
                break

        if not answer_choice:
            return example
        choices_num = [i for i in range(4) if i != answer_choice]
        answer_text = example['question']['choices'][answer_choice]['text']
        random_num = random.random()
        if random_num < 0.33:
            adv_num = choices_num[0]
        elif random_num < 0.66:
            adv_num = choices_num[1]
        else:
            adv_num = choices_num[2]

        example['question']['choices'][answer_choice]['text'] = example['question']['choices'][answer_choice]['text'] \
                                                                + ' is better than ' + \
                                                                example['question']['choices'][adv_num]['text']
        example['question']['choices'][adv_num]['text'] = example['question']['choices'][adv_num][
                                                              'text'] + ' is better than ' + answer_text
        example['adv'] = True
        example['adv_choice'] = adv_num
        choices_num_remain = [i for i in range(4) if i != answer_choice and i != adv_num]
        if len(choices_num_remain) == 2:
            example['question']['choices'][choices_num_remain[0]]['text'] = example['question']['choices'][choices_num_remain[0]][
                                                                        'text'] \
                                                                    + ' is better than ' + \
                                                                    example['question']['choices'][choices_num_remain[1]]['text']

            example['question']['choices'][choices_num_remain[1]]['text'] = example['question']['choices'][choices_num_remain[1]][
                                                                        'text'] \
                                                                    + ' is better than ' + \
                                                                    example['question']['choices'][choices_num_remain[0]]['text']
        else:
            print('remain choices:', choices_num_remain, ' skip')


    if adv_option:
        random_num = random.random()
        choices_num = [i for i in range(4) if i != answer_choice]
        adv_num = 0
        if random_num < 0.33:
            adv_num = 0
        elif random_num < 0.66:
            adv_num = 1
        else:
            adv_num = 2
        example['question']['choices'][choices_num[adv_num]]['text'] = adv_option
        example['adv'] = True
        example['adv_choice'] =choices_num[adv_num]
    return example

def write_examples_add_retrieve(input_file, examples):
    with open(input_file, 'w', encoding='utf-8') as fout:
        for example in examples:
            fout.write(json.dumps(example) + '\n')
    add_retrieved_text(input_file, input_file + '.add_retrieve')

def extract_srl_to_advs(input_file, examples_srl):
    examples_fustion = []
    examples_fustion_original = []
    examples_fustion_all = []
    examples_all_original = []
    input_file_all_original = input_file + '_original.all'
    input_file_srl_fusion = input_file + '.fusion'
    input_file_srl_fusion_original = input_file_srl_fusion + '.original'
    input_file_srl_fusion_all = input_file_srl_fusion + '.all'
    for example in examples_srl:
        example_fusion = collect_structure_fusion_examples(deepcopy(example))
        if example_fusion['adv']:
            examples_fustion.append(example_fusion)
            examples_fustion_original.append(example)
        examples_fustion_all.append(example_fusion)
        examples_all_original.append(example)
    # examples_fustion = [example for example in examples_fustion if example['adv'] == True]
    print('fusions write to:', input_file_srl_fusion)
    write_examples_add_retrieve(input_file_srl_fusion, examples_fustion)
    write_examples_add_retrieve(input_file_srl_fusion_original, examples_fustion_original)
    write_examples_add_retrieve(input_file_srl_fusion_all, examples_fustion_all)
    write_examples_add_retrieve(input_file_all_original, examples_all_original)

def analyse(input_file, args):
    examples_srl = []
    if args.convert_srl:
        examples = read_arc_ai2_retrieve(input_file, args)
        input_file_srl = input_file + '.srl'

        print('start multiple processing ', 'example nums:', len(examples))
        with Pool(args.threads, initializer=init) as p:
            annotate_ = partial(token_question_option)
            examples_tokens = list(tqdm(p.imap(annotate_, examples, chunksize=32), total=len(examples), desc='tokens'))
        print('start to extract srl:')
        for example in tqdm(examples_tokens):
            srl = extract_srl(example)
            examples_srl.append(srl)
        with open(input_file_srl, 'w', encoding='utf-8') as fout:
            json.dump(examples_srl, fout)
        print('start to collect fusions')
        extract_srl_to_advs(input_file, examples_srl)

    elif args.srl_file:
      with open(args.srl_file, 'r', encoding='utf-8') as fin:
        examples_srl = json.load(fin)
        extract_srl_to_advs('.'.join(args.srl_file.split('.')[:-1]), examples_srl)

    # question_tokens = []
    # option_tokens = []
    # question_tokens_2grams = []
    # option_tokens_2grams = []
    # question_tokens_3grams = []
    # option_tokens_3grams = []
    # for question_token, option_token, question_tokens_2gram, option_tokens_2gram, question_tokens_3gram, option_tokens_3gram in examples_tokens:
    #     question_tokens.extend(question_token)
    #     option_tokens.extend(option_token)
    #     question_tokens_3grams.extend(question_tokens_3gram)
    #     option_tokens_3grams.extend(option_tokens_3gram)
    #     question_tokens_2grams.extend(question_tokens_2gram)
    #     option_tokens_2grams.extend(option_tokens_2gram)
    # counter_q = collections.Counter(question_tokens)
    # counter_c = collections.Counter(option_tokens)
    # counter_q2 = collections.Counter(question_tokens_2grams)
    # counter_c2 = collections.Counter(option_tokens_2grams)
    # counter_q3 = collections.Counter(question_tokens_3grams)
    # counter_c3 = collections.Counter(option_tokens_3grams)

    print('question and option srl done!')


def read_arc_ai2_retrieve(input_file, args):
    examples = []
    with open(input_file, 'r', encoding='utf-8') as fin:
        for line in fin.readlines():
            examples.append(json.loads(line.strip('\n')))
    return examples


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default='')
    parser.add_argument('--convert_srl', default=False)
    parser.add_argument('--srl_file', default='')
    # parser.add_argument('--output_file', default='', required=True)
    parser.add_argument('--threads', default=cpu_count())
    args = parser.parse_args()
    analyse(args.input_file, args)
