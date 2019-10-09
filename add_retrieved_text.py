"""
Script to retrieve HITS for each answer choice and question
USAGE:
 python scripts/add_retrieved_text.py qa_file output_file

JSONL format of files
 1. qa_file:
 {
    "id":"Mercury_SC_415702",
    "question": {
       "stem":"George wants to warm his hands quickly by rubbing them. Which skin surface will
               produce the most heat?",
       "choices":[
                  {"text":"dry palms","label":"A"},
                  {"text":"wet palms","label":"B"},
                  {"text":"palms covered with oil","label":"C"},
                  {"text":"palms covered with lotion","label":"D"}
                 ]
    },
    "answerKey":"A"
  }

 2. output_file:
  {
   "id": "Mercury_SC_415702",
   "question": {
      "stem": "..."
      "choice": {"text": "dry palms", "label": "A"},
      "support": {
                "text": "...",
                "type": "sentence",
                "ir_pos": 0,
                "ir_score": 2.2,
            }
    },
     "answerKey":"A"
  }
  ...
  {
   "id": "Mercury_SC_415702",
   "question": {
      "stem": "..."
      "choice": {"text":"palms covered with lotion","label":"D"}
      "support": {
                "text": "...",
                "type": "sentence",
                "ir_pos": 1,
                "ir_score": 1.8,
            }
     "answerKey":"A"
  }
"""

import json
import os
import sys
from typing import List, Dict

from allennlp.common.util import JsonDict
from tqdm._tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))))
from es_search import EsSearch, EsHit

MAX_HITS = 20
from multiprocessing import cpu_count
from functools import partial
from multiprocessing import Pool

thread = cpu_count()


def init():
    global es_search
    es_search = EsSearch(max_hits_per_choice=MAX_HITS, max_hits_retrieved=200)

def add_retrieved_text(qa_file, output_file):
    with open(output_file, 'w') as output_handle, open(qa_file, 'r') as qa_handle:
        print("Writing to {} from {}".format(output_file, qa_file))
        line_tqdm = tqdm(qa_handle, dynamic_ncols=True)
        json_lines = []
        for line in line_tqdm:
            json_lines.append(json.loads(line))
        output_lines = []
        print('start multiple processing')
        with Pool(thread, initializer=init) as p:
            annotate = partial(add_hits_to_qajson)
            output_lines = list(tqdm(p.imap(annotate, json_lines, chunksize=32), desc='add hits to qajson', total=len(json_lines)))
        # output_dict, hits = add_hits_to_qajson(json_line)
        num_hits = 0
        for output_dict, hits in output_lines:
            output_handle.write(json.dumps(output_dict) + "\n")
            num_hits += hits
            line_tqdm.set_postfix(hits=num_hits)


def add_hits_to_qajson(qa_json: JsonDict):
    # print(qa_json['id'])
    question_text = qa_json["question"]["stem"]
    choices = [choice["text"] for choice in qa_json["question"]["choices"]]
    hits_per_choice = es_search.get_hits_for_question(question_text, choices)
    hits=0
    for i in range(len(qa_json['question']['choices'])):
        choice = qa_json['question']['choices'][i]
        choice['para'] = [t.text for t in hits_per_choice[choice["text"]][:MAX_HITS]]
        hits += len(choice['para'])
        choice['para'] = ' '.join(choice['para'])
        qa_json['question']['choices'][i] = choice
    hits = hits / 4
    return qa_json, int(hits)


def filter_hits_across_choices(hits_per_choice: Dict[str, List[EsHit]],
                               top_k: int):
    """
    Filter the hits from all answer choices(in-place) to the top_k hits based on the hit score
    """
    # collect ir scores
    ir_scores = [hit.score for hits in hits_per_choice.values() for hit in hits]
    # if more than top_k hits were found
    if len(ir_scores) > top_k:
        # find the score of the top_kth hit
        min_score = sorted(ir_scores, reverse=True)[top_k - 1]
        # filter hits below this score
        for choice, hits in hits_per_choice.items():
            hits[:] = [hit for hit in hits if hit.score >= min_score]


# Create the output json dictionary from the QA file json, answer choice json and retrieved HIT
def create_output_dict(qa_json: JsonDict, choice_json: JsonDict, hit: EsHit):
    output_dict = {
        "id": qa_json["id"],
        "question": {
            "stem": qa_json["question"]["stem"],
            "choice": choice_json,
            "support": {
                "text": hit.text,
                "type": hit.type,
                "ir_pos": hit.position,
                "ir_score": hit.score,
            }
        },
        "answerKey": qa_json["answerKey"]
    }
    return output_dict


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise ValueError("Provide at least two arguments: "
                         "question-answer json file, output file name")
    add_retrieved_text(sys.argv[1], sys.argv[2])