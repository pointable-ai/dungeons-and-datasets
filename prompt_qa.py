from collections import namedtuple
from string import Template
from typing import Dict, Iterable, List
import json
import logging

import openai

LOGGER = logging.getLogger(__name__)


QUESTION_EVAL_PROMPT = Template("""Please tell if a given piece of information is supported by the context.
You need to answer with either YES or NO.
Answer YES if any of the context supports the information, even if most of the context is unrelated. Some examples are provided below.

Information: Apple pie is generally double-crusted.
Context: An apple pie is a fruit pie in which the principal filling ingredient is apples.
Apple pie is often served with whipped cream, ice cream ('apple pie à la mode'), custard or cheddar cheese.
It is generally double-crusted, with pastry both above and below the filling; the upper crust may be solid or
latticed (woven of crosswise strips).
Answer: YES
Information: Apple pies tastes bad.
Context: An apple pie is a fruit pie in which the principal filling ingredient is apples.
Apple pie is often served with whipped cream, ice cream ('apple pie à la mode'), custard or cheddar cheese.
It is generally double-crusted, with pastry both above and below the filling; the upper crust may be solid or
latticed (woven of crosswise strips).
Answer: NO

Information: $query_str
Context: $context_str
Answer:""")


QA_EVAL_PROMPT = Template("""
    Your task is to evaluate if the response for the query is in line with the context information provided.
    You have two options to answer. Either YES/NO.
    Answer - YES, if the response for the query is in line with context information otherwise NO.
    Query and Response:
    $query_str

    Context:
    $context_str

    Answer:"""
)

# Note that a TSV is generated
QUESTION_GENERATION_PROMPT = Template("""Context information is below.

---------------------
$context_str
---------------------

Given the context information and not prior knowledge.
Generate only questions based on the below query.

You are a Teacher/Professor. Your task is to create $num questions and an answer key for an quiz/examination that is not multiple choice.
The questions should be diverse in nature across the document. Restrict the questions to the context information provided.

Please organize this into a tsv format with columns for the Question, the Answer, and the Information used to arrive at the answer.""")


SAMPLE = 'Question\tAnswer\tInformation Used\n1. What special abilities does the ancient deep crow possess?\tThe ancient deep crow has the special abilities "Magic Resistance" and "Shadow Stealth".\t- Special abilities listed in the context information.\n2. What is the saving throw DC for the ancient deep crow\'s Shadow Caw ability?\tThe saving throw DC for the ancient deep crow\'s Shadow Caw ability is 17.\t- Shadow Caw ability details in the context information.\n3. What condition can the ancient deep crow inflict on a target with its mandibles? The ancient deep crow can inflict the "restrained" condition on a target with its mandibles.\t- Mandibles ability details in the context information.'


QuestionSet = namedtuple("QuestionSet", ["question", "answer", "context"])

def format_generated_questions(generated_question_sets: Iterable, deliminter: str="\t", is_include_header: bool=False) -> List[QuestionSet]:
    if is_include_header:
        generated_question_sets = generated_question_sets[1:]

    question_set_list = []
    for question_set in generated_question_sets:
        question_set_components = question_set.split(deliminter)
        try:
            question = QuestionSet(*question_set_components)
        except TypeError as e:
            LOGGER.warning("Question cannot be processed. This is likely a missing delimiter in response. "
                f"\nQuestion:\n{question_set_components}")
            continue
        question_set_list.append(question)

    return question_set_list

def generate_question_set_response(context: str, num_of_questions: int, llm_model: str="gpt-3.5-turbo") -> Dict:
    question_prompt = QUESTION_GENERATION_PROMPT.substitute(context_str=context, num=num_of_questions)
    response = openai.ChatCompletion.create(
        model=llm_model,
        messages=[{"role": "user", "content": question_prompt}]
    )
    response_dict = response.to_dict()
    response_dict["generation_prompt"] = question_prompt
    return response_dict


# ==== This is "throwaway" code for the D&D dataset ====
if __name__ == '__main__':
    with open("monster_text.json", "r") as fp:
        monster_infos = json.load(fp)

    for monster_name, monster_info in monster_infos.items():
        response = generate_question_set_response(context=monster_info, num_of_questions=2)
        with open(f"generated_questions/{monster_name}.json", "w") as fp:
            json.dump(response, fp)
        break

