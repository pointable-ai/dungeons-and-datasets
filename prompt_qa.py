"""
TODO: Consider using click to make this a full fledged CLI
"""

from collections import namedtuple
from pathlib import Path
from pprint import pformat
from string import Template
from typing import Dict, Iterable, List
import argparse
import csv
import json
import logging
import re

import openai

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER.setLevel(logging.DEBUG)


QUESTION_EVAL_PROMPT = Template(
    """Please tell if a given piece of information is supported by the context.
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
Answer:"""
)


QA_EVAL_PROMPT = Template(
    """
    Your task is to evaluate if the response for the query is in line with the context information provided.
    You have two options to answer. Either YES/NO.
    Answer - YES, if the response for the query is in line with context information otherwise NO.
    Query and Response:
    $query_str

    Context:
    $context_str

    Answer:"""
)

# Note that a pipe deliminted report is generated
QUESTION_GENERATION_PROMPT = Template(
    """Context information is below.

---------------------
$context_str
---------------------

Given the context information and not prior knowledge. Generate only questions based on the below query.

Your task is to create $num questions and the answer key for a quiz/examination where there is one answer per one question.

The questions should follow the below constraints:
- The questions should be diverse in nature across the context provided.
- The questions are to be restricted to the context information provided.
- When referring to the context information, the question should add contextual clues make it as unambiguous as possible.
- Question and answer sets should have objective answers.

Please organize this into a delimited format using "$delimiter", with columns for the Question, the Answer, and the Information used to arrive at the answer."""
)


SAMPLE = 'Question\tAnswer\tInformation Used\n1. What special abilities does the ancient deep crow possess?\tThe ancient deep crow has the special abilities "Magic Resistance" and "Shadow Stealth".\t- Special abilities listed in the context information.\n2. What is the saving throw DC for the ancient deep crow\'s Shadow Caw ability?\tThe saving throw DC for the ancient deep crow\'s Shadow Caw ability is 17.\t- Shadow Caw ability details in the context information.\n3. What condition can the ancient deep crow inflict on a target with its mandibles? The ancient deep crow can inflict the "restrained" condition on a target with its mandibles.\t- Mandibles ability details in the context information.'


QuestionSet = namedtuple(
    "QuestionSet", ["question", "answer", "context", "ground_truth"]
)


def format_generated_questions(
    generated_question_sets: Iterable,
    ground_truth: str,
    delimiter: str = "|",
    is_include_header: bool = False,
) -> List[QuestionSet]:
    if is_include_header:
        generated_question_sets = generated_question_sets[1:]

    question_set_list = []
    for question_set in generated_question_sets:
        question_set_components = question_set.split(delimiter)
        try:
            question = QuestionSet(*question_set_components, ground_truth)
        except TypeError as e:
            LOGGER.warning(
                f"""Question cannot be processed. This is likely a missing or extra delimiter in the response.
                Question: {pformat(question_set_components)}
                Error: {e}
                Ground Truth: {ground_truth}\n"""
            )
            continue

        # Simplistic filter to remove results without alphanumerics
        match = re.search("\w+", question.question)
        if not match:
            LOGGER.warning(
                f"Question not included in results since no alphanumerics were found: {question}"
            )
            continue

        question_set_list.append(question)

    return question_set_list


def generate_question_set_response(
    context: str,
    focus: str,
    num_of_questions: int,
    delimiter: str = ";",
    llm_model: str = "gpt-3.5-turbo",
) -> Dict:
    # TODO: there's a bug when num_of_questions is 1; the llm will respond with a 1 "set" of questions, which is more than 1 question
    question_prompt = QUESTION_GENERATION_PROMPT.substitute(
        context_str=context,
        focus_str=focus,
        num=num_of_questions,
        delimiter=delimiter,
    )
    # TODO: this should be hotswappable
    response = openai.ChatCompletion.create(
        model=llm_model,
        messages=[
            {
                "role": "system",
                "content": "You are a test writer/professor writing exam questions.",
            },
            {"role": "user", "content": question_prompt},
        ],
    )
    response_dict = response.to_dict()
    response_dict["generation_prompt"] = question_prompt
    return response_dict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generates questions for a set of knowledge base using an LLM."
    )
    parser.add_argument(
        "--delimiter",
        type=str,
        help="The delimiter to be used for prompting and formatting. Note: Semicolons do not work very well with question generation.",
        default="|",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Whether to perform question generation. Default is to skip question generation.",
        default=False,
    )
    parser.add_argument(
        "--input_json",
        type=str,
        help="The json to input for question generation.",
        default="monster_text.json",
    )
    parser.add_argument(
        "--num_to_generate",
        type=int,
        help="The number of questions to generate per prompt.",
        default=2,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="The dir to output the results of the requests.",
        default="generated_questions/",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="The final file to output the results of processing questions from requests.",
        default="total_question.csv",
    )
    parser.add_argument(
        "--parse",
        action="store_true",
        help="Whether to parse generated questions. Default is to skip question parsing.",
        default=False,
    )
    parser.add_argument(
        "--start_index",
        type=int,
        help="The index to start generating questions.",
        default=0,
    )
    parser.add_argument(
        "--end_index",
        type=int,
        help="The index to stop generating questions, non-inclusive. Default will include the last item.",
        default=-1,
    )

    args = parser.parse_args()
    return args


def generate_and_save_questions(args):
    # This expects monster_text in the old style (prior to langchain processing) where it's monster name as key and info as content
    input_json = Path(args.input_json)
    if not input_json.is_file:
        LOGGER.error(
            f"{input_json} is not a file. Please make sure you have the correct filepath or name."
        )
        exit()

    with open(args.input_json, "r") as fp:
        monster_infos = json.load(fp)

    # Where generated responses from openai will go
    output_dir = Path(args.output_dir)
    if not output_dir.is_dir():
        LOGGER.error(
            f"{output_dir} is not an existing directory. Please create it before trying to put files into it."
        )
        exit()

    # Question generation for each monster
    # This is the snipping/transformer code for prompting
    if args.generate:
        for monster_info in monster_infos[args.start_index : args.end_index]:
            monster_name = monster_info.get("monster_name")
            LOGGER.info(f"Generating questions for: {monster_name}")
            response = generate_question_set_response(
                context=monster_info,
                focus=monster_name,
                num_of_questions=args.num_to_generate,
                delimiter=args.delimiter,
            )

            filename = f"{args.output_dir}{monster_name}"
            filepath = Path(f"{filename}.json")
            while filepath.exists():
                filename += "_1"
                filepath = Path(f"{filename}.json")
            with open(filepath, "w") as fp:
                json.dump(response, fp)


def parse_and_aggregate_generated_questions(args):
    # Where generated responses from openai will go
    output_dir = Path(args.output_dir)
    if not output_dir.is_dir():
        LOGGER.error(
            f"{output_dir} is not an existing directory. Please create it before trying to put files into it."
        )
        exit()

    # Format and create the list of questions in memory to be outputted
    total_question_set = []
    for filename in output_dir.glob("*.json"):
        with open(filename, "r") as fp:
            prompt_response = json.load(fp)
        try:
            choices = prompt_response.get("choices")
            choice = choices[0]
            message = choice.get("message")
            content = message.get("content")
        except (KeyError, TypeError) as e:
            LOGGER.warning(f"Failed processing {filename}. See error:\n{e}")
        content_list = content.split("\n")
        question_set_list = format_generated_questions(
            generated_question_sets=content_list[1:],
            ground_truth=filename,
            delimiter=args.delimiter,
        )
        total_question_set.extend(question_set_list)

    # TODO: deal with this ad hoc header stuff in a better way
    header = content_list[0].split(args.delimiter)
    header.append("ground_truth")

    # Output the file with generated and formatted questions
    with open(args.output_file, "w") as fp:
        # TODO: potentially add another arg for output delimiter
        writer = csv.writer(fp, delimiter=args.delimiter)
        writer.writerow(header)
        for question in total_question_set:
            writer.writerow(
                [
                    question.question,
                    question.answer,
                    question.context,
                    question.ground_truth,
                ]
            )


# This is "throwaway" code for the D&D dataset
if __name__ == "__main__":
    args = parse_args()

    if args.generate:
        generate_and_save_questions(args)

    if args.parse:
        parse_and_aggregate_generated_questions(args)
