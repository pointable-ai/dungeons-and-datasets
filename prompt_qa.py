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

from prompt_constants import (
    QUESTION_GENERATION_PROMPT_LLAMA,
    QUESTION_EVAL_PROMPT_OPENAI,
    QUESTION_EVAL_PROMPT_LLAMA,
    QUESTION_GENERATION_PROMPT_OPENAI,
)

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER.setLevel(logging.INFO)

MAX_CONTEXT = 4096
LOCAL_LLM = None

DELIMITER_ENGLISH = {
    "|": "pipes",
    ";": "semicolons",
    ",": "commas",
    "\\t": "tabs",
    "\\s": "spaces",
}

QuestionSet = namedtuple(
    "QuestionSet",
    ["question", "answer", "generation_prompt", "ground_truth", "original_context"],
)
EvalSet = namedtuple(
    "EvalSet",
    [
        "question",
        "answer",
        "generation_prompt",
        "ground_truth",
        "original_context",
        "evaluation",
    ],
)


def format_generated_questions(
    generated_question_sets: Iterable,
    ground_truth: str,
    original_context: str,
    delimiter: str = "|",
    is_include_header: bool = False,
) -> List[QuestionSet]:
    if is_include_header:
        generated_question_sets = generated_question_sets[1:]

    question_set_list = []
    for question_set in generated_question_sets:
        question_set_components = question_set.split(delimiter)
        try:
            question = QuestionSet(
                *question_set_components, ground_truth, original_context
            )
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


def generate_question_set_response_openai(
    context: str,
    num_of_questions: int,
    delimiter: str = ";",
    llm_model: str = "gpt-3.5-turbo",
) -> Dict:
    # TODO: there's a bug when num_of_questions is 1; the llm will respond with a 1 "set" of questions, which is more than 1 question

    # If delimiter isn't in our pool of know delimiters, we just go with what's provided
    string_delimiter = DELIMITER_ENGLISH.get(delimiter, delimiter)
    question_prompt = QUESTION_GENERATION_PROMPT_OPENAI.substitute(
        context_str=context,
        num=num_of_questions,
        delimiter=string_delimiter,
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
    response_dict["original_context"] = context
    response_dict["generation_prompt"] = question_prompt
    return response_dict


def generate_question_set_response_llama(
    context: str,
    num_of_questions: int,
    delimiter: str = ";",
) -> Dict:
    if LOCAL_LLM is None:
        raise RuntimeError("Missing local llm client")
    # TODO: there's a bug when num_of_questions is 1; the llm will respond with a 1 "set" of questions, which is more than 1 question

    # If delimiter isn't in our pool of know delimiters, we just go with what's provided
    string_delimiter = DELIMITER_ENGLISH.get(delimiter, delimiter)
    question_prompt = QUESTION_GENERATION_PROMPT_LLAMA.substitute(
        context_str=context,
        num=num_of_questions,
        delimiter=string_delimiter,
    )
    # TODO: this should be hotswappable and merged with remote version
    response = LOCAL_LLM(
        question_prompt,
        max_tokens=MAX_CONTEXT,
        echo=False,
        temperature=0,
    )
    response["original_context"] = context
    response["generation_prompt"] = question_prompt
    return response


def generate_question_set_response_openai(
    question_set: QuestionSet,
    llm_model: str = "gpt-3.5-turbo",
) -> Dict:
    # TODO: there's a bug when num_of_questions is 1; the llm will respond with a 1 "set" of questions, which is more than 1 question

    question_eval_prompt = QUESTION_EVAL_PROMPT_OPENAI.substitute(
        query_str=question_set.question, context_str=question_set.original_context
    )
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a proofreader checking exam question validity.",
            },
            {"role": "user", "content": question_eval_prompt},
        ],
    )

    response_dict = response.to_dict()
    response_dict["generation_prompt"] = question_eval_prompt
    return response_dict


def generate_question_eval_response_llama(question_set: QuestionSet) -> Dict:
    if LOCAL_LLM is None:
        raise RuntimeError("Missing local llm client")

    # TODO: there's a bug when num_of_questions is 1; the llm will respond with a 1 "set" of questions, which is more than 1 question

    question_eval_prompt = QUESTION_EVAL_PROMPT_LLAMA.substitute(
        query_str=question_set.question, context_str=question_set.original_context
    )

    # TODO: this should be hotswappable and merged with remote version
    response = LOCAL_LLM(
        question_eval_prompt,
        max_tokens=MAX_CONTEXT,
        echo=False,
        temperature=0,
    )

    response["generation_prompt"] = question_eval_prompt
    return response_dict


def generate_questions_and_save_response(args):
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
    for monster_info in monster_infos[args.start_index : args.end_index]:
        monster_name = monster_info.get("monster_name")
        LOGGER.info(f"Generating questions for: {monster_name}")
        if args.local_llm:
            response = generate_question_set_response_llama(
                context=monster_info,
                num_of_questions=args.num_to_generate,
                delimiter=args.delimiter,
            )
        else:
            response = generate_question_set_response_openai(
                context=monster_info,
                num_of_questions=args.num_to_generate,
                delimiter=args.delimiter,
            )

        filename = f"{args.output_dir}{monster_name}_{response['id']}"
        filepath = Path(f"{filename}.json")
        with open(filepath, "w") as fp:
            json.dump(response, fp)


def parse_and_aggregate_generated_questions(args):
    output_dir = Path(args.output_dir)
    if not output_dir.is_dir():
        LOGGER.error(
            f"{output_dir} is not an existing directory. Please create it before trying to put files into it."
        )
        exit()

    total_question_set = load_generated_questions(args, output_dir)

    # Output the file with generated and formatted questions
    with open(args.output_file, "w") as fp:
        # TODO: potentially add another arg for output delimiter
        writer = csv.writer(fp, delimiter=args.delimiter)
        # Writes the _fields as header of the doc
        writer.writerow(QuestionSet._fields)
        for question in total_question_set:
            writer.writerow(
                [
                    question.question,
                    question.answer,
                    question.generation_prompt,
                    question.ground_truth,
                    question.original_context,
                ]
            )


def parse_and_aggregate_evaluation(args):
    output_dir = Path(args.output_dir)
    if not output_dir.is_dir():
        LOGGER.error(
            f"{output_dir} is not an existing directory. Please create it before trying to put files into it."
        )
        exit()

    total_eval_set = load_generated_evals(args, output_dir)

    # Output the file with generated and formatted questions
    with open(args.output_file, "w") as fp:
        # TODO: potentially add another arg for output delimiter
        writer = csv.writer(fp, delimiter=args.delimiter)
        # Writes the _fields as header of the doc
        writer.writerow(EvalSet._fields)
        for evaluation in total_eval_set:
            writer.writerow(
                [
                    evaluation.question,
                    evaluation.answer,
                    evaluation.generation_prompt,
                    evaluation.ground_truth,
                    evaluation.original_context,
                    evaluation.evaluation,
                ]
            )


def evaluate_questions_and_save_response(args):
    # Where generated responses from openai will go
    output_dir = Path(args.output_dir)
    if not output_dir.is_dir():
        LOGGER.error(
            f"{output_dir} is not an existing directory. Please create it before trying to put files into it."
        )
        exit()

    total_question_set = load_generated_questions(output_dir)

    for question_set in total_question_set[args.start_index : args.end_index]:
        LOGGER.info(f"Evaluating questions for: {question_set.ground_truth}")
        if args.local_llm:
            response = generate_question_eval_response_llama(question_set)
        else:
            response = generate_question_eval_response_openai(question_set)

        # Write contexts to the prompt response
        response["question"] = question_set.question
        response["answer"] = question_set.answer
        response["original_context"] = question_set.original_context

        filepath = _get_eval_filename(Path(question_set.ground_truth), response["id"])
        with open(filepath, "w") as fp:
            json.dump(response, fp)


def _get_eval_filename(filepath: Path, uuid: str) -> Path:
    filename = filepath.stem + f"_eval_{uuid}.json"
    return filepath.parent.joinpath(filename)


def _get_content_openai(prompt_response: Dict) -> str:
    choices = prompt_response["choices"]
    choice = choices[0]
    message = choice["message"]
    return message["content"]


def _get_content_llama(prompt_response: Dict) -> str:
    choices = prompt_response["choices"]
    choice = choices[0]
    return choice["text"]


def load_generated_questions(
    args, output_dir: Path, glob_pattern: str = "*[!_eval].json"
):
    """Format and create the list of questions from file into memory.
    glob_pattern default ignores the eval files"""

    total_question_set = []
    for filename in output_dir.glob(glob_pattern):
        with open(filename, "r") as fp:
            prompt_response = json.load(fp)
        try:
            if args.local_llm:
                content = _get_content_llama(prompt_response)
            else:
                content = _get_content_openai(prompt_response)
        except (KeyError, TypeError) as e:
            LOGGER.warning(f"Failed processing {filename}. See error:\n", exc_info=True)
            continue

        try:
            original_context = prompt_response["original_context"]
        except (KeyError, TypeError) as e:
            LOGGER.warning(f"Failed processing {filename}. See error:\n", exc_info=True)
            continue

        content_list = content.split("\n")
        question_set_list = format_generated_questions(
            generated_question_sets=content_list[1:],
            ground_truth=filename,
            delimiter=args.delimiter,
            original_context=original_context,
        )
        total_question_set.extend(question_set_list)

    return total_question_set


def load_generated_evals(args, output_dir: Path, glob_pattern: str = "*_eval*.json"):
    """Format and create the list of questions from file into memory.
    glob_pattern default ignores the eval files"""

    total_eval_set = []
    for filename in output_dir.glob(glob_pattern):
        with open(filename, "r") as fp:
            prompt_response = json.load(fp)
        try:
            if args.local_llm:
                evaluation = _get_content_llama(prompt_response)
            else:
                evaluation = _get_content_openai(prompt_response)

            # Prior are content set ups, rest of these are the vars that we will use
            evaluation = message["content"]
            original_context = prompt_response["original_context"]
            generation_prompt = prompt_response["generation_prompt"]
            question = prompt_response["question"]
            answer = prompt_response["answer"]

        except (KeyError, TypeError) as e:
            LOGGER.warning(f"Failed processing {filename}. See error:\n", exc_info=True)

        eval_set = EvalSet(
            question, answer, generation_prompt, filename, original_context, evaluation
        )
        total_eval_set.append(eval_set)

    return total_eval_set


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
        "--eval",
        action="store_true",
        help="Whether to perform question evaluation. Default is to skip question evaluation.",
        default=False,
    )
    parser.add_argument(
        "--eval_parse",
        action="store_true",
        help="Whether to parse evaluations. Default is to skip parsing.",
        default=False,
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Whether to perform question generation. Default is to skip question generation.",
        default=False,
    )
    parser.add_argument(
        "--generate_parse",
        action="store_true",
        help="Whether to parse generated questions. Default is to skip parsing.",
        default=False,
    )
    parser.add_argument(
        "--local_llm",
        action="store_true",
        help="Use llama locally instead of openai.",
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


# This is "throwaway" code for the D&D dataset
if __name__ == "__main__":
    args = parse_args()

    if args.local_llm:
        from llama_cpp import Llama

        GGUF_MODEL_PATH = "./model/nous-hermes-llama2-13b-q4.gguf"

        LOCAL_LLM = Llama(
            model_path=GGUF_MODEL_PATH,
            n_gpu_layers=999,
            n_ctx=MAX_CONTEXT,
            verbose=False,
            use_mmap=False,
        )

    if args.generate:
        generate_questions_and_save_response(args)

    if args.generate_parse:
        parse_and_aggregate_generated_questions(args)

    if args.eval:
        evaluate_questions_and_save_response(args)

    if args.eval_parse:
        parse_and_aggregate_evaluation(args)
