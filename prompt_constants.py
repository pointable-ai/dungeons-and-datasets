from string import Template

"""Generic prompt"""
GENERIC_EVAL_PROMPT = Template(
    """Please tell if a given piece of Information is supported by the context.
You need to answer with either YES or NO.
Answer YES if any of the context supports the Information, even if most of the context is unrelated.

Some examples are provided below.

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

Information:
---------------
$query_str
---------------

Context:
---------------
$context_str
---------------
Answer:"""
)


"""OpenAI prompts"""

QUESTION_EVAL_PROMPT_OPENAI = Template(
    """Please determine if a Question is supported by the context and fair to ask someone who does not have access to the exact context.
You need to answer with either YES or NO.
Answer YES if any of the context supports the Question, even if most of the context is unrelated. If the you
cannot answer with a YES or NO, the answer is NO.

Some examples are provided below.

Information: Is an Apple pie is generally double-crusted?
Context: An apple pie is a fruit pie in which the principal filling ingredient is apples.
Apple pie is often served with whipped cream, ice cream ('apple pie à la mode'), custard or cheddar cheese.
It is generally double-crusted, with pastry both above and below the filling; the upper crust may be solid or
latticed (woven of crosswise strips).
Answer: YES

Information: Does Apple pies taste bad?
Context: An apple pie is a fruit pie in which the principal filling ingredient is apples.
Apple pie is often served with whipped cream, ice cream ('apple pie à la mode'), custard or cheddar cheese.
It is generally double-crusted, with pastry both above and below the filling; the upper crust may be solid or
latticed (woven of crosswise strips).
Answer: NO

Information: Does the context say apple can be pies?
Context: An apple pie is a fruit pie in which the principal filling ingredient is apples.
Apple pie is often served with whipped cream, ice cream ('apple pie à la mode'), custard or cheddar cheese.
It is generally double-crusted, with pastry both above and below the filling; the upper crust may be solid or
latticed (woven of crosswise strips).
Answer: NO

Information:
---------------
$query_str
---------------

Context:
---------------
$context_str
---------------
Answer:"""
)

QUESTION_GENERATION_PROMPT_OPENAI = Template(
    """Context information is below.

---------------------
$context_str
---------------------

Given the context information and not prior knowledge. Generate only questions based on the below query.

Your task is to create $num questions and the answer key for a quiz/examination that is not multiple choice and there is one answer per one question.

The questions should follow the below constraints:
- The questions should be diverse in nature across the context provided.
- The questions are to be restricted to the context information provided.
- When referring to the context information, the question should add contextual clues make it as unambiguous as possible.
- Question and answer sets should have objective answers.

Please organize this into a delimited format using "$delimiter", with columns for the Question, the Answer, and the Information used to arrive at the answer."""
)


"""LLAMA prompts
https://huggingface.co/NousResearch/Nous-Hermes-Llama2-13b#prompt-format

### Instruction:
<prompt>

### Response:
<leave a newline blank for model to respond>


----OR----

### Instruction:
<prompt>

### Input:
<additional context>

### Response:
<leave a newline blank for model to respond>

"""

QUESTION_EVAL_PROMPT_LLAMA = Template(
    """### Instruction:
Please determine if a Question is supported by the context and fair to ask someone who does not have access to the exact context.
You need to answer with either YES or NO.
Answer YES if any of the context supports the Question, even if most of the context is unrelated. If the you
cannot answer with a YES or NO, the answer is NO.

Some examples are provided below.

Information: Is an Apple pie is generally double-crusted?
Context: An apple pie is a fruit pie in which the principal filling ingredient is apples.
Apple pie is often served with whipped cream, ice cream ('apple pie à la mode'), custard or cheddar cheese.
It is generally double-crusted, with pastry both above and below the filling; the upper crust may be solid or
latticed (woven of crosswise strips).
Answer: YES

Information: Does Apple pies taste bad?
Context: An apple pie is a fruit pie in which the principal filling ingredient is apples.
Apple pie is often served with whipped cream, ice cream ('apple pie à la mode'), custard or cheddar cheese.
It is generally double-crusted, with pastry both above and below the filling; the upper crust may be solid or
latticed (woven of crosswise strips).
Answer: NO


Information: Does the context say apple can be pies?
Context: An apple pie is a fruit pie in which the principal filling ingredient is apples.
Apple pie is often served with whipped cream, ice cream ('apple pie à la mode'), custard or cheddar cheese.
It is generally double-crusted, with pastry both above and below the filling; the upper crust may be solid or
latticed (woven of crosswise strips).
Answer: NO

### Input:
Information: $query_str
Context: $context_str

### Response:
"""
)

QUESTION_GENERATION_PROMPT_LLAMA = Template(
    """### Instruction:
Given the Input and not prior knowledge, generate questions based on the below query.

Your task is to create $num questions and the answer key for a quiz/examination that is not multiple choice and there is one answer per one question.

The questions should follow the below constraints:
- The questions should be diverse in nature across the context provided.
- The questions are to be restricted to the context information provided.
- When referring to the context information, the question should add contextual clues make it as unambiguous as possible.
- Question and answer sets should have objective answers.

Please organize this into a delimited format using "$delimiter", with columns for the Question, the Answer, and the Information used to arrive at the answer.

### Input:
$context_str

### Response:
"""
)
