TEMPLATE = """###Task Description: An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given. 
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general. 
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric. 
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)" 
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
Translate the following text from {source_language} to {target_language}: {source}

###Response to evaluate:
{hypothesis}

###Reference Answer (Score 5):
{reference}

###Score Rubrics: [Accuracy, Fluency, Style]
Score 1: The translation contains major errors that significantly alter the meaning of the source text. It is barely comprehensible and reads like a poor machine translation. The style is completely inconsistent with the source text.
Score 2: The translation has several inaccuracies that affect the overall meaning. It is difficult to read and understand, with frequent awkward phrasings. The style only occasionally matches the source text.
Score 3: The translation is mostly accurate but has some minor errors that don't significantly alter the meaning. It is generally understandable but lacks natural flow in some parts. The style is somewhat consistent with the source text.
Score 4: The translation is accurate with only a few negligible errors. It reads naturally for the most part, with occasional minor awkwardness. The style largely matches that of the source text.
Score 5: The translation is highly accurate, conveying the full meaning of the source text. It reads as fluently as an original text in the target language. The style perfectly captures the tone and register of the source text.

###Feedback:
"""