import pandas as pd
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall, context_entity_recall, answer_similarity, answer_correctness
from ragas import evaluate
from get_response import get_response_llm
import warnings
from datasets import Dataset
warnings.filterwarnings("ignore", category=FutureWarning)

df = pd.read_csv("output.csv",nrows=5)

questions = list(df['question'])
ground_truth = list(df['answer'])
responses = []
for each_que in questions:
    structured_response = get_response_llm(each_que)
    responses.append(structured_response)

dt = {
    "question": questions,
    "answer": responses,
    "ground_truth": ground_truth
}

dataset = Dataset.from_dict(dt)
score = evaluate(dataset,metrics=[answer_relevancy, answer_correctness])
score_df = score.to_pandas()
score_df = score_df[['answer_relevancy','answer_correctness']].mean(axis=0)
print(score_df)