from scipy.spatial.distance import cosine
from transformers import T5ForConditionalGeneration, AutoTokenizer

def calculate_cosine_similarity(vector1, vector2):
    return 1 - cosine(vector1, vector2)


def cosine_similarity(query_emb, db_items, top_pick=5):
    similarities = []
    for filename, text, vector in db_items:
        similarity = calculate_cosine_similarity(query_emb, vector)
        similarities.append(similarity)

    top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:top_pick]
    top_items = [db_items[i][1] for i in top_indices]
    return top_items


# QA model
def model_qa(question, refs, model_card='google/flan-t5-base',CoT=True):
    """

    :param question: string
    :param refs:
    :param model_card:
    :return:
    """
    tokenizer = AutoTokenizer.from_pretrained(model_card)
    model = T5ForConditionalGeneration.from_pretrained(model_card)
    reference_doc = "\n".join(refs)
    if CoT:
        input_text = "context: {} Answer the following question by reasoning step_by_step: {}".format(reference_doc,question)

    else:
        input_text = "context: {} question: {}".format(reference_doc, question)
    inputs = tokenizer.encode(input_text, return_tensors='pt')

    # outputs = model.generate(inputs)
    # answer = tokenizer.decode(outputs[0])
    outputs = model.generate(inputs, max_length = 512, no_repeat_ngram_size=4)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer