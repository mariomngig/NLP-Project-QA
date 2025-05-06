from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from farasa.segmenter import FarasaSegmenter
import streamlit as st
import torch
import os
from arabert.preprocess import ArabertPreprocessor

model_name = "aubmindlab/bert-base-arabertv2"
arabert_prep = ArabertPreprocessor(model_name=model_name)
# Load model
model = AutoModelForQuestionAnswering.from_pretrained("MarioMamdouh121/arabic-qa-model")
tokenizer = AutoTokenizer.from_pretrained("MarioMamdouh121/arabic-qa-model")

# Streamlit interface
st.title("Arabic Question Answering")
st.write("أدخل سياقًا وسؤالًا بالعربية واحصل على الجواب.")

context = st.text_area("السياق", height=150)
question = st.text_input("السؤال")

if st.button("احصل على الجواب") and context and question:
    # Preprocess
    context_proc = arabert_prep.preprocess(context)
    question_proc = arabert_prep.preprocess(question)

    # Tokenize
    inputs = tokenizer(
        question_proc,
        context_proc,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)

    start_index = torch.argmax(outputs.start_logits)
    end_index = torch.argmax(outputs.end_logits)
    answer_tokens = inputs["input_ids"][0][start_index : end_index + 1]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    st.success(f"الجواب: {answer}")
