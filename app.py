from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from farasa.segmenter import FarasaSegmenter
import streamlit as st
import torch
import os

# Load model
model = AutoModelForQuestionAnswering.from_pretrained("checkpoint-2817")
tokenizer = AutoTokenizer.from_pretrained("checkpoint-2817")

segmenter = FarasaSegmenter(interactive=False)

# Streamlit interface
st.title("Arabic Question Answering")
st.write("أدخل سياقًا وسؤالًا بالعربية واحصل على الجواب.")

context = st.text_area("السياق", height=150)
question = st.text_input("السؤال")

if st.button("احصل على الجواب") and context and question:
    # Preprocess
    context_proc = segmenter.segment(context)
    question_proc = segmenter.segment(question)

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
    answer = segmenter.desegment(answer)

    st.success(f"الجواب: {answer}")
    print("Answer:", answer)