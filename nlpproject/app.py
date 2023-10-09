import streamlit as st
import numpy as np 
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import PyPDF2

#@st.cache(allow_output_mutation=True)
@st.cache_resource
def get_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained("PankajNk/toxichub")
    return tokenizer,model

tokenizer,model = get_model()
d ={
    1:'Toxic',
    0:'Non Toxic'
}

radio_choose = st.radio("choose the method to input text ",["Text","file input"])
if radio_choose == 'Text':
    user_input = st.text_area('Enter Test to be Analyze')
    button = st.button("Analyze")
    if user_input  and button:
        test_sample = tokenizer([user_input], padding=True, truncation=True, max_length=512,return_tensors='pt')
        outputs = model(**test_sample)
        #predication = torch.nn.functional.softmax(outputs.logits, dim = 1)
        st.write("logits: ", outputs.logits)
        y_predication = np.argmax(outputs.logits.detach().numpy(), axis =1)
        st.write("Predication",d[y_predication[0]])
elif radio_choose == 'file input':
    uploaded_file = st.file_uploader("Choose PDF file", type="pdf")
    button = st.button("Analyze")
    if uploaded_file and button:
        # Read the PDF file
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        # Extract the content
        content = ""
        for page in range(len(pdf_reader.pages)):
            content += pdf_reader.pages[page].extract_text()
        # Display the content
        st.write(content)
        test_sample = tokenizer([content], padding=True, truncation=True, max_length=512,return_tensors='pt')
        outputs = model(**test_sample)
        #predication = torch.nn.functional.softmax(outputs.logits, dim = 1)
        st.write("logits: ", outputs.logits)
        y_predication = np.argmax(outputs.logits.detach().numpy(), axis =1)
        st.write("Predication",d[y_predication[0]])










