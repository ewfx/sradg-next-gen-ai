import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import openai
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

import numpy as np

# --------------- Config -------------------
st.set_page_config(page_title="Anomaly Detector App", layout="wide")

# Set OpenAI API key (you can use environment variable instead)
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

# Hugging Face Model Setup (Llama2)
@st.cache_resource
def load_huggingface_model():
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
   # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    #model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", device_map="auto")
    return tokenizer, model

tokenizer, hf_model = load_huggingface_model()

# ---------------- Helper Functions ----------------


def detect_anomalies(df, contamination=0.1):
   # iso = IsolationForest(contamination=contamination, random_state=42)
    #df['anomaly'] = iso.fit_predict(df.select_dtypes(include=['float64', 'int64']))

   # Identify Numeric and Categorical Features
    numeric_features = ['Company','Account','AU','GL Balance', 'IHub Balance','Balance Difference']
    categorical_features = ['Secondary Account','Primary Account','Currency','Match Status']
    
    # Convert Categorical Features
    label_encoders = {}
    for col in categorical_features:
        df[col] = df[col].astype(str)  # Ensure string type
        label_encoders[col] = LabelEncoder()
        df[col] = label_encoders[col].fit_transform(df[col])

    # Generate Embeddings for Text Features (if any)
    text_column = "Primary Account"  # Example: transaction description
    openai_embedder = OpenAIEmbeddings(model="text-embedding-ada-002")
    #df_new=df
    df["text_embedding"] = df[text_column].apply(lambda x: openai_embedder.embed_query(x) if isinstance(x, str) else np.zeros(1536))
    #df_new["text_embedding"] = df[text_column].apply(lambda x: openai_embedder.embed_query(x) if isinstance(x, str) else np.zeros(1536))

    # Convert Text Embeddings into Separate Features
    text_embedding_matrix = np.array(df["text_embedding"].tolist())
    df_embeddings = pd.DataFrame(text_embedding_matrix, columns=[f"text_emb_{i}" for i in range(text_embedding_matrix.shape[1])])

    # Combine All Features
    df_final = pd.concat([df[numeric_features], df[categorical_features], df_embeddings], axis=1)
    #df_final = pd.concat([df[numeric_features], df[categorical_features]], axis=1)


    # Apply Isolation Forest model for anomaly detection
    model = IsolationForest(contamination=contamination, random_state=42) 
    model.fit(df_final)

    # Predict anomalies (1 = normal, -1 = anomaly)
    df['anomaly'] = model.predict(df_final) 
    # Decode values after training
    
    for col in categorical_features:
        df[col] = label_encoders[col].inverse_transform(df[col])

    return df

def query_openai(prompt):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
                    {"role": "system","content": "You are an AI Assistant."},
                    {"role": "user","content": prompt}],
        max_tokens=200,
        temperature=0
    )
    return response.choices[0].message.content.strip()
#response.choices[0].message['content'].strip()

def query_huggingface(prompt):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hf_model.to(device)  # Move model to the correct device

    inputs = tokenizer(prompt, return_tensors="pt").to(device)  # Ensure inputs match
    outputs = hf_model.generate(**inputs, max_new_tokens=200)

    #inputs = tokenizer(prompt, return_tensors="pt").to('cpu')
    print("input : ") 
    print(inputs) 
    #outputs = hf_model.generate(**inputs, max_new_tokens=200)
    print("output : ") 
    print(outputs) 
    return tokenizer.decode(outputs[0], skip_special_tokens=False)

# ----------------- Streamlit UI ------------------

st.title("🔍 General Ledger (GL) vs IHub Reconcilliation")

# Function to load Excel data
def load_excel_data(file):
 try:
    df = pd.read_excel(file)
    return df
 except Exception as e:
    st.error(f"Error loading Excel file: {e}")
 return None

# Function to load csv data
def load_csv_data(file):
 try:
    df = pd.read_csv(file)
    return df
 except Exception as e:
    st.error(f"Error loading csv file: {e}")
 return None


# File upload
uploaded_file = st.file_uploader("Upload Excel File", type=["csv", "xlsx"])
#uploaded_file = st.file_uploader("Upload your CSV,xlsx file", type=["xlsx"])
if uploaded_file is not None:
            st.write(f"Processing File Name: {uploaded_file.name}")
            try:
                # Handle CSV
                if uploaded_file.name.endswith('.csv'):
                    df = load_csv_data(uploaded_file)
                    if df is not None:
                        st.write("Excel Data Preview:")
                        #st.dataframe(df.head())
                    
                # Handle Excel
                elif uploaded_file.name.endswith('.xlsx'):
                    df = load_excel_data(uploaded_file)
                    if df is not None:
                        st.write("Excel Data Preview:")
                        #st.dataframe(df.head())

                else:
                    st.error("Unsupported file type!")

            except pd.errors.EmptyDataError:
                st.error("The file is empty or not formatted correctly.")

            except pd.errors.ParserError:
                st.error("Failed to parse the file. It might be corrupted or improperly formatted.")

            except UnicodeDecodeError:
                st.error("Encoding error! Unable to decode the file content properly.")

            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")



            #if uploaded_file:
            #df = pd.read_excel(uploaded_file)
            st.subheader("📄 Uploaded Data")
            st.dataframe(df.head())

            # Detect Anomalies
            st.subheader("🚨 Anomaly Detection")
            contamination = st.slider("Set contamination rate:", 0.01, 0.5, 0.1, 0.01)
            df = detect_anomalies(df, contamination)
            # Convert Data to LangChain Documents for Similarity Search
            docs = [Document(page_content=str(row.to_dict())) for _, row in df.iterrows()]
            openai_embedder = OpenAIEmbeddings(model="text-embedding-ada-002")
            vector_store = FAISS.from_documents(docs, openai_embedder)

            anomalies = df[df['anomaly'] == -1]
            st.write(f"Detected {len(anomalies)} anomalies.")
            st.dataframe(anomalies.drop(columns=["text_embedding"]))
            #st.dataframe(anomalies)

            # LLM Analysis
            st.subheader("⏬ Expand to see anomalies")

            for idx, row in anomalies.iterrows():
                    row_context = row.to_dict()
                    row_text = ', '.join([f"{k}={v}" for k, v in row_context.items() if k != 'anomaly'])

                    with st.expander(f"Anomaly at index {idx} for Account : {row_context['Account']}"):
                        st.write(f"**Row Data:** {row_text}")

                        # Chat with OpenAI
                        st.header("Chat with AI Agent 🤖 ")
                        # User question input
                        user_message = st.text_input("Ask a question about your data:",key=f"input{idx}")
                        if user_message:
                            query = str(row.to_dict())
                            print(query)
                            similar_cases = vector_store.similarity_search(query, k=2)

                            st.write(f"Your question was : {user_message}")
                            # Setup LLM
                        
                            # Prompt Template
                       
                            prompt =f"""
                                    Analyze the following transaction for anomalies:\n{row_text}\nSimilar past cases:\n{similar_cases}"
                                    
                                    Conditions:

                                    - If the Match Status is break and  
                                    - If the anomaly score is -1 and If the Balance Difference is between -0.5 and 1.0, say it's normal.  
                                    - If the anomaly score is -1 and If the balance difference is positive.
                                    - If the anomaly score is -1 and If the balance difference is negative.
                                    
                                    Your task:
                                    - Identify unusual patterns, anomalies, or outliers.
                                    - Explain why these are considered anomalies.
                                    - Suggest possible reasons.

                                    Provide clear bullet-point analysis.

                                    User message: {user_message}  
                                    """
                            # OpenAI GPT
                            with st.spinner("Analyzing ..."):
                                gpt_output = query_openai(prompt)
                                st.write("**GPT-3.5 Analysis:**")
                                st.info(gpt_output)

                            # Hugging Face Model
                           # llama_prompt = f"The following data might be anomalous: {row_text}. Do you agree? Why?"
                            #llama_output = query_huggingface(llama_prompt)
                            #st.write("**Llama2 Validation:**")
                            #print(llama_output)
                            #st.success(llama_output)

