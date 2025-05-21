from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_ollama import OllamaLLM,OllamaEmbeddings
from langchain_core.prompts import MessagesPlaceholder,PromptTemplate
from langchain.chains.summarize import load_summarize_chain


import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
llm=ChatGroq(groq_api_key=os.environ['GROQ_API_KEY'],model='llama3-8b-8192')
loader=PyMuPDFLoader('redshift-gsg.pdf')
docs=loader.load()
text_splitter=RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=200)
splitter=text_splitter.split_documents(docs)

##STUFF SUMMARISATION
prompt_template="""Summarise the pdf in 100 words- given input is {text}"""
prompt=PromptTemplate(input_variables=['text'],template=prompt_template)
chain=load_summarize_chain(llm=llm,chain_type='stuff',prompt=prompt,verbose=True)
result=chain.run(docs)


##MAP-REDUCE
chunk_prompt="""Summarize the given text : {text} . Summary :"""
prompt_mr=PromptTemplate(input_variables=['text'],template=chunk_prompt)
final_prompt=""" Provide final summary of the entire document {text}. Divide the summary into sections with bullet points"""
final_template=PromptTemplate(input_variables=['text'],template=final_prompt)
mr_chain=load_summarize_chain(llm, chain_type='map_reduce',map_prompt=prompt_mr,combine_prompt=final_template,verbose=True)
mr_chain.run(splitter)


##REFINE SUMMARISATION
refine_chain=load_summarize_chain(llm=llm,chain_type='refine',verbose=True)
refine_chain.run(splitter)
