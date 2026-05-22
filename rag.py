from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import ast
import os
load_dotenv()
# variables (same as your tutor's)
CHUNK_SIZE=1000
EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
VECTORESTORE_DIR=Path(__file__).parent/"resources/vectorstore"
COLLECTION_NAME="web_assistant"
llm=None
vector_store=None
def initialize_components():
    # same as your tutor's not touched
    global llm,vector_store
    if llm is None:
        llm=ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.9,
            max_tokens=1000
        )
    if vector_store is None:
        embeddings=HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={ "trust_remote_code": True }
        ) 
        vector_store=Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=VECTORESTORE_DIR
        ) 
# NEW FUNCTION 1: process_codebase
# why: you tutor's process_urls loads from website.
# this does the same thing but reads .py files instead
# it uses ast to split at function/class level
# (smarter than splitting every 1000 characters)
def process_codebase(folder_path):
    yield "Initializing components..."           
    initialize_components()
    yield "Reseting vector store..."
    vector_store.reset_collection()
    yield "Reading .py files from folder..."
    docs=[]
    for root,_,files in os.walk(folder_path):
        for file in files:
            if not file.endswith(".py"):
                continue
            full_path=os.path.join(root,file)
            with open(full_path,"r",encoding="utf-8") as f:
                source=f.read()
            # new: use ast to find every function and class in the file
            # why : splitting code at function/class level keeps meaning intact
            try:
                tree=ast.parse(source)
                lines=source.split('\n')
                for node in ast.walk(tree):
                    if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
                        print("Processing:", file)
                        print("Added:", node.name) 
                        chunk_text="\n".join(lines[node.lineno-1:node.end_lineno])
                        # store chunk as a LangChain Document(same format your tutor uses)
                        from langchain_core.documents import Document
                        docs.append(Document(
                            page_content=chunk_text,
                            metadata={
                                "source":full_path,
                                "type":"class" if isinstance(node,ast.ClassDef) else "function",
                                "name":node.name
                            }
                        ))
            except Exception as e:
                print("ERROR:", e)  
                          
            #except SyntaxError:
                # if file has error,load it as one whole chunk
            #    from langchain_core.documents import Document
            #    docs.append(Document(page_content=source,metadata={"source":file}))
    yield f"found {len(docs)} function/classes."

    if not docs:
        yield "No functions/classes found."
        return

    yield "Adding to ChromaDB..."

    ids = [str(uuid4()) for _ in range(len(docs))]

    vector_store.add_documents(docs, ids=ids)        
    yield "Codebase indexed successfully!"

def process_urls(urls):
    # same as your tutor's - not touched
    yield "Initializing components..."
    initialize_components()
    yield "Resetting the vector store..."
    vector_store.reset_collection()
    yield "loading data from URLS"
    loader=UnstructuredURLLoader(urls=urls)
    documents=loader.load()
    yield "Splitting data into small chunks..."
    splitter=RecursiveCharacterTextSplitter(
        separators=["\n\n","\n","."," "],
        chunk_size=CHUNK_SIZE
    )
    docs=splitter.split_documents(documents)
    yield "Adding docs into ChromaDB..."
    ids=[str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(docs,ids=ids)
    yield "Vector store successfully updated!"

#--------------------------------------------------
# NEW FUNCTION 2: retrieval_agent
# why your tutor's generate answer does everything in a one function
# we just pull out the " find relevant chunks" part out and name it.
# thus is what makes it a multi-agent system-
# each agent has one job
#----------------------------------------------------
def retrieval_agent(query):
    retriever=vector_store.as_retriever()
    docs=retriever.invoke(query)
    return docs # just returns the chunks nothing else

#---------------------------------------------------
# new function 3: reasoning_agent
# why: looks at what retrieval_agent found and
# and figures out which part are actually relevant
# in your tutor's code was hidden inside generate answer.
#-------------------------------------------------
def reasoning_agent(query,docs):
    content="\n\n".join([doc.page_content for doc in docs])
    prompt=f"""
            You are a code analysis assistant.
            A user asked: "{query}"
            Look at the code below and identify what is relevant to the question.
            Be brief.

            CODE:
            {content}
            What is related and why:"""
    response=llm.invoke(prompt)
    return response.content.strip(),content # return analysis+raw content
#----------------------------------------------------
# new function 4: response_agent
# why : takes the reasoning and writes the final clean answer
# this was also hidden inside the generate answer before
def response_agent(query,analysis,content,docs):
    prompt=f"""
            You are a helpful assistant.
            Answer the question ONLY using the content given below.
            If the answer is not present,say "I don't know". Don't hallucinate.
            ANALYSIS OF RELEVANT PARTS:
            {analysis}
            FULL CONTENT:
            {content}
            question:
            {query}
            """
    response=llm.invoke(prompt)
    sources="\n".join(list(set([doc.metadata.get("source","") for doc in docs])))
    return response.content.strip(),sources
# --------------------------------------------------------------------
# generate_answer: now calls the 3 agents in order
# WHY: same name as your tutor's function so main.py still works.
#      Only difference: internally it calls the 3 agents.
# -------------------------------------------------------
def generate_answer(query):
    if not vector_store:
        raise RuntimeError("Vector database is empty")
    # agent 1 find relevant chunks
    docs=retrieval_agent(query)
    # agent2: reason about what's relevant
    analysis,content=reasoning_agent(query,docs)
    # agent 3: write the final answer
    answer,sources=response_agent(query,analysis,content,docs)

    return answer,sources

# -------------------------------------------------------
# NEW FUNCTION 5: generate_frd
# WHY: problem statement asks to auto-generate documentation.
#      FRD = plain English description of what the code does.
#      We retrieve relevant chunks then ask LLM to document them.
# -------------------------------------------------------

def generate_frd(module_name=None):
    if not vector_store:
        raise RuntimeError("Vector database is empty")
    # find chunks for the requested module (or everything)
    query=f"all functions and classes in {module_name}" if module_name else "all code"
    docs=retrieval_agent(query)
    # if a specific file was requested, filter to only that file
    if module_name:
        docs=[d for d in docs if module_name in d.metadata.get("source","")]
        content="\n\n".join([doc.page_content for doc in docs])
        prompt=f"""
                You are a technical writer.Read this code and write a simple document.
                Use this structure:
                ## Purpose
                What does this code do overall?
                ## FUNCTIONS / CLASSES
                For each one:name,what it does,inputs,output.
                ## How to USE
                A short example.
                CODE:
                {content}
                Write the document now:
                """
        response=llm.invoke(prompt)
        return response.content.strip()





                           







