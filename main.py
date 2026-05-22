import streamlit as st
# import functions from rag.py
from rag import (
    process_urls,process_codebase,generate_answer,generate_frd
)
st.title("MultiAgent Codebase RAG Assistant")
url1=st.sidebar.text_input("URL-1")
url2=st.sidebar.text_input("URL-2")
url3=st.sidebar.text_input("URL-3")
placeholder=st.empty()
process_url_button=st.sidebar.button("Process URLS")
# NEW: codebase folder input
# why: so user can also load a python project,not just URLS
st.sidebar.markdown("---")
codebase_path=st.sidebar.text_input("Codebase Folder Path",placeholder="e.g. ./my_project")
process_code_button=st.sidebar.button("Process Codebase")

#-------------original urlprocessing same as your tutor's code----------------
if process_url_button:
    urls=[url for url in (url1,url2,url3) if url!=""]
    if len(urls)==0:
        placeholder.text("Please enter atleast one URL")
    else:
        for status in process_urls(urls):
            placeholder.text(status)   
# new : codebase processing button handler
# why: same pattern as your tutor's URL button,just calls process codebase instead
if process_code_button:
    if not codebase_path:
        placeholder.text("Please enter a folder path")
    else:
        for status in process_codebase(codebase_path):
            placeholder.text(status)
# new: tabs to separate to separate Q&A from FRD generation
# why: keeps the UI clean; tutor's original Q&A still works in tab1    
tab1,tab2=st.tabs(["Ask a question","Generate Documentation"])  
# ---tab1 : same as yourtutor's question input----
with tab1:
    query=st.text_input("Question")
    if query:
        try:
            answer,sources=generate_answer(query)
            st.header("Answer")
            st.write(answer)
            if sources:
                st.subheader("Sources")
                for source in sources.split("\n"):
                    st.write(source)
        except RuntimeError as e:
            placeholder.text("Please click on the process URL button or process codebase button first") 
# new : tab2-FRD(functional requirement document) generation
# why : problem statement requires auto generating documentation
with tab2:
    module_input=st.text_input(
        "Which file to document? (leave blank for everything)",
        placeholder="e.g. calculator.py"
        ) 
    frd_button=st.button("Generate FRD")  
    if frd_button:
        try:
            frd_text=generate_frd(module_name=module_input if module_input else None) 
            st.markdown(frd_text)
            # download button so user can save the document
            st.download_button("Download as .md",
                               frd_text,file_name="frd.md")
        except RuntimeError:
            st.warning("Please process a codebase first.")             




                              