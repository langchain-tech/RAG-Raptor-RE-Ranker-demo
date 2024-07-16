import streamlit as st
import random
from langchain_components.replier import *
import fitz

def display_pdf(pdf_path):
    try:
        pdf_document = fitz.open(pdf_path)
        num_pages = pdf_document.page_count
        st.sidebar.write(f"Total pages: {num_pages}")

        for page_num in range(num_pages):
            page = pdf_document.load_page(page_num)
            image = page.get_pixmap()
            st.sidebar.image(image.tobytes(), caption=f"Page {page_num + 1}", use_column_width=True)
            
    except Exception as e:
        st.sidebar.error(f"Error loading PDF: {e}")


def main():
    st.header('Interact with your complex PDF that includes text, images, tables, and graphs using Raptor and Re-Ranker RAG.')
    
    if "activate_chat" not in st.session_state:
        st.session_state.activate_chat = False

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        username = st.text_input("Please enter your name here")
        if st.button('Press Button to Start chat with your pdf..'):
            if "user_id" not in st.session_state:
                st.session_state.user_id = username
            
            if "session_id" not in st.session_state:
                random_number = random.randint(1, 1000000)
                st.session_state.session_id = str(random_number)

            if "vectorstore" not in st.session_state:
                collection_name="c1"
                st.session_state.vectorstore = get_vectorstore_from_postgres(collection_name)

            if "chain" not in st.session_state:
                st.session_state.chain = prepare_prompt_and_chain_with_history()

            st.session_state.activate_chat = True

        pdf_path = "data/fy2024.pdf"
        st.subheader("PDF Viewer")
        if st.button('Show PDF'):
            st.session_state.pdf_path = pdf_path
        if st.download_button(label="Download PDF", data=open(pdf_path, 'rb').read(), file_name=pdf_path.split("/")[-1]):
            pass

    if "pdf_path" in st.session_state:
        pdf_path = st.session_state.pdf_path
        display_pdf(pdf_path)

    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar = message['avatar']):
            st.markdown(message["content"])

    if st.session_state.activate_chat == True:
        if prompt := st.chat_input("Ask your question from the PDF? "):
            with st.chat_message("user", avatar = '👨🏻'):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user",  "avatar" :'👨🏻', "content": prompt})
            
            user_id = st.session_state.user_id
            session_id = st.session_state.session_id
            vectorstore = st.session_state.vectorstore
            chain = st.session_state.chain
            data=get_context_from_vectorstore(prompt, vectorstore)
            ai_msg =chain.invoke({"data": data, "input": prompt}, config={"configurable": {"user_id": user_id, "session_id": session_id}})
            cleaned_response=ai_msg.content
            with st.chat_message("assistant", avatar='🤖'):
                st.markdown(cleaned_response)
            st.session_state.messages.append({"role": "assistant",  "avatar" :'🤖', "content": cleaned_response})


if __name__ == '__main__':
    main()