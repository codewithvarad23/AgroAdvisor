
"""Chatbot code"""

import streamlit as st

def main():
    def chat_bot():
        try:
            st.chat_input('enter your message....')
            st.chat_message('human')
            st.chat_message('ai')
        except Exception as e:
            st.error(f'error : {e}')



    def api_page():
        st.write('this is API page')

    def sidebar():
        st.title('AgroAdvisor')
        page=st.sidebar.selectbox('select',['chatbot','API'])
        if page=='chatbot':
            chat_bot()
        elif page=='API':
            api_page()

    sidebar()

if __name__=="__main__":
    main()