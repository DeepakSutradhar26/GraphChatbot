import streamlit as st
from backend import ingest_pdf
from backend import chatbot, retrieve_all_threads, submit_async_task
from langchain_core.messages import HumanMessage, AIMessage
import uuid

# *************** Utility Functions ***************

def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    st.session_state['thread_id'] = generate_thread_id()
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    # Check if messages key exists in state values, return empty list if not
    return state.values.get("messages", [])

# *************** Streamlit App ***************

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()

if 'ingested_docs' not in st.session_state:
    st.session_state['ingested_docs'] = {}

add_thread(st.session_state['thread_id'])

thread_key = str(st.session_state['thread_id'])
thread_docs = st.session_state['ingested_docs'].setdefault(thread_key, {})
threads = st.session_state['chat_threads'][::-1]
selected_thread = None

# **************** Sidebar UI****************

st.sidebar.title("GraphChatbot")

if st.sidebar.button('New Chat', use_container_width=True):
    reset_chat()
    st.rerun()

if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"Using `{latest_doc.get('filename')}` "
        f"({latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages)"
    )
else:
    st.sidebar.info("No PDF indexed yet.")

uploaded_pdf = st.sidebar.file_uploader("Upload a PDF for this chat", type=["pdf"])
if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"`{uploaded_pdf.name}` already processed for this chat.")
    else:
        with st.sidebar.status("Indexing PDF…", expanded=True) as status_box:
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name,
            )
            thread_docs[uploaded_pdf.name] = summary
            status_box.update(label="✅ PDF indexed", state="complete", expanded=False)

st.sidebar.header('My Conversations')

for thread_id in st.session_state['chat_threads']:
    if st.sidebar.button(str(thread_id)):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []

        for message in messages:
            if isinstance(message, HumanMessage):
                temp_messages.append({'role': 'user', 'content': message.content})
            else:
                temp_messages.append({'role': 'assistant', 'content': message.content})

        st.session_state['message_history'] = temp_messages

# **************** Main Chat UI ****************

for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('Type Here...')

if user_input:

    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    CONFIG = {
        'configurable': {'thread_id': st.session_state['thread_id']},
        'metadata': {
            'thread_id': st.session_state['thread_id'],
        },
        'run_name': 'chat_turn'
    }

    with st.chat_message('assistant'): 
        import queue
        _q = queue.Queue()
        _SENTINEL = object()

        async def _ai_only_stream():
            try:
                async for message_chunk, _ in chatbot.astream(
                    {'messages': [HumanMessage(content=user_input)]},
                    CONFIG,
                    stream_mode='messages'
                ):
                    if isinstance(message_chunk, AIMessage):
                        _q.put(message_chunk.content)
            finally:
                _q.put(_SENTINEL)

        submit_async_task(_ai_only_stream())

        def _sync_stream():
            while True:
                item = _q.get()
                if item is _SENTINEL:
                    break
                yield item

        ai_message = st.write_stream(_sync_stream())

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})