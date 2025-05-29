import os
import streamlit as st
import validators
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.document_loaders import UnstructuredURLLoader

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Top spoken languages for user selection
top_languages = [
    ("English", "en"),
    ("Mandarin Chinese", "zh"),
    ("Hindi", "hi"),
    ("Spanish", "es"),
    ("French", "fr"),
    ("Standard Arabic", "ar"),
    ("Bengali", "bn"),
    ("Portuguese", "pt"),
    ("Russian", "ru"),
    ("Japanese", "ja"),
]

# Initialize LLM from Groq
def get_llm(model_name):
    return ChatGroq(model=model_name)

# Function to fetch and chunk YouTube transcript
def summarize_youtube_transcript(url):
    try:
        video_id = url.split("v=")[-1].split("&")[0] if "v=" in url else url.split("/")[-1]
        transcript = YouTubeTranscriptApi.get_transcript(video_id=video_id, languages=["hi", "en", "en-US"])
        full_text = " ".join([entry["text"] for entry in transcript])

        # Split the transcript into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_text(full_text)
        docs = [Document(page_content=chunk) for chunk in chunks]
        return docs
    except Exception as e:
        st.error(f"Transcript not available: {e}")
        return None

# Main App
st.set_page_config(page_title="Universal Summarizer", page_icon="🧠")
st.title("🧠 Summarize YouTube or Website Content")

# Input section
url = st.text_input("Enter a YouTube or Website URL:")
model_name = st.selectbox("Choose Groq model:", ["gemma2-9b-it","Llama-3.1-8b-Instant","Llama3-8b-8192","Llama-3.3-70b-Versatile", "DeepSeek-R1-Distill-Llama-70B"])
language_name = st.selectbox("Select language for summary:", [name for name, code in top_languages])
language_code = dict(top_languages)[language_name]

if st.button("Generate Summary"):
    if not url.strip():
        st.error("Please enter a valid URL.")
    elif not validators.url(url):
        st.error("The provided string is not a valid URL.")
    else:
        try:
            with st.spinner("Processing..."):
                llm = get_llm(model_name)

                # Load documents based on source
                if "youtube.com" in url or "youtube" in url:
                    docs = summarize_youtube_transcript(url)
                    if not docs:
                        st.stop()
                else:
                    loader = UnstructuredURLLoader(
                        urls=[url],
                        ssl_verify=False,
                        headers={
                            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) "
                                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                                          "Chrome/116.0.0.0 Safari/537.36"
                        }
                    )
                    raw_docs = loader.load()
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                    docs = text_splitter.split_documents(raw_docs)

                # Prompts
                map_prompt = PromptTemplate(
                    template="""
You are a helpful assistant. Extract the key points from the following content. Respond in {language}.Avoid writing any internal thoughts.

TEXT:
{text}
""",
                    input_variables=["text", "language"]
                )

                combine_prompt = PromptTemplate(
                    template="""
You are a helpful assistant. Using the key points below, write a clear and concise summary in {language} such that it include all the relavant points.Avoid writing any internal thoughts.
KEY POINTS:
{text}

Summary:
""",
                    input_variables=["text", "language"]
                )

                # Build summarization chain
                chain = load_summarize_chain(
                    llm=llm,
                    chain_type="map_reduce",
                    map_prompt=map_prompt,
                    combine_prompt=combine_prompt,
                    verbose=False
                )

                summary = chain.run({"input_documents": docs, "language": language_name})
                st.subheader(f"📄 Summary ({language_name}):")
                st.success(summary)

        except Exception as e:
            st.exception(e)
