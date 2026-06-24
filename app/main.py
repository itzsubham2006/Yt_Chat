import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, InvalidVideoId, NoTranscriptFound, VideoUnavailable
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')


llm = HuggingFaceEndpoint(
    repo_id= "deepseek-ai/DeepSeek-V4-Flash", task = 'text-generation'
)

model = ChatHuggingFace(llm=llm)



video_id = "LPZh9BOjkQs"
ytt_api = YouTubeTranscriptApi()


try:
    transcript_list = ytt_api.fetch(video_id, languages=['en'])
    
    transcript = " ".join(chunk.text for chunk in transcript_list)
    
except(TranscriptsDisabled, InvalidVideoId, NoTranscriptFound, VideoUnavailable):
    print("Transcript not available or invalid video ID.")
    transcript = None
    
except AttributeError as e:
    print("Transcript API method not found:", e)
    