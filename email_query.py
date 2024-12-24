from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from llama_index.core.prompts.prompts import SimpleInputPrompt
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
import os
import base64
import email
import torch
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sentence_transformers import SentenceTransformer

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def authenticate_gmail():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    service = build('gmail', 'v1', credentials=creds)
    return service

def fetch_emails_and_attachments(service, user_id='me'):
    results = service.users().messages().list(userId=user_id).execute()
    messages = results.get('messages', [])
    save_path = "gmail_data"
    os.makedirs(save_path, exist_ok=True)

    email_data = []

    if not messages:
        print("No messages found.")
        raise ValueError("No messages found.")

    for message in messages:
        msg = service.users().messages().get(userId=user_id, id=message['id']).execute()
        payload = msg.get('payload', {})
        headers = payload.get('headers', [])
        
        subject = None
        sender = None
        
       
        for header in headers:
            if header.get('name') == 'Subject':
                subject = header.get('value', None)  
            if header.get('name') == 'From':
                sender = header.get('value', None)  
        
        if not subject:
            subject = "No Subject"
        if not sender:
            sender = "Unknown Sender"

        body = None
        if 'parts' in payload:
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain':
                    body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                elif part['mimeType'] == 'text/html':
                    body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
        
        email_filename = f"{subject[:50].replace(' ', '_')}_{message['id']}.txt"  # Ensure unique filename
        email_file_path = os.path.join(save_path, email_filename)
        with open(email_file_path, 'w') as f:
            f.write(f"Subject: {subject}\n")
            f.write(f"From: {sender}\n\n")
            f.write(f"Body:\n{body}\n")

        email_data.append({
            'subject': subject,
            'sender': sender,
            'body': body
        })

        for part in payload.get('parts', []):
            if part['filename']:
                attachment_id = part['body']['attachmentId']
                attachment = service.users().messages().attachments().get(
                    userId=user_id, messageId=message['id'], id=attachment_id
                ).execute()
                file_data = base64.urlsafe_b64decode(attachment['data'])
                file_path = os.path.join(save_path, part['filename'])
                with open(file_path, 'wb') as f:
                    f.write(file_data)

    return email_data, save_path

def setup_llama_index(data_path):
    system_prompt ="""<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
"""

    query_wrapper_prompt = SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")
    llm = HuggingFaceLLM(
        context_window=2048,
        max_new_tokens=256,
        query_wrapper_prompt=query_wrapper_prompt,
        model_name="StabilityAI/stablelm-tuned-alpha-3b",
        tokenizer_name="StabilityAI/stablelm-tuned-alpha-3b",
        device_map="auto",
        model_kwargs={"torch_dtype": torch.float16},
    )
    Settings.llm = llm
    
    hf_embed = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
    Settings.embed_model = hf_embed 
    
    documents = SimpleDirectoryReader(data_path).load_data()
    vector_index = VectorStoreIndex.from_documents(documents, settings=Settings)
    return vector_index.as_query_engine()


service = authenticate_gmail()
email_data, files_path = fetch_emails_and_attachments(service)

# for email_info in email_data:
#     print(f"Subject: {email_info['subject']}")
#     print(f"From: {email_info['sender']}")
#     print(f"Body: {email_info['body']}")
#     print("-" * 50)

query_engine = setup_llama_index(files_path)

user_query = input("Ask your question: ")
response = query_engine.query(user_query)
print(response)