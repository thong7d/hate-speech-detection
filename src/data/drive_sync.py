import os
import pickle
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# If modifying these SCOPES, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/drive.file']

def get_service():
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    return build('drive', 'v3', credentials=creds)

def create_folder(service, folder_name, parent_id=None):
    file_metadata = {
        'name': folder_name,
        'mimeType': 'application/vnd.google-apps.folder'
    }
    if parent_id:
        file_metadata['parents'] = [parent_id]
    
    # Check if folder already exists
    query = f"name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder'"
    if parent_id:
        query += f" and '{parent_id}' in parents"
    results = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    files = results.get('files', [])
    
    if files:
        return files[0]['id']
    
    file = service.files().create(body=file_metadata, fields='id').execute()
    return file.get('id')

def upload_file(service, file_path, folder_id):
    file_name = os.path.basename(file_path)
    file_metadata = {
        'name': file_name,
        'parents': [folder_id]
    }
    
    # Check if file already exists to update or skip (simple version: always upload new/overwrite)
    media = MediaFileUpload(file_path, resumable=True)
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    print(f"Uploaded {file_name} (ID: {file.get('id')})")

def upload_directory(service, local_dir, drive_folder_id):
    for item in os.listdir(local_dir):
        item_path = os.path.join(local_dir, item)
        if os.path.isfile(item_path):
            upload_file(service, item_path, drive_folder_id)
        elif os.path.isdir(item_path):
            new_drive_folder_id = create_folder(service, item, drive_folder_id)
            upload_directory(service, item_path, new_drive_folder_id)

def main():
    if not os.path.exists('credentials.json'):
        print("Error: 'credentials.json' not found. Download it from Google Cloud Console (OAuth 2.0 Client ID).")
        return

    service = get_service()
    
    root_folder_id = create_folder(service, 'vihsd_backup')
    print(f"Root Folder ID: {root_folder_id}")
    
    for folder in ['data', 'results']:
        if os.path.exists(folder):
            print(f"Uploading {folder}...")
            drive_subfolder_id = create_folder(service, folder, root_folder_id)
            upload_directory(service, folder, drive_subfolder_id)

if __name__ == '__main__':
    main()
