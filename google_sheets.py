import os
import logging
from google.oauth2 import service_account
from googleapiclient.discovery import build
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GoogleSheetsService:
    def __init__(self):
        try:
            # Get credentials from environment variables
            required_vars = [
                'GOOGLE_PROJECT_ID',
                'GOOGLE_PRIVATE_KEY_ID',
                'GOOGLE_PRIVATE_KEY',
                'GOOGLE_CLIENT_EMAIL',
                'GOOGLE_CLIENT_ID',
                'GOOGLE_CLIENT_X509_CERT_URL'
            ]
            
            # Validate all required variables are present
            missing_vars = [var for var in required_vars if not os.getenv(var)]
            if missing_vars:
                raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
            
            # Log the presence of each variable (but not their values)
            for var in required_vars:
                logger.info(f"{var} is {'set' if os.getenv(var) else 'not set'}")
            
            # Get private key and ensure it's properly formatted
            private_key = os.getenv('GOOGLE_PRIVATE_KEY')
            if not private_key:
                raise ValueError("GOOGLE_PRIVATE_KEY is not set")
            
            # Replace escaped newlines with actual newlines
            private_key = private_key.replace('\\n', '\n')
            
            credentials = {
                "type": "service_account",
                "project_id": os.getenv('GOOGLE_PROJECT_ID'),
                "private_key_id": os.getenv('GOOGLE_PRIVATE_KEY_ID'),
                "private_key": private_key,
                "client_email": os.getenv('GOOGLE_CLIENT_EMAIL'),
                "client_id": os.getenv('GOOGLE_CLIENT_ID'),
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "client_x509_cert_url": os.getenv('GOOGLE_CLIENT_X509_CERT_URL')
            }
            
            logger.info("Initializing Google Sheets service...")
            
            # Create credentials object
            creds = service_account.Credentials.from_service_account_info(
                credentials,
                scopes=['https://www.googleapis.com/auth/spreadsheets']
            )
            
            # Build the service
            self.service = build('sheets', 'v4', credentials=creds)
            logger.info("Google Sheets service initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Google Sheets service: {str(e)}")
            raise

    def log_victory(self, model_name, score, ip_address, secret_key):
        """Log a victory against DennisBot to Google Sheets"""
        try:
            # Verify the secret key
            expected_key = os.getenv('VICTORY_LOG_SECRET')
            if not expected_key:
                logger.error("VICTORY_LOG_SECRET environment variable is not set")
                return
                
            if secret_key != expected_key:
                logger.error("Invalid secret key")
                return
            
            # Get the spreadsheet ID from environment variable
            spreadsheet_id = os.getenv('GOOGLE_SHEETS_ID')
            if not spreadsheet_id:
                raise ValueError("GOOGLE_SHEETS_ID environment variable not set")
            
            # First, check if the model name already exists in the sheet
            result = self.service.spreadsheets().values().get(
                spreadsheetId=spreadsheet_id,
                range='Sheet1!A:A'  # Check only the model name column
            ).execute()
            
            values = result.get('values', [])
            if values:
                # Check if model_name exists in the first column
                existing_models = [row[0] for row in values]
                if model_name in existing_models:
                    logger.info(f"Model {model_name} already exists in the sheet. Skipping duplicate entry.")
                    return
            
            # If model doesn't exist, proceed with logging
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            values = [[model_name, score, timestamp, ip_address]]
            
            body = {
                'values': values
            }
            
            result = self.service.spreadsheets().values().append(
                spreadsheetId=spreadsheet_id,
                range='Sheet1!A:D',
                valueInputOption='USER_ENTERED',
                body=body
            ).execute()
            
            logger.info(f"Successfully logged victory for {model_name} to Google Sheets")
            return result
            
        except Exception as e:
            logger.error(f"Error logging victory to Google Sheets: {str(e)}")
            raise

# Create a singleton instance
sheets_manager = GoogleSheetsService() 