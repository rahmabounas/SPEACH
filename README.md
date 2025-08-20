# Speech-Enabled Chatbot

A Streamlit application that combines speech recognition and chatbot algorithms to create a speech-enabled chatbot. The chatbot can take voice input from the user, transcribe it into text using the speech recognition algorithm, and then use the chatbot algorithm to generate a response.

## Features

- Text-based chatbot functionality
- Speech recognition for voice input
- Multi-language support (11 languages)
- Enhanced error handling
- Session state management for conversation history

## Supported Languages

The speech recognition supports the following languages:
- English (US and UK)
- French
- Spanish
- German
- Italian
- Portuguese
- Russian
- Japanese
- Chinese (Mandarin)
- Arabic

## Requirements

- Python 3.7+
- NLTK
- Streamlit
- SpeechRecognition

## Installation

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Download the required NLTK data:
   ```
   python deploy_nltk_download.py
   ```

3. Run the application:
   ```
   streamlit run main.py
   ```

## Deployment

For deployment platforms like Streamlit Cloud, the application will automatically download the required NLTK data on first run.

## Usage

1. Upload a text corpus file in the sidebar (optional)
2. Select your preferred language for speech recognition
3. Type your message in the text input or click "Speak instead" to use voice input
4. The chatbot will respond based on the content of the corpus

## Error Handling

The application provides detailed error messages for speech recognition failures:
- Timeout errors with suggestions to try speaking again
- Recognition errors with suggestions to speak more clearly
- Service errors with suggestions to check internet connection

## Contributing

Feel free to fork this repository and submit pull requests with improvements.