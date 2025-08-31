@echo off
echo Starting Retrieval QA Chatbot...
echo.
echo Make sure you have set your OpenAI API key:
echo $env:OPENAI_API_KEY="your-api-key-here"
echo.
echo Starting Streamlit application...
streamlit run Chatbot.py
pause
