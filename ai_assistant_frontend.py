import win32com.client
import speech_recognition as sr
import webbrowser
import google.generativeai as genai
import datetime
import os

    # Load Gemini API key from environment variable
os.environ["GEMINI_API_KEY"] = "AIzaSyA0jX1JDZD7Tkhgm4crgO08bAjG9KFBUYc"  # Replace this with your secure API key

    # Configure the Gemini API
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

    # Create a model instance
model = genai.GenerativeModel('gemini-2.0-flash')

    # Chat history storage
chatStr = ""

def chat(query):
    """Handles the chat interaction with the AI model."""
    global chatStr
    print(f"👤 User: {query}")  # Display user input

    response = model.generate_content(query)
    ai_response = response.text.strip()

    # Store conversation history
    chatStr += f"👤 User: {query}\n🤖 AI: {ai_response}\n"

    # Display the response
    print(f"🤖 AI: {ai_response}")

    say(ai_response)

def ai(prompt):
    """Handles AI-based responses and saves output."""
    global chatStr
    print(f"🔍 Processing AI query: {prompt}")

    try:
        response = model.generate_content(prompt)
        ai_response = response.text.strip()

        # Store the AI response in chat history
        chatStr += f"👤 User: {prompt}\n🤖 AI: {ai_response}\n"

        print(f"🤖 AI Response: {ai_response}")

        # Ensure the "Openai" directory exists
        os.makedirs("Openai", exist_ok=True)

        # Create a safe filename using a timestamp
        filename = f"Openai/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"

        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"Prompt: {prompt}\nResponse:\n{ai_response}")

        say(ai_response)

    except Exception as e:
        print(f"⚠️ Error while calling AI: {e}")
        say("Sorry, I couldn't process the AI request.")

def say(text):
    """Speaks the provided text."""
    speaker = win32com.client.Dispatch('SAPI.SpVoice')
    speaker.Speak(text)

def takeCommand():
    """Captures user's voice command and converts it to text."""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Listening...")
        audio = r.listen(source)

        try:
            print("🔍 Recognizing...")
            query = r.recognize_google(audio, language='en-in')
            print(f"👤 User said: {query}")
            return query.lower()
        except Exception:
            print("❌ Could not understand. Please repeat.")
            return ""

if __name__ == "__main__":
    say("Hello! I am Sam, your AI assistant. How can I help you today?")

    while True:
        query = takeCommand()

        sites = {
           "youtube": "https://www.youtube.com",
            "wikipedia": "https://www.wikipedia.com",
           "google": "https://www.google.com"
        }
            # Open website commands
        for site in sites:
            if f"open {site}" in query:
                say(f"Opening {site}")
                webbrowser.open(sites[site])
                break  # Prevent multiple actions from happening

            # Exit command
            if "exit" in query or "stop" in query:
                say("Goodbye! Have a nice day.")
                exit()        

            # Get current time
            if "the time" in query:
                strfTime = datetime.datetime.now().strftime("%H:%M:%S")
                say(f"The time is {strfTime}")
                break

            # Process AI query
            if "using artificial intelligence" in query:
                ai(query)
            
            # Reset chat history
            if "reset chat" in query.lower():
                chatStr = ""
                say("Chat history has been reset.")

            # if "chat with me" in query.lower():
            #     if query !='':
            #         chat(query)  # All other queries go to chat with LLM