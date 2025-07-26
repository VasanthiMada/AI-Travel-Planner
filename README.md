# 🌍 Enhanced Travel Itinerary Planner

A powerful AI-based travel planning assistant that creates personalized multi-day travel itineraries using LangChain, LangGraph, Groq (LLM), Gradio, and live APIs (weather + events). 🚀

## 🔥 Features

✅ Multi-day, city-specific itinerary generator  
✅ LLM-powered activity suggestions using Groq's LLaMA3-70B  
✅ Live weather info from OpenWeather API  
✅ Local events from Ticketmaster API  
✅ Budget-based cost breakdown  
✅ Dietary & accessibility consideration  
✅ Fully interactive Gradio interface  
✅ Modular LangGraph workflow structure

---

## 🧠 Tech Stack

- [LangChain](https://www.langchain.com/)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [Groq API](https://console.groq.com/)
- [Gradio UI](https://www.gradio.app/)
- OpenWeatherMap API  
- Ticketmaster Discovery API

---

## 🚀 Installation

```bash
git clone https://github.com/yourusername/enhanced-travel-planner.git
cd enhanced-travel-planner
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

---

## 🔐 Environment Setup

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key
OPENWEATHER_API_KEY=your_openweather_api_key
TICKETMASTER_API_KEY=your_ticketmaster_api_key
```

---

## 📦 Dependencies

Your `requirements.txt` should include:

```txt
langchain
langgraph
langchain-groq==0.3.6
groq==0.30.0
openai
pydantic
gradio
requests
python-dotenv
```

---

## 🧪 Run the App

```bash
python travel.py
```

The app will start on:  
`http://localhost:7860` or a public shareable URL via Gradio.

---

## 🛠️ How It Works

The app runs a **LangGraph workflow** that:

1. Collects input and validates
2. Fetches weather using OpenWeather API
3. Pulls local events via Ticketmaster
4. Estimates costs based on group size and duration
5. Uses `ChatGroq` (LLaMA3-70B) to generate full itinerary

---

## 🧠 AI Model Used

The app uses `llama3-70b-8192` via Groq's blazing-fast inference.

You can also try:
```python
model_name="llama3-8b-8192"
```

---

## 🧩 Example Inputs

- **City**: Paris  
- **Interests**: museums, cafes, art  
- **Duration**: 4  
- **Budget**: Mid-range ($$)  
- **Group Size**: 2  
- **Accommodation**: Boutique Hotel  
- **Transport**: Walking  
- **Dietary Restrictions**: vegetarian  
- **Accessibility Needs**: wheelchair accessible  
- **Travel Dates**: June 10–13, 2024

---


## ❤️ Credits

- Built with love by [Your Name]
- Groq for lightning-fast LLMs
- LangChain & LangGraph for workflow management
- Gradio for the intuitive interface
