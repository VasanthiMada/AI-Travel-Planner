import os
import gradio as gr
import requests
from typing import TypedDict, Annotated, List, Dict
from datetime import datetime
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# === Enhanced State Definition ===
class EnhancedPlannerState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], "Conversation history"]
    city: str
    interests: List[str]
    duration: int
    budget: str
    group_size: int
    accommodation_type: str
    transport_mode: str
    dietary_restrictions: List[str]
    accessibility_needs: List[str]
    travel_dates: str
    itinerary: str
    weather_info: str
    local_events: List[str]
    estimated_costs: Dict[str, float]

# === Initialize LLM ===
llm = ChatGroq(
    temperature=0,
    model_name="llama3-70b-8192",
    api_key=os.environ.get("GROQ_API_KEY")
)

# === Prompts ===
itinerary_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert travel assistant. Create a comprehensive {duration}-day itinerary for {city} 
    based on the following details:
    - Interests: {interests}
    - Budget: {budget}
    - Group size: {group_size}
    - Accommodation: {accommodation_type}
    - Transportation: {transport_mode}
    - Dietary restrictions: {dietary_restrictions}
    - Accessibility needs: {accessibility_needs}
    - Weather info: {weather_info}
    - Local events: {local_events}
    
    Provide a detailed day-by-day itinerary with specific times, locations, estimated costs, and practical tips.
    Include backup indoor activities if weather is unfavorable."""),
    ("human", "Create a comprehensive travel itinerary."),
])

budget_prompt = ChatPromptTemplate.from_messages([
    ("system", """Calculate estimated costs for a {duration}-day trip to {city} for {group_size} people with {budget} budget.
    Break down costs into: accommodation, food, activities, transportation, and miscellaneous.
    Provide realistic estimates in USD."""),
    ("human", "Calculate travel costs."),
])

# === APIs ===
class WeatherAPI:
    def get_forecast(self, city: str, dates: str = None) -> str:
        try:
            api_key = os.getenv("OPENWEATHER_API_KEY")
            url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
            response = requests.get(url)
            data = response.json()
            weather_desc = data['weather'][0]['description'].capitalize()
            temp_min = data['main']['temp_min']
            temp_max = data['main']['temp_max']
            return f"Weather in {city}: {weather_desc}, {temp_min}–{temp_max}°C. Pack accordingly."
        except Exception as e:
            return f"Weather information unavailable. Error: {str(e)}"

class EventsAPI:
    def get_events(self, city: str, dates: str = None) -> List[str]:
        try:
            api_key = os.getenv("TICKETMASTER_API_KEY")
            url = f"https://app.ticketmaster.com/discovery/v2/events.json?apikey={api_key}&city={city}&size=5"
            response = requests.get(url)
            data = response.json()
            events = []
            if "_embedded" in data and "events" in data["_embedded"]:
                for event in data["_embedded"]["events"]:
                    name = event.get("name", "Unnamed Event")
                    venue = event["_embedded"]["venues"][0]["name"]
                    events.append(f"{name} at {venue}")
            return events or ["No major events found."]
        except Exception as e:
            return [f"Error fetching events: {str(e)}"]

class APIManager:
    def __init__(self):
        self.weather_api = WeatherAPI()
        self.events_api = EventsAPI()
    def fetch_all_data(self, city: str, dates: str = None):
        return {
            "weather": self.weather_api.get_forecast(city, dates),
            "events": self.events_api.get_events(city, dates)
        }

api_manager = APIManager()

# === Node Functions ===
def collect_basic_info(state: EnhancedPlannerState) -> EnhancedPlannerState:
    return state

def fetch_weather_data(state: EnhancedPlannerState) -> EnhancedPlannerState:
    weather_info = api_manager.weather_api.get_forecast(state['city'], state.get('travel_dates'))
    return {**state, "weather_info": weather_info}

def fetch_local_events(state: EnhancedPlannerState) -> EnhancedPlannerState:
    events = api_manager.events_api.get_events(state['city'], state.get('travel_dates'))
    return {**state, "local_events": events}

def calculate_budget(state: EnhancedPlannerState) -> EnhancedPlannerState:
    try:
        llm.invoke(budget_prompt.format_messages(
            duration=state['duration'],
            city=state['city'],
            group_size=state['group_size'],
            budget=state['budget']
        ))
        # You can parse LLM actual cost output if desired here
        cost_categories = {
            "accommodation": state['duration'] * state['group_size'] * 50,
            "food": state['duration'] * state['group_size'] * 30,
            "activities": state['duration'] * state['group_size'] * 40,
            "transport": state['group_size'] * 25,
            "miscellaneous": state['duration'] * state['group_size'] * 15
        }
        return {**state, "estimated_costs": cost_categories}
    except:
        return {**state, "estimated_costs": {}}

def create_comprehensive_itinerary(state: EnhancedPlannerState) -> EnhancedPlannerState:
    try:
        response = llm.invoke(itinerary_prompt.format_messages(
            duration=state['duration'],
            city=state['city'],
            interests=', '.join(state['interests']),
            budget=state['budget'],
            group_size=state['group_size'],
            accommodation_type=state['accommodation_type'],
            transport_mode=state['transport_mode'],
            dietary_restrictions=', '.join(state['dietary_restrictions']) if state['dietary_restrictions'] else 'None',
            accessibility_needs=', '.join(state['accessibility_needs']) if state['accessibility_needs'] else 'None',
            weather_info=state.get('weather_info', ''),
            local_events=', '.join(state.get('local_events', [])) if state.get('local_events') else 'None'
        ))
        return {
            **state,
            "itinerary": response.content,
            "messages": state['messages'] + [AIMessage(content=response.content)]
        }
    except Exception as e:
        return {
            **state,
            "itinerary": f"Error generating itinerary: {str(e)}"
        }

# === Build LangGraph Workflow ===
def build_workflow():
    workflow = StateGraph(EnhancedPlannerState)
    workflow.add_node("collect_info", collect_basic_info)
    workflow.add_node("fetch_weather", fetch_weather_data)
    workflow.add_node("fetch_events", fetch_local_events)
    workflow.add_node("calculate_costs", calculate_budget)
    workflow.add_node("create_itinerary", create_comprehensive_itinerary)
    workflow.set_entry_point("collect_info")
    workflow.add_edge("collect_info", "fetch_weather")
    workflow.add_edge("fetch_weather", "fetch_events")
    workflow.add_edge("fetch_events", "calculate_costs")
    workflow.add_edge("calculate_costs", "create_itinerary")
    workflow.add_edge("create_itinerary", END)
    return workflow.compile()

app = build_workflow()

# === Main Travel Planner Function ===
def enhanced_travel_planner(city, interests, duration, budget, group_size, 
                          accommodation, transport, dietary_restrictions, 
                          accessibility_needs, travel_dates):
    if not city or not interests:
        return "Please provide both city and interests.", {}, "", ""
    state = {
        "messages": [HumanMessage(content=f"Planning trip to {city}")],
        "city": city,
        "interests": [i.strip() for i in interests.split(',') if i.strip()],
        "duration": max(1, duration),
        "budget": budget,
        "group_size": max(1, group_size),
        "accommodation_type": accommodation,
        "transport_mode": transport,
        "dietary_restrictions": [d.strip() for d in dietary_restrictions.split(',') if d.strip()] if dietary_restrictions else [],
        "accessibility_needs": [a.strip() for a in accessibility_needs.split(',') if a.strip()] if accessibility_needs else [],
        "travel_dates": travel_dates if travel_dates else "",
        "itinerary": "",
        "weather_info": "",
        "local_events": [],
        "estimated_costs": {}
    }
    try:
        result = app.invoke(state)
        itinerary = result.get("itinerary", "Unable to generate itinerary")
        costs = result.get("estimated_costs", {})
        weather = result.get("weather_info", "Weather information unavailable")
        events = "\n".join(result.get("local_events", [])) or "No local events found"
        return itinerary, costs, weather, events
    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        return error_msg, {}, "", ""

# === Interface ===
def create_interface():
    interface = gr.Interface(
        fn=enhanced_travel_planner,
        inputs=[
            gr.Textbox(label="🗺️ Destination City", placeholder="e.g., Paris, Tokyo, New York", value=""),
            gr.Textbox(label="⭐ Interests", placeholder="e.g., museums, food, nightlife, nature, history", value=""),
            gr.Slider(minimum=1, maximum=14, value=3, step=1, label="📅 Duration (days)"),
            gr.Dropdown(choices=["Budget ($)", "Mid-range ($$)", "Luxury ($$$)"], label="💰 Budget Range", value="Mid-range ($$)"),
            gr.Number(value=2, label="👥 Group Size", minimum=1, maximum=20),
            gr.Dropdown(choices=["Hotel", "Hostel", "Airbnb", "Resort", "Boutique Hotel"], label="🏨 Accommodation Type", value="Hotel"),
            gr.Dropdown(choices=["Walking", "Public Transport", "Rental Car", "Taxi/Uber", "Mixed"], label="🚗 Transportation Mode", value="Public Transport"),
            gr.Textbox(label="🥦 Dietary Restrictions", placeholder="e.g., vegetarian, vegan, halal, gluten-free", value=""),
            gr.Textbox(label="♿ Accessibility Needs", placeholder="e.g., wheelchair accessible, hearing impaired", value=""),
            gr.Textbox(label="🗓️ Travel Dates", placeholder="e.g., March 15-20, 2024", value="")
        ],
        outputs=[
            gr.Textbox(label="📖 Detailed Itinerary", lines=15, max_lines=20, interactive=False),
            gr.JSON(label="💵 Cost Breakdown (USD)", interactive=False),
            gr.Textbox(label="🌤️ Weather Forecast", lines=3, interactive=False),
            gr.Textbox(label="🎪 Local Events", lines=5, interactive=False)
        ],
        title="🌏 <span style='color:#87ffe8'>Enhanced Travel Itinerary Planner</span>",
        description="""
        <div style='font-size:18px; color:#e8ecfa;'>
        Plan comprehensive trips with <b>AI-powered recommendations!</b><br>
        <span style='color:#78FFD6;'>Weather, events, budget, dietary and accessibility needs all included.</span>
        <hr style="border:1px solid #3fa7d6;">
        </div>
        """,
        theme=gr.themes.Glass(primary_hue="cyan", secondary_hue="indigo"),
        css="""
        body { background: linear-gradient(90deg, #011936 0%, #2385a4 100%) !important; }
        .gradio-container { background: transparent !important; }
        .form, .block { background: rgba(20,30,56,0.82) !important; border-radius: 22px !important; }
        .input, .output { border-radius: 15px !important; }
        label { color: #87ffe8 !important; font-weight: bold; font-size: 1.1em;}
        .tabitem { background: #102040 !important; }
        ::placeholder { color: #ddf !important; }
        """
    )
    return interface

# === Launch Application ===
if __name__ == "__main__":
    print("🌈 Enhanced Travel Itinerary Planner Launching ...")
    interface = create_interface()
    interface.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
