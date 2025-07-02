import json
import os
from typing import Dict, Any, List, TypedDict, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
import requests
import dotenv
import re
from pydantic import SecretStr

dotenv.load_dotenv()

llm = ChatGroq(model="llama3-70b-8192", api_key=SecretStr(os.getenv("GROQ_API_KEY", "YOUR_API_KEY")))
HOTEL_DATA = {
    "name": "Sunset Resort",
    "location": "Goa, India",
    "amenities": ["pool", "spa", "restaurant", "free Wi-Fi"],
    "check_in_time": "2:00 PM",
    "check_out_time": "11:00 AM",
    "room_types": {
        "standard": {"price": 5000, "capacity": 2},
        "deluxe": {"price": 8000, "capacity": 4},
        "suite": {"price": 12000, "capacity": 6}
    }
}

RESERVATION_FILE = "reservations.json"

def load_reservations():
    if not os.path.exists(RESERVATION_FILE):
        return []
    try:
        with open(RESERVATION_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []

def save_reservations(reservations):
    try:
        with open(RESERVATION_FILE, "w") as f:
            json.dump(reservations, f, indent=2)
    except Exception as e:
        print(f"Error saving reservations: {e}")

class AgentState(TypedDict, total=False):
    messages: List[HumanMessage]
    context: Dict[str, Any]
    intent: str
    reservation_id: Optional[int]

def new_agent_state() -> AgentState:
    return {
        "messages": [],
        "context": {},
        "intent": "",
        "reservation_id": None
    }

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a hotel booking assistant for Sunset Resort. Handle booking, rescheduling, and Q&A. Be concise and friendly. Hotel data: " + json.dumps(HOTEL_DATA)),
    *[{"role": "user", "content": "{{message}}"}]
])

def detect_intent(state: AgentState) -> AgentState:
    user_message = state.get("messages", [])[-1].content
    intent_prompt = f"Classify the intent of this message as 'booking', 'rescheduling', or 'question': {user_message}"
    intent_response = llm.invoke([HumanMessage(content=intent_prompt)])
    response_text = getattr(intent_response, "content", "question").strip().lower()
    # Extract intent using regex or keyword search
    match = re.search(r"(booking|rescheduling|question)", response_text)
    if match:
        intent = match.group(1)
    else:
        intent = "question"  # fallback
    state["intent"] = intent
    return state

def handle_booking(state: AgentState) -> AgentState:
    ctx = state.get("context", {})
    if "check_in_date" not in ctx:
        state.setdefault("messages", []).append(HumanMessage(content="Please provide check-in date (YYYY-MM-DD)."))
        return state
    if "check_out_date" not in ctx:
        state.setdefault("messages", []).append(HumanMessage(content="Please provide check-out date (YYYY-MM-DD)."))
        return state
    if "room_type" not in ctx:
        room_types = ", ".join(HOTEL_DATA["room_types"].keys())
        state.setdefault("messages", []).append(HumanMessage(content=f"Please choose a room type: {room_types}."))
        return state
    if "num_guests" not in ctx:
        state.setdefault("messages", []).append(HumanMessage(content="How many guests?"))
        return state
    reservations = load_reservations()
    reservation_id = (max([r["id"] for r in reservations], default=0) + 1) if reservations else 1
    reservation = {
        "id": reservation_id,
        "user_id": ctx.get("user_id"),
        "check_in_date": ctx["check_in_date"],
        "check_out_date": ctx["check_out_date"],
        "room_type": ctx["room_type"],
        "num_guests": ctx["num_guests"]
    }
    reservations.append(reservation)
    save_reservations(reservations)
    state["reservation_id"] = reservation_id
    state.setdefault("messages", []).append(HumanMessage(content=f"Booking confirmed! Reservation ID: {reservation_id}"))
    return state

def handle_rescheduling(state: AgentState) -> AgentState:
    ctx = state.get("context", {})
    if "reservation_id" not in ctx:
        state.setdefault("messages", []).append(HumanMessage(content="Please provide your reservation ID."))
        return state
    if "new_check_in_date" not in ctx:
        state.setdefault("messages", []).append(HumanMessage(content="Please provide new check-in date (YYYY-MM-DD)."))
        return state
    if "new_check_out_date" not in ctx:
        state.setdefault("messages", []).append(HumanMessage(content="Please provide new check-out date (YYYY-MM-DD)."))
        return state
    reservations = load_reservations()
    updated = False
    for r in reservations:
        if r["id"] == ctx["reservation_id"]:
            r["check_in_date"] = ctx["new_check_in_date"]
            r["check_out_date"] = ctx["new_check_out_date"]
            updated = True
            break
    if updated:
        save_reservations(reservations)
        state.setdefault("messages", []).append(HumanMessage(content="Reservation updated successfully!"))
    else:
        state.setdefault("messages", []).append(HumanMessage(content="Reservation ID not found."))
    return state

def handle_question(state: AgentState) -> AgentState:
    user_message = state.get("messages", [])[-1].content
    response = llm.invoke([HumanMessage(content=user_message)])
    state.setdefault("messages", []).append(HumanMessage(content=getattr(response, "content", str(response))))
    return state

def send_instagram_message(user_id: str, message: str, access_token: str) -> None:
    print(f"[MOCK INSTAGRAM DM to {user_id}]: {message}")

def process_input(state: AgentState) -> AgentState:
    user_message = state.get("messages", [])[-1].content
    ctx = state.get("context", {})
    if state.get("intent") == "booking":
        if "check_in_date" not in ctx:
            ctx["check_in_date"] = user_message
        elif "check_out_date" not in ctx:
            ctx["check_out_date"] = user_message
        elif "room_type" not in ctx:
            ctx["room_type"] = user_message
        elif "num_guests" not in ctx:
            if isinstance(user_message, str):
                try:
                    ctx["num_guests"] = int(user_message)
                except Exception:
                    state.setdefault("messages", []).append(HumanMessage(content="Please enter a valid number of guests."))
            else:
                state.setdefault("messages", []).append(HumanMessage(content="Please enter a valid number of guests."))
    elif state.get("intent") == "rescheduling":
        if "reservation_id" not in ctx:
            if isinstance(user_message, str):
                try:
                    ctx["reservation_id"] = int(user_message)
                except Exception:
                    state.setdefault("messages", []).append(HumanMessage(content="Please enter a valid reservation ID."))
            else:
                state.setdefault("messages", []).append(HumanMessage(content="Please enter a valid reservation ID."))
        elif "new_check_in_date" not in ctx:
            ctx["new_check_in_date"] = user_message
        elif "new_check_out_date" not in ctx:
            ctx["new_check_out_date"] = user_message
    return state

graph = StateGraph(AgentState)
graph.add_node("detect_intent", detect_intent)
graph.add_node("process_input", process_input)
graph.add_node("handle_booking", handle_booking)
graph.add_node("handle_rescheduling", handle_rescheduling)
graph.add_node("handle_question", handle_question)
graph.add_edge("detect_intent", "process_input")
graph.add_conditional_edges(
    "process_input",
    lambda state: state["intent"],
    {
        "booking": "handle_booking",
        "rescheduling": "handle_rescheduling",
        "question": "handle_question"
    }
)
graph.add_edge("handle_booking", END)
graph.add_edge("handle_rescheduling", END)
graph.add_edge("handle_question", END)
graph.set_entry_point("detect_intent")
app = graph.compile()

def handle_instagram_dm(user_id: str, message: str, access_token: str):
    try:
        state = new_agent_state()
        state.get("context", {})["user_id"] = user_id
        state.setdefault("messages", []).append(HumanMessage(content=message))
        result = app.invoke(state)
        response = result["messages"][-1].content
        send_instagram_message(user_id, response, access_token)
        return response
    except Exception as e:
        error_msg = "Sorry, something went wrong. Please try again."
        try:
            send_instagram_message(user_id, error_msg, access_token)
        except Exception:
            pass
        print(f"Error: {e}")
        return error_msg

if __name__ == "__main__":
    access_token = "YOUR_INSTAGRAM_ACCESS_TOKEN"
    user_id = "test_user123"
    print(handle_instagram_dm(user_id, "I want to book a room", access_token))
    print(handle_instagram_dm(user_id, "2025-07-01", access_token))
    print(handle_instagram_dm(user_id, "2025-07-03", access_token))
    print(handle_instagram_dm(user_id, "deluxe", access_token))
    print(handle_instagram_dm(user_id, "2", access_token))
    print(handle_instagram_dm(user_id, "What are the hotel amenities?", access_token))