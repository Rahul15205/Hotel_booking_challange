import json
import os
from typing import Dict, Any, List, TypedDict, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import requests
import dotenv
import re
from pydantic import SecretStr
from datetime import datetime

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
CHAT_SESSIONS_FILE = "chat_sessions.json"

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

def load_chat_sessions():
    if not os.path.exists(CHAT_SESSIONS_FILE):
        return {}
    try:
        with open(CHAT_SESSIONS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def save_chat_sessions(sessions):
    try:
        with open(CHAT_SESSIONS_FILE, "w") as f:
            json.dump(sessions, f, indent=2)
    except Exception as e:
        print(f"Error saving chat sessions: {e}")

class AgentState(TypedDict, total=False):
    messages: List[HumanMessage]
    context: Dict[str, Any]
    intent: str
    reservation_id: Optional[int]
    conversation_history: List[Dict[str, str]]

def new_agent_state() -> AgentState:
    return {
        "messages": [],
        "context": {},
        "intent": "",
        "reservation_id": None,
        "conversation_history": []
    }

def load_agent_state(user_id: str) -> AgentState:
    """Load existing conversation state for a user"""
    sessions = load_chat_sessions()
    if user_id in sessions:
        session_data = sessions[user_id]
        state = new_agent_state()
        state["context"] = session_data.get("context", {})
        state["conversation_history"] = session_data.get("conversation_history", [])
        return state
    return new_agent_state()

def save_agent_state(user_id: str, state: AgentState):
    """Save conversation state for a user"""
    sessions = load_chat_sessions()
    sessions[user_id] = {
        "context": state.get("context", {}),
        "conversation_history": state.get("conversation_history", []),
        "last_updated": datetime.now().isoformat()
    }
    save_chat_sessions(sessions)

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
        state.setdefault("messages", []).append(AIMessage(content="Please provide check-in date (YYYY-MM-DD)."))
        return state
    if "check_out_date" not in ctx:
        state.setdefault("messages", []).append(AIMessage(content="Please provide check-out date (YYYY-MM-DD)."))
        return state
    if "room_type" not in ctx:
        room_types = ", ".join(HOTEL_DATA["room_types"].keys())
        state.setdefault("messages", []).append(AIMessage(content=f"Please choose a room type: {room_types}."))
        return state
    if "num_guests" not in ctx:
        state.setdefault("messages", []).append(AIMessage(content="How many guests?"))
        return state
    
    # All booking details collected, create reservation
    reservations = load_reservations()
    reservation_id = (max([r["id"] for r in reservations], default=0) + 1) if reservations else 1
    reservation = {
        "id": reservation_id,
        "user_id": ctx.get("user_id"),
        "check_in_date": ctx["check_in_date"],
        "check_out_date": ctx["check_out_date"],
        "room_type": ctx["room_type"],
        "num_guests": ctx["num_guests"],
        "total_price": HOTEL_DATA["room_types"][ctx["room_type"]]["price"],
        "created_at": datetime.now().isoformat()
    }
    reservations.append(reservation)
    save_reservations(reservations)
    state["reservation_id"] = reservation_id
    
    # Clear booking context after successful booking
    ctx.pop("check_in_date", None)
    ctx.pop("check_out_date", None)
    ctx.pop("room_type", None)
    ctx.pop("num_guests", None)
    
    confirmation_msg = f"🎉 Booking confirmed! Your reservation ID is: {reservation_id}\n\nDetails:\n• Check-in: {reservation['check_in_date']}\n• Check-out: {reservation['check_out_date']}\n• Room: {reservation['room_type']}\n• Guests: {reservation['num_guests']}\n• Total: ₹{reservation['total_price']}\n\nThank you for choosing Sunset Resort!"
    state.setdefault("messages", []).append(AIMessage(content=confirmation_msg))
    return state

def handle_rescheduling(state: AgentState) -> AgentState:
    ctx = state.get("context", {})
    if "reservation_id" not in ctx:
        state.setdefault("messages", []).append(AIMessage(content="Please provide your reservation ID."))
        return state
    if "new_check_in_date" not in ctx:
        state.setdefault("messages", []).append(AIMessage(content="Please provide new check-in date (YYYY-MM-DD)."))
        return state
    if "new_check_out_date" not in ctx:
        state.setdefault("messages", []).append(AIMessage(content="Please provide new check-out date (YYYY-MM-DD)."))
        return state
    
    reservations = load_reservations()
    updated = False
    for r in reservations:
        if r["id"] == ctx["reservation_id"]:
            r["check_in_date"] = ctx["new_check_in_date"]
            r["check_out_date"] = ctx["new_check_out_date"]
            r["updated_at"] = datetime.now().isoformat()
            updated = True
            break
    
    if updated:
        save_reservations(reservations)
        # Clear rescheduling context
        ctx.pop("reservation_id", None)
        ctx.pop("new_check_in_date", None)
        ctx.pop("new_check_out_date", None)
        state.setdefault("messages", []).append(AIMessage(content="✅ Reservation updated successfully! Your new dates have been confirmed."))
    else:
        state.setdefault("messages", []).append(AIMessage(content="❌ Reservation ID not found. Please check your reservation ID and try again."))
    return state

def handle_question(state: AgentState) -> AgentState:
    user_message = state.get("messages", [])[-1].content
    
    # Enhanced prompt for better responses
    enhanced_prompt = f"""You are a helpful hotel booking assistant for Sunset Resort in Goa, India. 
    
Hotel Information:
- Name: {HOTEL_DATA['name']}
- Location: {HOTEL_DATA['location']}
- Amenities: {', '.join(HOTEL_DATA['amenities'])}
- Check-in: {HOTEL_DATA['check_in_time']}
- Check-out: {HOTEL_DATA['check_out_time']}

Room Types and Prices:
{json.dumps(HOTEL_DATA['room_types'], indent=2)}

User Question: {user_message}

Please provide a helpful, friendly response about the hotel, booking process, or any other relevant information."""
    
    response = llm.invoke([HumanMessage(content=enhanced_prompt)])
    response_content = getattr(response, "content", str(response))
    state.setdefault("messages", []).append(AIMessage(content=response_content))
    return state

def send_instagram_message(user_id: str, message: str, access_token: str) -> None:
    print(f"[MOCK INSTAGRAM DM to {user_id}]: {message}")

def process_input(state: AgentState) -> AgentState:
    user_message = state.get("messages", [])[-1].content
    ctx = state.get("context", {})
    
    if state.get("intent") == "booking":
        if "check_in_date" not in ctx:
            # Validate date format
            if re.match(r'\d{4}-\d{2}-\d{2}', user_message):
                ctx["check_in_date"] = user_message
            else:
                state.setdefault("messages", []).append(AIMessage(content="Please provide the check-in date in YYYY-MM-DD format (e.g., 2025-07-01)."))
        elif "check_out_date" not in ctx:
            if re.match(r'\d{4}-\d{2}-\d{2}', user_message):
                ctx["check_out_date"] = user_message
            else:
                state.setdefault("messages", []).append(AIMessage(content="Please provide the check-out date in YYYY-MM-DD format (e.g., 2025-07-03)."))
        elif "room_type" not in ctx:
            if user_message.lower() in HOTEL_DATA["room_types"]:
                ctx["room_type"] = user_message.lower()
            else:
                room_types = ", ".join(HOTEL_DATA["room_types"].keys())
                state.setdefault("messages", []).append(AIMessage(content=f"Please choose a valid room type: {room_types}."))
        elif "num_guests" not in ctx:
            if isinstance(user_message, str) and user_message.isdigit():
                num_guests = int(user_message)
                if num_guests > 0:
                    ctx["num_guests"] = num_guests
                else:
                    state.setdefault("messages", []).append(AIMessage(content="Please enter a valid number of guests (greater than 0)."))
            else:
                state.setdefault("messages", []).append(AIMessage(content="Please enter a valid number of guests."))
    
    elif state.get("intent") == "rescheduling":
        if "reservation_id" not in ctx:
            if isinstance(user_message, str) and user_message.isdigit():
                ctx["reservation_id"] = int(user_message)
            else:
                state.setdefault("messages", []).append(AIMessage(content="Please enter a valid reservation ID (numbers only)."))
        elif "new_check_in_date" not in ctx:
            if re.match(r'\d{4}-\d{2}-\d{2}', user_message):
                ctx["new_check_in_date"] = user_message
            else:
                state.setdefault("messages", []).append(AIMessage(content="Please provide the new check-in date in YYYY-MM-DD format."))
        elif "new_check_out_date" not in ctx:
            if re.match(r'\d{4}-\d{2}-\d{2}', user_message):
                ctx["new_check_out_date"] = user_message
            else:
                state.setdefault("messages", []).append(AIMessage(content="Please provide the new check-out date in YYYY-MM-DD format."))
    
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
        # Load existing state or create new one
        state = load_agent_state(user_id)
        state.get("context", {})["user_id"] = user_id
        state.setdefault("messages", []).append(HumanMessage(content=message))
        
        result = app.invoke(state)
        response = result["messages"][-1].content
        
        # Update conversation history
        state.setdefault("conversation_history", []).append({
            "user": message,
            "assistant": response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Save updated state
        save_agent_state(user_id, state)
        
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

def start_chat_session(user_id: str = "default_user"):
    """Start an interactive chat session"""
    print("🏨 Welcome to Sunset Resort Hotel Booking Assistant!")
    print("I can help you with:")
    print("• Booking a room")
    print("• Rescheduling existing reservations")
    print("• Answering questions about our hotel")
    print("• Viewing your reservations")
    print("\nType 'quit' to exit the chat.\n")
    
    state = load_agent_state(user_id)
    state.get("context", {})["user_id"] = user_id
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Thank you for chatting with us! Have a great day!")
                break
            
            if not user_input:
                continue
            
            # Add user message to state
            state.setdefault("messages", []).append(HumanMessage(content=user_input))
            
            # Process through the agent
            result = app.invoke(state)
            response = result["messages"][-1].content
            
            # Update conversation history
            state.setdefault("conversation_history", []).append({
                "user": user_input,
                "assistant": response,
                "timestamp": datetime.now().isoformat()
            })
            
            # Save state
            save_agent_state(user_id, state)
            
            print(f"Assistant: {response}\n")
            
        except KeyboardInterrupt:
            print("\n👋 Chat session ended. Goodbye!")
            break
        except Exception as e:
            print(f"❌ An error occurred: {e}")
            print("Please try again.\n")

def view_user_reservations(user_id: str):
    """View all reservations for a user"""
    reservations = load_reservations()
    user_reservations = [r for r in reservations if r.get("user_id") == user_id]
    
    if not user_reservations:
        print("No reservations found for this user.")
        return
    
    print(f"\n📋 Your Reservations:")
    for reservation in user_reservations:
        print(f"ID: {reservation['id']}")
        print(f"Check-in: {reservation['check_in_date']}")
        print(f"Check-out: {reservation['check_out_date']}")
        print(f"Room: {reservation['room_type']}")
        print(f"Guests: {reservation['num_guests']}")
        print(f"Total: ₹{reservation.get('total_price', 'N/A')}")
        print("-" * 30)

if __name__ == "__main__":
    # Start interactive chat session
    start_chat_session()