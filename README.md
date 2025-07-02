# Hotel Booking Agent (LangGraph & LLMs)

## Overview
This project is an AI-powered hotel booking agent built with LangGraph, LangChain, and Groq LLM. It can:
- Book hotel rooms (collecting check-in/check-out dates, room type, number of guests)
- Reschedule existing reservations
- Answer hotel-related questions (amenities, check-in times, etc.)
- Maintain conversation history for context-aware responses
- Interact with users via Instagram DMs (mocked due to time constraints in obtaining Instagram Graph API access)

---

## Features
- **Conversational AI**: Guides users through booking, rescheduling, and Q&A.
- **Stateful**: Maintains context and conversation history using LangGraph.
- **Lightweight Storage**: Uses a JSON file for reservation data.
- **Instagram Integration**: Mocked due to time constraints in obtaining Instagram Graph API access; prints responses to console instead of sending real DMs.
- **Robust Error Handling**: Handles invalid input, API errors, and file errors.

---

## Setup Instructions

### 1. Clone the Repository
```
git clone <your-fork-url>
cd Hiring_challange
```

### 2. Install Dependencies
```
pip install -r requirements.txt
```

### 3. Set Up Environment Variables
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_actual_groq_api_key_here
```

### 4. Run the Agent
```
python hotel_booking_agent.py
```
You will see the agent's responses printed to the console (mock Instagram DMs due to time constraints in obtaining Instagram Graph API access).

---

## Architecture & Design Choices
- **LangGraph State Machine**: Manages the flow between intent detection, input processing, booking, rescheduling, and Q&A.
- **TypedDict State**: Ensures type safety and compatibility with LangGraph.
- **LLM (Groq)**: Handles intent classification and Q&A. Model: `llama3-70b-8192` (free tier, fast, and reliable).
- **JSON Storage**: Simple, portable, and easy to inspect for demo/testing.
- **Instagram API Mock**: Due to time constraints in obtaining Instagram Graph API access, the `send_instagram_message` function prints to console. Swap in real API logic for production when Instagram Graph API access is available.
- **Error Handling**: All user input and API calls are wrapped with error handling for robustness.

---

## LangGraph Flow Diagram

```mermaid
flowchart TD
    A([detect_intent]) --> B([process_input])
    B -->|booking| C([handle_booking])
    B -->|rescheduling| D([handle_rescheduling])
    B -->|question| E([handle_question])
    C --> F([END])
    D --> F
    E --> F
```

---

## Example Usage
```
I want to book a room
2025-07-01
2025-07-03
deluxe
2
What are the hotel amenities?
```

---

## Submission Checklist
- [x] Booking, rescheduling, and Q&A flows
- [x] LangGraph state machine
- [x] Code quality and error handling
- [x] Documentation and flow diagram

---

## Notes
- Instagram Graph API integration was not implemented due to time constraints in obtaining API access (requires app review, business account setup, and permissions approval).
- For real Instagram integration, implement the `send_instagram_message` function with the Instagram Graph API and provide a valid access token once access is granted.
