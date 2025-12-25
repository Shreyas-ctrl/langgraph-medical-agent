import os
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# 🔐 API KEY (use environment variable, NOT getpass on web)


llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=0.2
)

# ------------------ NODES ------------------

def get_symptom(state: dict) -> dict:
    return state

def classify_symptom(state: dict) -> dict:
    prompt = (
        "You are a helpful medical assistant. "
        "Classify the symptom into: general, emergency, or mental health. "
        f"Symptom: {state['symptom']}. "
        "Respond with ONE word only."
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    state["category"] = response.content.strip().lower()
    return state

def symptom_router(state: dict) -> str:
    cat = state["category"]
    if "emergency" in cat:
        return "emergency"
    if "mental" in cat:
        return "mental_health"
    return "general"

def general_node(state: dict) -> dict:
    state["answer"] = "This seems like a general issue. Please consult a doctor."
    return state

def emergency_node(state: dict) -> dict:
    state["answer"] = "This is a medical emergency. Seek immediate help."
    return state

def mental_health_node(state: dict) -> dict:
    state["answer"] = "This may be related to mental health. Please speak to a counselor."
    return state

# ------------------ GRAPH ------------------

def build_graph():
    builder = StateGraph(dict)

    builder.set_entry_point("get_symptom")
    builder.add_node("get_symptom", get_symptom)
    builder.add_node("classify", classify_symptom)
    builder.add_node("general", general_node)
    builder.add_node("emergency", emergency_node)
    builder.add_node("mental_health", mental_health_node)

    builder.add_edge("get_symptom", "classify")
    builder.add_conditional_edges("classify", symptom_router, {
        "general": "general",
        "emergency": "emergency",
        "mental_health": "mental_health"
    })

    builder.add_edge("general", END)
    builder.add_edge("emergency", END)
    builder.add_edge("mental_health", END)

    return builder.compile()
