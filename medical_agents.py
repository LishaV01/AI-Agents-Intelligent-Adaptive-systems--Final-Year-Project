import os
from typing import TypedDict, Annotated, List, Literal
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from brain_of_the_doctor import analyze_image_with_query, encode_image

# 1. Define the State
class MedicalState(TypedDict):
    transcription: str
    image_path: str
    severity: str # "MILD" or "SEVERE"
    doctor_analysis: str
    nurse_advice: str
    final_consultation: str

# 2. Define the Nodes

def health_analyzer_node(state: MedicalState):
    """The Triage Officer: Assesses severity."""
    llm = ChatGroq(api_key=os.environ.get("GROQ_API_KEY"), model_name="llama-3.3-70b-versatile")
    
    prompt = f"""You are a medical triage officer. 
    Analyze the patient's concern: '{state['transcription']}'
    Classify the severity as 'MILD' or 'SEVERE'. 
    - MILD: Common cold, small rashes, minor aches, dandruff.
    - SEVERE: Intense pain, deep wounds, difficulty breathing, major infections.
    
    Return ONLY the word 'MILD' or 'SEVERE'."""
    
    response = llm.invoke(prompt)
    severity = response.content.strip().upper()
    
    # Validation
    if "SEVERE" in severity:
        return {"severity": "SEVERE"}
    return {"severity": "MILD"}

def doctor_agent_node(state: MedicalState):
    """The Specialist: Handles SEVERE cases."""
    system_prompt = "You are a senior physician handling a SEVERE case. Analyze the symptoms and image results provided. Be very thorough and professional."
    
    # We reuse the existing vision logic for the doctor agent
    analysis = analyze_image_with_query(
        query=system_prompt + " Patient says: " + state["transcription"],
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        encoded_image=encode_image(state["image_path"]),
        api_key=os.environ.get("GROQ_API_KEY")
    )
    return {"doctor_analysis": "DOCTOR DIAGNOSIS: " + analysis}

def nurse_agent_node(state: MedicalState):
    """The Caregiver: Handles MILD cases."""
    llm = ChatGroq(api_key=os.environ.get("GROQ_API_KEY"), model_name="llama-3.3-70b-versatile")
    
    prompt = f"""You are a registered nurse handling a MILD case. 
    The patient says: {state['transcription']}.
    Provide practical home care advice and common remedies. Keep it concise and supportive."""
    
    response = llm.invoke(prompt)
    return {"nurse_advice": "NURSE CARE PLAN: " + response.content}

def summary_agent_node(state: MedicalState):
    """The Coordinator: Creates the final report using LOCAL Gemma 3."""
    # Using local Ollama for the final summary layer
    llm = ChatOllama(model="gemma3:12b")
    
    # We aggregate whatever analysis was done
    report_source = state["doctor_analysis"] if state["severity"] == "SEVERE" else state["nurse_advice"]
    
    prompt = f"""Summarize this medical consultation for the patient.
    Medical Personnel found: {report_source}
    Create a final professional consultation summary paragraph."""
    
    response = llm.invoke(prompt)
    return {"final_consultation": response.content}

# 3. Defining the Routing Logic
def route_to_professional(state: MedicalState) -> Literal["doctor", "nurse"]:
    if state["severity"] == "SEVERE":
        return "doctor"
    return "nurse"

# 4. Build the Graph
def create_medical_graph():
    workflow = StateGraph(MedicalState)

    # Add Nodes
    workflow.add_node("analyzer", health_analyzer_node)
    workflow.add_node("doctor", doctor_agent_node)
    workflow.add_node("nurse", nurse_agent_node)
    workflow.add_node("summarizer", summary_agent_node)

    # Add Edges
    workflow.set_entry_point("analyzer")
    
    # Conditional Edge from Analyzer to either Doctor or Nurse
    workflow.add_conditional_edges(
        "analyzer",
        route_to_professional
    )
    
    workflow.add_edge("doctor", "summarizer")
    workflow.add_edge("nurse", "summarizer")
    workflow.add_edge("summarizer", END)

    return workflow.compile()

# Global graph instance
medical_team = create_medical_graph()

def run_medical_consultation(transcription, image_path):
    initial_state = {
        "transcription": transcription,
        "image_path": image_path,
        "severity": "",
        "doctor_analysis": "",
        "nurse_advice": "",
        "final_consultation": ""
    }
    
    final_output = medical_team.invoke(initial_state)
    return final_output
