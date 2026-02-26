import json
import os
import ollama
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from graph_utils.states import GenerateState

# Define the state for LangGraph

def generate_dialogues(state: GenerateState) -> Dict[str, Any]:
    """
    Node to generate dialogues using Ollama and Llama 3.1.
    """
    print("Generating dialogues... This might take a moment.")
    num_dialogues = state.num_dialogues
    
    prompt = f"""
    Generate a JSON object of {num_dialogues} diverse customer support chat dialogues between a 'client' and an 'agent'.
    
    You must include various scenarios:
    - проблеми з оплатою (payment issues)
    - технічні помилки (technical errors)
    - доступ до акаунту (account access)
    - питання по тарифу (tariff questions)
    - повернення коштів (refunds)
    
    You must also include different case types:
    - успішні кейси (successful cases)
    - проблемні кейси (problematic cases)
    - конфліктні кейси (conflict cases)
    - випадки з помилками агента (cases with agent errors)
    
    CRITICAL INSTRUCTIONS:
    1. Include cases with hidden dissatisfaction (client formally thanks the agent, but the problem is not actually resolved).
    2. Include cases where the agent makes tonal mistakes (e.g., rude tone) or logical mistakes (e.g., incorrect info, ignoring questions).
    3. Output the dialogue in Ukrainian.
    4. Every dialogue must be 5 OR MORE messages long
    5. Generate EXACTLY the amount of dialogues required, right now it is {num_dialogues}
    
    Respond STRICTLY with a valid JSON object using this exact schema:
    {{
      "dialogue_id_1": {{
        "scenario": "one of the scenarios above",
        "case_type": "one of the case types above",
        "messages": [
          {{"role": "client", "text": "message text"}},
          {{"role": "agent", "text": "message text"}},
          {{"role": "client", "text": "message text"}},
          {{"role": "agent", "text": "message text"}},
          {{"role": "client", "text": "message text"}}
        ]
      }},
      "dialogue_id_2": {{
        "scenario": "...",
        "case_type": "...",
        "messages": []
      }}
    }}

    NOTE: The example above only shows 2 messages for brevity, but YOU MUST generate between 5 and 12 messages for EVERY dialogue in your final JSON.
    """

    # Create the dialogue using Ollama
    response = ollama.chat(
        model='llama3.1',
        messages=[{'role': 'user', 'content': prompt}],
        format='json',
        options={'temperature': 0.3}
    )
    
    try:
        dialogues = json.loads(response['message']['content'])
        print(dialogues)
    except json.JSONDecodeError:
        print("Error: Model did not return valid JSON.")
        dialogues = []
        
    return {"dialogues": dialogues}

def save_dialogues(state: GenerateState) -> Dict[str, Any]:
    """
    Node to save the generated dialogues to a JSON file.
    """
    dialogues = state.dialogues
    if not dialogues:
        print(dialogues)
        raise ValueError("Incorrect dialogues creation")

    output_file = state.output_file
    if not output_file:
        output_file = "data/raw_dialogues.json"
    
    print(f"Saving {len(dialogues)} dialogues to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dialogues, f, indent=4, ensure_ascii=False)
        
    print("Save complete.")
    return {"output_file": output_file}

def build_workflow() -> StateGraph:
    """
    Constructs the LangGraph workflow.
    """
    workflow = StateGraph(state_schema=GenerateState)
    
    # Add nodes
    workflow.add_node("generate_dialogues", generate_dialogues)
    workflow.add_node("save_dialogues", save_dialogues)
    
    # Connect nodes
    workflow.add_edge(START, "generate_dialogues")
    workflow.add_edge("generate_dialogues", "save_dialogues")
    workflow.add_edge("save_dialogues", END)
    
    return workflow.compile()

if __name__ == "__main__":
    app = build_workflow()
    
    # Define the initial state based on the project structure
    initial_state = {
        "num_dialogues": 1
    }
    
    # Run the graph
    # app.get_graph().draw_mermaid_png(output_file_path="./graph_generate.png")
    result = app.invoke(initial_state)
    print("Pipeline execution finished successfully.")