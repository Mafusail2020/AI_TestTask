import json
import os
import ollama
from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, START, END
from graph_utils.states import AnalyzeState


def load_dialogues(state: AnalyzeState) -> Dict[str, Any]:
    """
    Node to load the raw dialogues from the JSON file.
    """
    input_file = state.input_file
    print(f"Loading dialogues from {input_file}...")
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Cannot find {input_file}. Please run generate.py first.")
        
    with open(input_file, 'r', encoding='utf-8') as f:
        raw_dialogues = json.load(f)
        
    print(f"Loaded {len(raw_dialogues)} dialogues.")
    return {"raw_dialogues": raw_dialogues}

def analyze_dialogues(state: AnalyzeState) -> Dict[str, Any]:
    """
    Node to analyze each dialogue using Ollama and Llama 3.1.
    """
    raw_dialogues = state.raw_dialogues
    if not raw_dialogues:
        raise ValueError("No raw_dialogues provided, try to check source file")
    analyzed_dialogues = {}
    
    print("Analyzing dialogues... This will take a moment per dialogue.")
    
    for dialogue_id, dialogue_data in raw_dialogues.items():
        print(f"Analyzing {dialogue_id}...")
        
        messages = dialogue_data.get("messages", [])
        dialogue_text = "\n".join([f"{msg['role'].capitalize()}: {msg['text']}" for msg in messages])
        
        prompt = f"""
        You are an Expert Customer Support QA Analyst. Your task is to analyze the following customer support dialogue and evaluate the agent's performance and the client's true satisfaction.

        DIALOGUE TO ANALYZE:
        {dialogue_text}

        EVALUATION CRITERIA:
        1. "intent": Identify the main topic. Must be EXACTLY one of: "проблеми з оплатою", "технічні помилки", "доступ до акаунту", "питання по тарифу", "повернення коштів", or "other".
        2. "satisfaction": Evaluate the client's true state. Must be EXACTLY one of: "satisfied", "neutral", or "unsatisfied". 
           CRITICAL: Look for HIDDEN DISSATISFACTION. If the client says "thank you" or "fine" but the problem wasn't actually solved, or they gave up out of frustration, mark them as "unsatisfied".
        3. "quality_score": Rate the agent from 1 to 5 (integer). 
           - 5: Perfect resolution, polite tone, efficient, perfect grammar.
           - 3-4: Resolved, but agent was slow, slightly robotic, or had minor typos.
           - 1-2: Agent made errors, was rude, failed to resolve the issue, used noticeably poor grammar, or used Russian words.
        4. "agent_mistakes": A list of specific mistakes made by the agent. Choose from: ["ignored_question", "incorrect_info", "rude_tone", "no_resolution", "unnecessary_escalation", "poor_grammar", "used_russian"]. 
           CRITICAL LANGUAGE CHECK: If the agent uses Russian words/surzhik or has noticeably bad grammar, you MUST include "used_russian" or "poor_grammar" in the list and lower the quality score. If the agent was perfect, return an empty list [].

        Respond STRICTLY with a valid JSON object using this exact schema:
        {{
          "reasoning": "A brief 1-2 sentence explanation of your evaluation, specifically noting if there was hidden dissatisfaction, agent errors, or language issues.",
          "intent": "string",
          "satisfaction": "string",
          "quality_score": integer,
          "agent_mistakes": ["string"]
        }}
        """

        response = ollama.chat(
            model='llama3.1',
            messages=[{'role': 'user', 'content': prompt}],
            format='json',
            options={'temperature': 0.0, 'seed': 52}
        )
        
        try:
            analysis_result = json.loads(response['message']['content'])
            
            # Combining the original data with the new analysis
            analyzed_dialogues[dialogue_id] = {
                "original_data": dialogue_data,
                "analysis": analysis_result
            }
        except json.JSONDecodeError:
            print(f"Error: Model did not return valid JSON for {dialogue_id}.")
            analyzed_dialogues[dialogue_id] = {
                "original_data": dialogue_data,
                "analysis": {"error": "Failed to parse LLM response"}
            }

    return {"analyzed_dialogues": analyzed_dialogues}

def save_analysis(state: AnalyzeState) -> Dict[str, Any]:
    """
    Node to save the analyzed dialogues to a JSON file.
    """
    analyzed_dialogues = state.analyzed_dialogues
    if not analyzed_dialogues:
        print(analyzed_dialogues)
        raise ValueError("Incorrect dialogues")
    output_file = state.output_file
    if not output_file:
        output_file = "data/analyzed_dialogues.json"

    print(f"Saving analysis to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analyzed_dialogues, f, indent=4, ensure_ascii=False)
        
    print("Analysis complete and saved.")
    return {"output_file": output_file}

def build_analyze_workflow() -> StateGraph:
    """
    Constructs the LangGraph workflow for analysis.
    """
    workflow = StateGraph(AnalyzeState)
    
    # Add nodes
    workflow.add_node("load_dialogues", load_dialogues)
    workflow.add_node("analyze_dialogues", analyze_dialogues)
    workflow.add_node("save_analysis", save_analysis)
    
    # Connect nodes
    workflow.add_edge(START, "load_dialogues")
    workflow.add_edge("load_dialogues", "analyze_dialogues")
    workflow.add_edge("analyze_dialogues", "save_analysis")
    workflow.add_edge("save_analysis", END)
    
    return workflow.compile()

if __name__ == "__main__":
    app = build_analyze_workflow()
    
    initial_state = {
        "input_file": "data/raw_dialogues.json",
    }
    
    result = app.invoke(initial_state)
    print("Analysis pipeline execution finished successfully.")