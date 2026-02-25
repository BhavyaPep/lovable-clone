from dotenv import load_dotenv

#from langchain_groq.chat_models import ChatGroq
from langchain_openai import AzureChatOpenAI
import os
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent

from tools import init_project_root, set_project_root
from agent.prompts import *
from agent.states import *
from agent.tools import write_file, read_file, get_current_directory, list_file

from pathlib import Path

_ = load_dotenv()



#llm = ChatGroq(model="openai/gpt-oss-120b")


# Azure OpenAI via langchain-openai
# Uses your .env values:
#   AZURE_OPENAI_ENDPOINT
#   AZURE_OPENAI_API_KEY
#   AZURE_OPENAI_GPT4_DEPLOYMENT
#   AZURE_OPENAI_API_VERSION
llm = AzureChatOpenAI(
    deployment_name=os.environ["AZURE_OPENAI_GPT4_DEPLOYMENT"],   # e.g., "gpt-4-1" (deployment name)
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/"),
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    temperature=0.2,
)

def planner_agent(state: dict) -> dict:
    """Converts user prompt into a structured Plan."""
    user_prompt = state["user_prompt"]
    resp = llm.with_structured_output(Plan, method="function_calling").invoke(
        planner_prompt(user_prompt)
    )
    if resp is None:
        raise ValueError("Planner did not return a valid response.")
    return {"plan": resp}


def architect_agent(state: dict) -> dict:
    """Creates TaskPlan from Plan."""
    plan: Plan = state["plan"]
    resp = llm.with_structured_output(TaskPlan, method="function_calling").invoke(
        architect_prompt(plan=plan.model_dump_json())
    )
    if resp is None:
        raise ValueError("Planner did not return a valid response.")

    resp.plan = plan
    print(resp.model_dump_json())
    return {"task_plan": resp}


def coder_agent(state: dict) -> dict:
    """LangGraph tool-using coder agent."""
    coder_state: CoderState = state.get("coder_state")
    if coder_state is None:
        coder_state = CoderState(task_plan=state["task_plan"], current_step_idx=0)

    steps = coder_state.task_plan.implementation_steps
    if coder_state.current_step_idx >= len(steps):
        return {"coder_state": coder_state, "status": "DONE"}

    current_task = steps[coder_state.current_step_idx]
    existing_content = read_file.run(current_task.filePath)

    system_prompt = coder_system_prompt()
    user_prompt = (
        f"Task: {current_task.task_description}\n"
        f"File: {current_task.filePath}\n"
        f"Existing content:\n{existing_content}\n"
        "Use write_file(path, content) to save your changes."
    )

    coder_tools = [read_file, write_file, list_file, get_current_directory]
    react_agent = create_react_agent(llm, coder_tools)

    react_agent.invoke({"messages": [{"role": "system", "content": system_prompt},
                                     {"role": "user", "content": user_prompt}]})

    coder_state.current_step_idx += 1
    return {"coder_state": coder_state}


graph = StateGraph(dict)

graph.add_node("planner", planner_agent)
graph.add_node("architect", architect_agent)
graph.add_node("coder", coder_agent)

graph.add_edge("planner", "architect")
graph.add_edge("architect", "coder")
graph.add_conditional_edges(
    "coder",
    lambda s: "END" if s.get("status") == "DONE" else "coder",
    {"END": END, "coder": "coder"}
)

graph.set_entry_point("planner")
agent = graph.compile()
if __name__ == "__main__":
    project_root = init_project_root.invoke({})
    print(f"Project root: {project_root}")
    set_project_root(Path(project_root))
    result = agent.invoke({"user_prompt": "Create a simple calculator app"},
                          config={"recursion_limit": 100})
    print("Final State:", result)