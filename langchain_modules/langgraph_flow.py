from langgraph.graph import StateGraph
from langchain.schema.runnable import RunnableLambda
from typing import TypedDict 
from .qa_chain import answer_question


# Define the state schema for the graph
class GraphState(TypedDict):
    question: str
    answer: str


def qa_flow_graph():
    builder = StateGraph(GraphState)
    builder.add_node("question_handler", RunnableLambda(lambda state: answer_question(state["question"])))

    builder.set_entry_point("question_handler")

    graph = builder.compile()
    return graph
