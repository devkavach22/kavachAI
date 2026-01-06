"""
SQL Agent using LangGraph for querying databases.

This module provides a SQL agent that can:
- Connect to databases using a provided URL
- Query specific tables
- Execute natural language queries against the database
"""

import os
from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

# Load environment variables
load_dotenv()


# Define the agent state
class AgentState(TypedDict):
    """State for the SQL agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    database_url: str
    table_name: str
    query: str


class SQLAgent:
    """SQL Agent using LangGraph for database queries."""

    def __init__(self, llm_model: str = "gpt-3.5-turbo", temperature: float = 0):
        """
        Initialize the SQL Agent.

        Args:
            llm_model: The LLM model to use (default: gpt-3.5-turbo)
            temperature: Temperature for the LLM (default: 0)
        """
        self.llm = ChatOpenAI(model=llm_model, temperature=temperature)
        self.graph = None

    def create_agent(self, database_url: str, table_name: str = None):
        """
        Create a SQL agent for the given database.

        Args:
            database_url: Database connection URL
            table_name: Optional specific table to focus on

        Returns:
            Compiled LangGraph agent
        """
        # Connect to the database
        db = SQLDatabase.from_uri(database_url)

        # Create SQL toolkit
        toolkit = SQLDatabaseToolkit(db=db, llm=self.llm)
        tools = toolkit.get_tools()

        # Bind tools to LLM
        llm_with_tools = self.llm.bind_tools(tools)

        # Define the agent node
        def agent_node(state: AgentState):
            """Process the agent's reasoning and tool calls."""
            messages = state["messages"]

            # Add context about the table if specified
            if state.get("table_name"):
                context_msg = f"Focus on the table: {state['table_name']}"
                messages = [HumanMessage(content=context_msg)] + list(messages)

            response = llm_with_tools.invoke(messages)
            return {"messages": [response]}

        # Define the decision function
        def should_continue(state: AgentState):
            """Determine if we should continue or end."""
            messages = state["messages"]
            last_message = messages[-1]

            # If there are no tool calls, we're done
            if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
                return "end"
            return "continue"

        # Create the graph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", ToolNode(tools))

        # Set entry point
        workflow.set_entry_point("agent")

        # Add conditional edges
        workflow.add_conditional_edges(
            "agent", should_continue, {"continue": "tools", "end": END}
        )

        # Add edge from tools back to agent
        workflow.add_edge("tools", "agent")

        # Compile the graph
        self.graph = workflow.compile()
        return self.graph

    def query_database(
        self,
        database_url: str,
        query: str,
        table_name: str = None,
        verbose: bool = True,
    ):
        """
        Query the database using natural language.

        Args:
            database_url: Database connection URL
            query: Natural language query
            table_name: Optional specific table to query
            verbose: Whether to print intermediate steps

        Returns:
            The agent's response
        """
        # Create the agent
        agent = self.create_agent(database_url, table_name)

        # Prepare initial state
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "database_url": database_url,
            "table_name": table_name or "",
            "query": query,
        }

        # Run the agent
        result = agent.invoke(initial_state)

        if verbose:
            print("\n=== Agent Execution ===")
            for message in result["messages"]:
                if isinstance(message, HumanMessage):
                    print(f"\n[Human]: {message.content}")
                elif isinstance(message, AIMessage):
                    print(f"\n[AI]: {message.content}")
                    if hasattr(message, "tool_calls") and message.tool_calls:
                        print(f"[Tool Calls]: {message.tool_calls}")

        # Extract final answer
        final_message = result["messages"][-1]
        return (
            final_message.content
            if hasattr(final_message, "content")
            else str(final_message)
        )


def run_sql_query(
    database_url: str,
    query: str,
    table_name: str = None,
    llm_model: str = "gpt-3.5-turbo",
    verbose: bool = True,
):
    """
    Convenience function to run a SQL query using the agent.

    Args:
        database_url: Database connection URL (e.g., "sqlite:///mydb.db" or "postgresql://user:pass@host/db")
        query: Natural language query
        table_name: Optional specific table to query
        llm_model: LLM model to use
        verbose: Whether to print intermediate steps

    Returns:
        The query result

    Example:
        >>> result = run_sql_query(
        ...     database_url="sqlite:///chinook.db",
        ...     query="How many customers are there?",
        ...     table_name="customers"
        ... )
    """
    agent = SQLAgent(llm_model=llm_model)
    return agent.query_database(
        database_url=database_url, query=query, table_name=table_name, verbose=verbose
    )


if __name__ == "__main__":
    # Example usage with MySQL database
    print("SQL Agent with LangGraph")
    print("=" * 50)

    # MySQL database configuration
    DB_USER = "root"
    DB_PASSWORD = "Kavach1234"
    DB_HOST = "localhost"
    DB_PORT = "3306"
    DB_NAME = "excel_db"
    TABLE_NAME = "excel_data_temp"

    DATABASE_URL = (
        f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

    print(f"\nConnecting to: {DB_NAME}")
    print(f"Table: {TABLE_NAME}\n")

    # Example queries you can run
    example_queries = [
        "How many records are in the table?",
        "What are the column names and their types?",
        "Show me a sample of 3 rows from the table",
    ]

    # Uncomment to run a query:

    query = "Which employee has the highest salary? and show his salary also"

    result = run_sql_query(
        database_url=DATABASE_URL,
        query=query,
        table_name=TABLE_NAME,
        llm_model="gpt-3.5-turbo",
        verbose=True,
    )
    
    

    print(f"\n{'=' * 70}")
    print(f"Final Answer: {result}")
    print("=" * 70)

    """
    print("\nExample queries you can try:")
    for i, q in enumerate(example_queries, 1):
        print(f"  {i}. {q}")

    print(
        "\nTo run a query, uncomment the code block above and modify the 'query' variable."
    )
    """
