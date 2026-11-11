import asyncio
from pathlib import Path

import chromadb

try:
    import dotenv
except Exception:  # pragma: no cover
    dotenv = None

try:
    from agents import Agent, Runner, function_tool, trace
except Exception:  # pragma: no cover
    Agent = Runner = trace = None
    def function_tool(func):
        return func


repo_root = Path(__file__).resolve().parents[1]
chroma_client = chromadb.PersistentClient(str(repo_root / "chroma"))
hr_support_db = chroma_client.get_collection(name="hr_support_qna")


@function_tool
def hr_support_lookup_tool(query: str, max_results: int = 3) -> str:
    """Search the HR support ticket knowledge base for relevant employee support answers."""
    results = hr_support_db.query(query_texts=[query], n_results=max_results)

    if not results["documents"][0]:
        return f"No HR support guidance found for: {query}"

    formatted_results = []
    for doc in results["documents"][0]:
        formatted_results.append(doc)

    return "Relevant HR support answers:\n" + "\n\n".join(formatted_results)


hr_support_agent = Agent(
    name="HR Support Assistant",
    instructions="""
    You are a helpful HR support assistant who responds to employee questions.
    Always use the hr_support_lookup_tool first when a question is related to leave,
    payroll, benefits, reimbursements, onboarding, policies, grievances, or employee support.
    Keep answers concise, clear, and practical.
    """,
    tools=[hr_support_lookup_tool],
)


async def run_example():
    if dotenv is not None:
        dotenv.load_dotenv()

    with trace("HR Support Assistant with HR support RAG"):
        result = await Runner.run(
            hr_support_agent,
            "How do I request annual leave from HR?",
        )
        print(result.final_output)


if __name__ == "__main__":
    asyncio.run(run_example())
