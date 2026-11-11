import os

import chainlit as cl
import dotenv
from agents import InputGuardrailTripwireTriggered, Runner, SQLiteSession
from nutrition_agent import exa_search_mcp, nutrition_agent
from openai.types.responses import ResponseTextDeltaEvent

dotenv.load_dotenv()


@cl.on_chat_start
async def on_chat_start():
    session = SQLiteSession("conversation_history")
    cl.user_session.set("agent_session", session)
    # This is the only change in this file compared to the chatbot/agentic_chatbot.py file
    await exa_search_mcp.connect()


@cl.on_message
async def on_message(message: cl.Message):
    session = cl.user_session.get("agent_session")

    result = Runner.run_streamed(
        nutrition_agent,
        message.content,
        session=session,
    )

    msg = cl.Message(content="")
    async with cl.Step(name="Thinking") as step:
        step.output = "Working on your request..."
        await step.update()

        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(
                event.data, ResponseTextDeltaEvent
            ):
                await msg.stream_token(token=event.data.delta)

        step.output = "Response ready!"

    await msg.update()


@cl.password_auth_callback
def auth_callback(username: str, password: str):
    if (username, password) == (
        os.getenv("CHAINLIT_USERNAME"),
        os.getenv("CHAINLIT_PASSWORD"),
    ):
        return cl.User(
            identifier="admin",
            metadata={"role": "admin", "provider": "credentials"},
        )
    else:
        return None
