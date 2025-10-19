import os
from dotenv import load_dotenv
from typing import cast
import chainlit as cl
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, handoff
from agents.run import RunConfig, RunContextWrapper

# -------------------------------------------------------------------
# Load environment variables
# -------------------------------------------------------------------
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("❌ GEMINI_API_KEY is not set. Please define it in your .env file.")

# -------------------------------------------------------------------
# Chat start event
# -------------------------------------------------------------------
@cl.on_chat_start
async def start():
    """Initialize the chat session and agents."""

    external_client = AsyncOpenAI(
        api_key=gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    model = OpenAIChatCompletionsModel(
        model="gemini-2.5-flash",
        openai_client=external_client
    )

    config = RunConfig(
        model=model,
        model_provider=external_client,
        tracing_disabled=True
    )

    # ---------------------------------------------------------------
    # Handoff callback
    # ---------------------------------------------------------------
    async def on_handoff(agent: Agent, ctx: RunContextWrapper[None]):
        agent_name = agent.name
        print("--------------------------------")
        print(f"Handing off to {agent_name}...")
        print("--------------------------------")

        await cl.Message(
            content=(
                f"🔄 **Handing off to {agent_name}...**\n\n"
                f"I'm transferring your request to our {agent_name.lower()} "
                f"who will be able to better assist you."
            ),
            author="System"
        ).send()

    # ---------------------------------------------------------------
    # Define sub-agents
    # ---------------------------------------------------------------
    billing_agent = Agent(
        name="Billing Agent",
        instructions="You are a billing agent.",
        model=model
    )

    refund_agent = Agent(
        name="Refund Agent",
        instructions="You are a refund agent.",
        model=model
    )

    # ---------------------------------------------------------------
    # Define main triage agent
    # ---------------------------------------------------------------
    triage_agent = Agent(
        name="Triage Agent",
        instructions="You are a triage agent.",
        model=model,
        handoffs=[
            handoff(billing_agent, on_handoff=lambda ctx: on_handoff(billing_agent, ctx)),
            handoff(refund_agent, on_handoff=lambda ctx: on_handoff(refund_agent, ctx))
        ]
    )

    # ---------------------------------------------------------------
    # Store session data
    # ---------------------------------------------------------------
    cl.user_session.set("agent", triage_agent)
    cl.user_session.set("config", config)
    cl.user_session.set("chat_history", [])

    await cl.Message(content="👋 Welcome to the Panaversity AI Assistant! How can I help you today?").send()


# -------------------------------------------------------------------
# Chat message handler
# -------------------------------------------------------------------
@cl.on_message
async def main(message: cl.Message):
    """Handle incoming user messages and generate model responses."""

    thinking_msg = cl.Message(content="🤔 Thinking...")
    await thinking_msg.send()

    agent: Agent = cast(Agent, cl.user_session.get("agent"))
    config: RunConfig = cast(RunConfig, cl.user_session.get("config"))

    # Retrieve and update chat history
    history = cl.user_session.get("chat_history") or []
    history.append({"role": "user", "content": message.content})

    try:
        # Run the agent (async safe)
        result = await Runner.run(agent, history, run_config=config)

        response_content = result.final_output

        # Update the temporary message
        thinking_msg.content = response_content
        await thinking_msg.update()

        # Append assistant response
        history.append({"role": "assistant", "content": response_content})

        # Save updated history
        cl.user_session.set("chat_history", history)

        print(f"✅ Chat history updated: {history}")

    except Exception as e:
        thinking_msg.content = f"⚠️ Error: {str(e)}"
        await thinking_msg.update()
        print(f"❌ Error: {str(e)}")
