import asyncio
from copilot import CopilotClient, PermissionHandler

async def ask_model():
    client = CopilotClient()
    await client.start()
    session = await client.create_session(
        model="claude-haiku-4.5",
        on_permission_request=PermissionHandler.approve_all,
    )
    
    done = asyncio.Event()
    response_text = ""
    
    def on_event(event):
        nonlocal response_text
        event_type = getattr(event.type, 'value', event.type) if hasattr(event, 'type') else None
        
        if event_type == "assistant.message":
            response_text = event.data.content # if not streaming, it has all
        elif event_type == "session.idle":
            done.set()
        elif event_type == "error":
            print(f"Error:", event)
            done.set()

    session.on(on_event)
    await session.send("Hello, a quick check!")
    
    try:
        await asyncio.wait_for(done.wait(), timeout=15.0)
        print("Final Response:", response_text)
    except asyncio.TimeoutError:
        print("Timeout waiting for response!")
        
    await session.disconnect()
    await client.stop()

if __name__ == "__main__":
    asyncio.run(ask_model())
