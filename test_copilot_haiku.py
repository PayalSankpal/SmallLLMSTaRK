import asyncio
from copilot import CopilotClient, PermissionHandler

async def main():
    print("Starting Copilot Client...", flush=True)
    client = CopilotClient()
    await client.start()

    model_name = "claude-haiku-4.5"
    print(f"\nCreating session with model: {model_name}...", flush=True)
    session = await client.create_session(
        model=model_name,
        on_permission_request=PermissionHandler.approve_all,
    )

    done = asyncio.Event()

    def on_event(event):
        # Using type.value might not be needed if it's a string, or maybe it's an enum.
        event_type = getattr(event.type, 'value', event.type) if hasattr(event, 'type') else None
        
        if event_type == "assistant.message":
            print("\nResponse from Copilot Agent:")
            print(event.data.content, flush=True)
            done.set()
        elif event_type == "session.idle":
            done.set()
        elif event_type == "error":
            print("Error occurred:", event, flush=True)
            done.set()

    session.on(on_event)

    prompt = "Hello! Please reply briefly. What model are you?"
    print(f"Sending prompt: {prompt}", flush=True)
    await session.send(prompt)
    
    try:
        await asyncio.wait_for(done.wait(), timeout=30.0)
    except asyncio.TimeoutError:
        print("\nTimed out waiting for response.", flush=True)

    print("\nCleaning up...", flush=True)
    await session.disconnect()
    await client.stop()

if __name__ == "__main__":
    asyncio.run(main())
