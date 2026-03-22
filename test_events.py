import asyncio
import sys
from copilot import CopilotClient, PermissionHandler

async def test_events():
    client = CopilotClient()
    await client.start()
    session = await client.create_session(
        model="claude-haiku-4.5",
        on_permission_request=PermissionHandler.approve_all,
    )
    
    done = asyncio.Event()
    full_response = ""
    
    def on_event(event):
        nonlocal full_response
        try:
            event_type = getattr(event.type, 'value', event.type) if hasattr(event, 'type') else str(event)
            if event_type == "assistant.message":
                full_response = getattr(event.data, "content", str(event.data))
                print("Got chunk length:", len(full_response), flush=True)
            elif event_type == "session.idle":
                print("Session Idle explicitly hit", flush=True)
                done.set()
        except Exception as e:
            print("Error:", e)

    session.on(on_event)
    await session.send("Explain quantum physics in 2 paragraphs.")
    
    await asyncio.wait_for(done.wait(), timeout=30.0)
    print("\nFINAL TEXT:", full_response[:100], "...", flush=True)
    
    # Fast exit
    sys.exit(0)

if __name__ == "__main__":
    asyncio.run(test_events())
