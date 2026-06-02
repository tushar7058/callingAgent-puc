import logging
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    TurnHandlingOptions,
    cli,
    inference,
    room_io,
)
from livekit.plugins import ai_coustics, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from agents.sales_agent import DefaultAgent

logger = logging.getLogger("IRONMAN")

load_dotenv(".env")

server = AgentServer()

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

server.setup_fnc = prewarm

@server.rtc_session(agent_name="IRONMAN")
async def entrypoint(ctx: JobContext):
    # Wait for the SIP caller to join the room
    participant = await ctx.wait_for_participant()
    
    if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
        # Retrieve caller ID
        phone_number = participant.attributes.get('sip.phoneNumber', 'unknown')
        logger.info(f"SIP caller joined from number: {phone_number}")

    session = AgentSession(
        stt=inference.STT(model="elevenlabs/scribe_v2_realtime", language="en"),
        llm=inference.LLM(
            model="google/gemini-3.1-flash-lite",
            extra_kwargs={"reasoning_effort": "low"},
        ),
        # tts=inference.TTS(
        #     model="cartesia/sonic-3",
        #     voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        #     language="en"
        # ),
        tts = inference.TTS(
            model="elevenlabs/eleven_multilingual_v2",
            voice="ecp3DWciuUyW7BYM7II1",
            language="en"
        ),
        turn_handling=TurnHandlingOptions(turn_detection=MultilingualModel()),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )
    ctx.proc.userdata["dc_results"] = None

    await session.start(
        agent=DefaultAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=ai_coustics.audio_enhancement(
                    model=ai_coustics.EnhancerModel.QUAIL_VF_L,
                ),
            ),
        ),
    )

if __name__ == "__main__":
    cli.run_app(server)
