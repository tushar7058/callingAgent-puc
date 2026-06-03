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
    room_io,
)
from livekit.plugins import ai_coustics, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from agents.sales_agent import DefaultAgent
from config.language_audio import build_stt, build_tts, resolve_language_profile
from livekit.plugins import google

logger = logging.getLogger("IRONMAN")

load_dotenv(".env")

server = AgentServer()

def prewarm(proc: JobProcess):
    # Aggressively reduce silence duration so the AI responds faster when user stops speaking
    proc.userdata["vad"] = silero.VAD.load(min_silence_duration=0.4)

server.setup_fnc = prewarm

@server.rtc_session(agent_name="IRONMAN")
async def entrypoint(ctx: JobContext):
    # Wait for the SIP caller to join the room
    participant = await ctx.wait_for_participant()

    if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
        # Retrieve caller ID
        phone_number = participant.attributes.get('sip.phoneNumber', 'unknown')
        logger.info(f"SIP caller joined from number: {phone_number}")

    default_profile = resolve_language_profile("English")

    session = AgentSession(
        stt=build_stt(default_profile),
        llm=google.LLM(
            model="gemini-3.1-flash-lite",
            # max_output_tokens= 100,
            # temperature= 0,
        ),
        tts=build_tts(default_profile),
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
