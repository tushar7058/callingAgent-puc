import logging
import os

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    AgentSession,
    JobContext,
    JobProcess,
    TurnHandlingOptions,
    WorkerOptions,
    cli,
    llm,
    room_io,
)
from livekit.plugins import ai_coustics, google, groq, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from agents.sales_agent import DefaultAgent
from config.language_audio import build_stt, build_tts, resolve_language_profile
from utils.logging_config import setup_daily_file_logging

load_dotenv(".env")
setup_daily_file_logging()

logger = logging.getLogger("interview-agent")


def prewarm(proc: JobProcess):
    # Aggressively reduce silence duration so the AI responds faster when user stops speaking
    proc.userdata["vad"] = silero.VAD.load(min_silence_duration=0.4)
    logger.info("agent prewarm complete: silero VAD loaded")


def build_turn_handling_options(turn_detection=None) -> TurnHandlingOptions:
    return TurnHandlingOptions(
        turn_detection=turn_detection or MultilingualModel(),
        endpointing={
            "mode": "fixed",
            "min_delay": 0.35,
            "max_delay": 2.0,
        },
        interruption={
            "mode": "adaptive",
            "min_duration": 0.35,
            "min_words": 0,
            "resume_false_interruption": True,
            "false_interruption_timeout": 1.5,
        },
        preemptive_generation={
            "enabled": True,
            "preemptive_tts": False,
        },
    )


async def entrypoint(ctx: JobContext):
    logger.info("agent job started")

    await ctx.connect()
    logger.info("agent connected to room")

    # Wait for the SIP caller to join the room
    participant = await ctx.wait_for_participant()
    logger.info(
        "participant joined: identity=%s kind=%s",
        getattr(participant, "identity", "unknown"),
        participant.kind,
    )

    if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
        # Retrieve caller ID
        phone_number = participant.attributes.get("sip.phoneNumber", "unknown")
        logger.info("SIP caller joined from number: %s", phone_number)

    default_profile = resolve_language_profile("English")

    session = AgentSession(
        stt=build_stt(default_profile),
        llm=llm.FallbackAdapter(
            [
                openai.LLM(
                    model=os.getenv("OPENAI_MODEL"),
                    temperature=0,
                ),
                google.LLM(
                    model=os.getenv("GOOGLE_MODEL"),
                    vertexai=True,
                ),
                groq.LLM(
                    model=os.getenv("GROQ_MODEL"),
                    temperature=0,
                    api_key=os.getenv("GROQ_API_KEY"),
                ),
            ],
            attempt_timeout=15.0,  # Gemini requires minimum 10s deadline; default 5s causes 400 errors
        ),
        tts=build_tts(default_profile),
        turn_handling=build_turn_handling_options(),
        vad=ctx.proc.userdata["vad"],
        min_consecutive_speech_delay=0.2,
    )
    ctx.proc.userdata["dc_results"] = None
    ctx.proc.userdata["dc_task_results"] = {}
    ctx.proc.userdata["dc_external_sent"] = False
    logger.info("agent session initialized; starting data collection agent")
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
    logger.info("agent session started")


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint, prewarm_fnc=prewarm, agent_name="interview-agent"
        )
    )
