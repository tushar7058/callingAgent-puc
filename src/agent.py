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

from agents.sales_agent import DefaultAgent
from config.language_audio import build_stt, build_tts, resolve_language_profile
from utils.logging_config import setup_daily_file_logging
from openai.types.beta.realtime.session import TurnDetection
from config.language_audio import build_tts
from livekit.plugins import google

load_dotenv(".env")
setup_daily_file_logging()

logger = logging.getLogger("interview-agent")


# def build_llm():
#     return llm.FallbackAdapter(
#         [
#             openai.LLM(
#                 model=os.getenv("OPENAI_MODEL"),
#                 temperature=0,
#                 max_completion_tokens=200,
#             ),
#             google.LLM(
#                 model=os.getenv("GOOGLE_MODEL"),
#                 vertexai=True,
#                 credentials="key.json",
#                 max_output_tokens=200,
#             ),
#             groq.LLM(
#                 model=os.getenv("GROQ_MODEL"),
#                 temperature=0,
#                 api_key=os.getenv("GROQ_API_KEY"),
#                 max_completion_tokens=200,
#             ),
#         ],
#         attempt_timeout=15.0,  # Gemini requires minimum 10s deadline; default 5s causes 400 errors
#     )


def prewarm(proc: JobProcess):
    # min_silence_duration=0.4 keeps response latency low once the user has
    # genuinely stopped talking.
    # min_speech_duration=0.15 and a slightly higher activation_threshold
    # filter out short noise bursts (breaths, mic bumps, room noise) so they
    # don't get classified as "user started speaking" and falsely interrupt
    # the agent mid-sentence.
    proc.userdata["vad"] = silero.VAD.load(
        min_silence_duration=0.4,
        min_speech_duration=0.15,
        activation_threshold=0.6,
    )
    logger.info("agent prewarm complete: silero VAD loaded")


def build_turn_handling_options(turn_detection=None) -> TurnHandlingOptions:
    return TurnHandlingOptions(
        turn_detection=turn_detection or "vad",
        endpointing={
            "mode": "fixed",
            # Raised from 0.35 -> 0.5: gives short noise spikes more room to
            # decay before the turn is considered started, reducing false
            # "resumed false interrupted speech" cycles.
            "min_delay": 0.5,
            "max_delay": 1.2,
        },
        interruption={
            "resume_false_interruption": True,
            "false_interruption_timeout": 0.8,
        },
        preemptive_generation={
            "enabled": True,
            "preemptive_tts": True,
            "max_speech_duration": 8.0,
            "max_retries": 2,
        },
        user_turn_limit={
            "max_duration": 30.0,
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
        ctx.proc.userdata["sip_phone_number"] = (
            None if phone_number == "unknown" else phone_number
        )
        logger.info("SIP caller joined from number: %s", phone_number)

    default_profile = resolve_language_profile("English")

    session = AgentSession(
        llm=openai.realtime.RealtimeModel(voice="marin"),
        tts=build_tts(default_profile),
    )
    ctx.proc.userdata["dc_results"] = None
    ctx.proc.userdata["dc_task_results"] = {}
    ctx.proc.userdata["dc_external_sent"] = False
    ctx.proc.userdata["speech_language_code"] = default_profile.code
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
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name="interview-agent",
        )
    )