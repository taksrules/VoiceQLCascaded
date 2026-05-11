"""
VoiceQL - Voice-to-SQL with Conversational Response Generation
Reimplementation of Song & Wong (2022) cascaded pipeline + novel response layer

Architecture:
  Mic -> VAD -> Whisper STT -> VoiceQLProcessor -> ElevenLabs TTS -> Speaker

VoiceQLProcessor internally:
  1. SchemaLoader    - reads SQLite schema once at startup
  2. PromptBuilder   - constructs schema-aware prompt with history
  3. SQLGenerator    - GPT-4o generates SQL + validation
  4. QueryExecutor   - runs SQL on SQLite
  5. ConversationMemory - stores turn history for multi-turn
  6. ResponseGenerator  - GPT-4o generates 3-part spoken response
"""
import asyncio
import os
import sys
import sqlite3
import json
from dotenv import load_dotenv
from openai import OpenAI

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.processors.aggregators.llm_response import (
    LLMUserContextAggregator,
    LLMAssistantContextAggregator,
)
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.transports.local.audio import (
    LocalAudioTransport,
    LocalAudioTransportParams,
)
from pipecat.frames.frames import TextFrame, LLMFullResponseEndFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.openai.tts import OpenAITTSService

load_dotenv()

# ── CONFIG ────────────────────────────────────────────────────────────────────
WHISPER_MODEL = "whisper-1"
ELEVENLABS_VOICE = "Rachel"  # Change to your preferred voice ID
TTS_VOICE = "alloy"          # OpenAI TTS voice (alloy, echo, fable, onyx, nova, shimmer)

# Path to Spider database - Relative to the script's directory
DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "database", "pets_1", "pets_1.sqlite")
SPIDER_DB_PATH = os.getenv("SPIDER_DB_PATH", DEFAULT_DB_PATH)

# Max retries for SQL generation
MAX_SQL_RETRIES = 3


# ── COMPONENT 2: SCHEMA LOADER ───────────────────────────────────────────────
class SchemaLoader:
    """Reads SQLite database schema once at startup and caches it."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.schema_text = ""
        self.table_names = []
        self.column_map = {}  # table -> [columns]
        self._load_schema()

    def _load_schema(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        self.table_names = [row[0] for row in cursor.fetchall()]

        schema_parts = []
        for table in self.table_names:
            # Get column info
            cursor.execute(f"PRAGMA table_info('{table}');")
            columns = cursor.fetchall()
            col_names = [col[1] for col in columns]
            col_types = [col[2] for col in columns]
            self.column_map[table] = col_names

            # Get foreign keys
            cursor.execute(f"PRAGMA foreign_key_list('{table}');")
            fks = cursor.fetchall()

            # Get sample rows (3 rows)
            cursor.execute(f"SELECT * FROM '{table}' LIMIT 3;")
            samples = cursor.fetchall()

            # Build schema text
            schema_parts.append(f"Table: {table}")
            col_strs = [f"  {name} ({ctype})" for name, ctype in zip(col_names, col_types)]
            schema_parts.append("  Columns:")
            for cs in col_strs:
                schema_parts.append(f"    {cs}")

            if fks:
                schema_parts.append("  Foreign Keys:")
                for fk in fks:
                    schema_parts.append(f"    {fk[3]} -> {fk[2]}.{fk[4]}")

            if samples:
                schema_parts.append("  Sample rows:")
                for row in samples:
                    schema_parts.append(f"    {row}")

            schema_parts.append("")

        conn.close()
        self.schema_text = "\n".join(schema_parts)
        print(f"📊 Schema loaded: {len(self.table_names)} tables")
        print(f"   Tables: {', '.join(self.table_names)}")

    def validate_sql(self, sql: str) -> tuple[bool, str]:
        """Check if SQL references valid tables and columns."""
        sql_upper = sql.upper()
        errors = []

        # Basic validation - check referenced tables exist
        for table in self.table_names:
            # Tables are valid, skip
            pass

        # Try to explain the query without executing
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(f"EXPLAIN {sql}")
            conn.close()
            return True, ""
        except Exception as e:
            return False, str(e)


# ── COMPONENT 5: CONVERSATION MEMORY ─────────────────────────────────────────
class ConversationMemory:
    """Stores conversation history for multi-turn context."""

    def __init__(self, max_turns: int = 10):
        self.turns = []
        self.max_turns = max_turns

    def add_turn(self, question: str, sql: str, results: str, response: str):
        self.turns.append({
            "question": question,
            "sql": sql,
            "results": results,
            "response": response,
        })
        # Keep only recent turns
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def get_history_text(self) -> str:
        if not self.turns:
            return "No previous conversation."

        parts = []
        for i, turn in enumerate(self.turns, 1):
            parts.append(f"Turn {i}:")
            parts.append(f"  User: {turn['question']}")
            parts.append(f"  SQL: {turn['sql']}")
            parts.append(f"  Results: {turn['results']}")
            parts.append(f"  Response: {turn['response']}")
        return "\n".join(parts)


# ── COMPONENT 3: PROMPT BUILDER ──────────────────────────────────────────────
class PromptBuilder:
    """Constructs schema-aware prompts for GPT-4o."""

    def __init__(self, schema_loader: SchemaLoader, memory: ConversationMemory):
        self.schema = schema_loader
        self.memory = memory

    def build_sql_prompt(self, user_question: str) -> str:
        return f"""You are an expert SQL generator. Given the database schema below,
                    generate a valid SQLite SQL query for the user's question.

                    DATABASE SCHEMA:
                    {self.schema.schema_text}

                    CONVERSATION HISTORY:
                    {self.memory.get_history_text()}

                    USER QUESTION: {user_question}

                    INSTRUCTIONS:
                    1. First identify which tables are relevant
                    2. Then identify which columns are needed
                    3. Generate a valid SQLite SQL query
                    4. Use only tables and columns that exist in the schema above
                    5. If the question references something from conversation history, use that context

                    Respond with ONLY the SQL query. No explanation, no markdown, no backticks."""

    def build_retry_prompt(self, user_question: str, failed_sql: str, error: str) -> str:
        return f"""You are an expert SQL generator. Your previous SQL query had an error.
                    Fix it based on the error message.

                    DATABASE SCHEMA:
                    {self.schema.schema_text}

                    USER QUESTION: {user_question}

                    FAILED SQL: {failed_sql}

                    ERROR: {error}

                    Generate a corrected SQLite SQL query. Respond with ONLY the SQL query.
                    No explanation, no markdown, no backticks."""

    def build_response_prompt(self, user_question: str, sql: str, results: str) -> str:
        return f"""You are VoiceQL, a conversational database assistant. The user asked
                    a question and you executed a SQL query. Generate a natural spoken response.

                    Your response MUST have exactly three parts separated by newlines:

                    INTENT: [One sentence confirming what the user asked]
                    QUERY: [One sentence describing what SQL query you ran in plain English - do NOT use raw SQL syntax]
                    INSIGHT: [2-3 sentences summarizing results conversationally. Note patterns or interesting findings. End with a follow-up suggestion.]

                    USER QUESTION: {user_question}
                    SQL EXECUTED: {sql}
                    QUERY RESULTS: {results}

                    CONVERSATION HISTORY:
                    {self.memory.get_history_text()}

                    Remember: This will be spoken aloud, so keep it natural and conversational.
                    Do not use technical jargon. Do not say "SQL" or "query" to the user.
                    Say things like "I looked up" or "I checked the database" instead."""


# ── COMPONENT 4 & 5: SQL GENERATOR + QUERY EXECUTOR ─────────────────────────
class SQLEngine:
    """Generates SQL via GPT-4o, validates, executes, and retries."""

    def __init__(self, schema_loader: SchemaLoader, prompt_builder: PromptBuilder):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.schema = schema_loader
        self.prompts = prompt_builder
        self.db_path = schema_loader.db_path

    def _call_gpt4o(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()

    def _clean_sql(self, raw: str) -> str:
        """Remove markdown backticks or extra text from GPT response."""
        sql = raw.strip()
        if sql.startswith("```sql"):
            sql = sql[6:]
        if sql.startswith("```"):
            sql = sql[3:]
        if sql.endswith("```"):
            sql = sql[:-3]
        return sql.strip()

    def _execute_sql(self, sql: str) -> tuple[bool, str]:
        """Execute SQL on SQLite and return results."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            conn.close()

            if not rows:
                return True, "No results found."

            # Format results nicely
            result_parts = []
            result_parts.append(f"Columns: {', '.join(columns)}")
            result_parts.append(f"Row count: {len(rows)}")
            for i, row in enumerate(rows[:20]):  # Limit to 20 rows for context
                result_parts.append(f"  Row {i+1}: {row}")
            if len(rows) > 20:
                result_parts.append(f"  ... and {len(rows) - 20} more rows")

            return True, "\n".join(result_parts)
        except Exception as e:
            return False, str(e)

    def generate_and_execute(self, user_question: str) -> tuple[str, str]:
        """
        Full pipeline: generate SQL -> validate -> execute -> retry if needed.
        Returns (sql, results) tuple.
        """
        # First attempt
        prompt = self.prompts.build_sql_prompt(user_question)
        raw_sql = self._call_gpt4o(prompt)
        sql = self._clean_sql(raw_sql)
        print(f"  🔧 SQL attempt 1: {sql}")

        for attempt in range(MAX_SQL_RETRIES):
            # Validate
            valid, val_error = self.schema.validate_sql(sql)
            if not valid:
                print(f"  ❌ Validation failed: {val_error}")
                retry_prompt = self.prompts.build_retry_prompt(user_question, sql, val_error)
                raw_sql = self._call_gpt4o(retry_prompt)
                sql = self._clean_sql(raw_sql)
                print(f"  🔧 SQL attempt {attempt + 2}: {sql}")
                continue

            # Execute
            success, results = self._execute_sql(sql)
            if success:
                print(f"  ✅ Query executed successfully")
                return sql, results
            else:
                print(f"  ❌ Execution failed: {results}")
                retry_prompt = self.prompts.build_retry_prompt(user_question, sql, results)
                raw_sql = self._call_gpt4o(retry_prompt)
                sql = self._clean_sql(raw_sql)
                print(f"  🔧 SQL attempt {attempt + 2}: {sql}")

        return sql, "Sorry, I could not execute that query after multiple attempts."


# ── COMPONENT 6: RESPONSE GENERATOR ─────────────────────────────────────────
class ResponseGenerator:
    """Generates 3-part conversational response from query results."""

    def __init__(self, prompt_builder: PromptBuilder):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.prompts = prompt_builder

    def generate(self, user_question: str, sql: str, results: str) -> str:
        prompt = self.prompts.build_response_prompt(user_question, sql, results)
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()


# ── COMPONENT 1: VOICEQL PROCESSOR (Pipecat integration) ────────────────────
class VoiceQLProcessor(FrameProcessor):
    """
    Custom Pipecat processor that replaces the LLM in the pipeline.
    Receives transcript text -> runs SQL pipeline -> outputs response text.
    """

    def __init__(self, db_path: str):
        super().__init__()

        # Initialize all components
        print("\n🔄 Initializing VoiceQL components...")
        self.schema_loader = SchemaLoader(db_path)
        self.memory = ConversationMemory()
        self.prompt_builder = PromptBuilder(self.schema_loader, self.memory)
        self.sql_engine = SQLEngine(self.schema_loader, self.prompt_builder)
        self.response_gen = ResponseGenerator(self.prompt_builder)
        print("✅ All VoiceQL components ready\n")

    async def process_frame(self, frame, direction):
        # DEBUG: Print frame type to trace pipeline flow
        if not isinstance(frame, (TextFrame, LLMFullResponseEndFrame)):
             # print(f"DEBUG: VoiceQL received frame: {type(frame).__name__}")
             pass
        
        await super().process_frame(frame, direction)

        # We only care about text frames coming from the STT
        if isinstance(frame, TextFrame) and direction == FrameDirection.DOWNSTREAM:
            transcript = frame.text.strip()
            if not transcript:
                return

            print(f"\n{'='*60}")
            print(f"🎤 User said: \"{transcript}\"")
            print(f"{'='*60}")

            try:
                # Step 1: Generate and execute SQL
                print("\n📝 Generating SQL...")
                sql, results = await asyncio.to_thread(self.sql_engine.generate_and_execute, transcript)

                # Step 2: Generate conversational response
                print("💬 Generating response...")
                response = await asyncio.to_thread(self.response_gen.generate, transcript, sql, results)

                # Step 3: Store in memory for multi-turn
                self.memory.add_turn(transcript, sql, results, response)

                print(f"\n🔊 Response:\n{response}\n")

                # Step 4: Push response to TTS
                await self.push_frame(TextFrame(text=response))
                await self.push_frame(LLMFullResponseEndFrame())

            except Exception as e:
                error_msg = f"I encountered an error processing your question. {str(e)}"
                print(f"❌ Error: {e}")
                await self.push_frame(TextFrame(text=error_msg))
                await self.push_frame(LLMFullResponseEndFrame())

        else:
            # Pass through any other frames
            await self.push_frame(frame, direction)


# ── PIPECAT CONTEXT (simplified for VoiceQL) ─────────────────────────────────
class VoiceQLContext:
    """Minimal context object for Pipecat aggregators."""

    def __init__(self):
        self.messages = [
            {
                "role": "system",
                "content": (
                    "You are VoiceQL, a voice-powered database assistant. "
                    "Users ask questions about data and you query the database "
                    "to provide conversational answers."
                ),
            }
        ]
        self.tools = []
        self.tool_choice = None

    def add_message(self, message):
        if isinstance(message, dict):
            self.messages.append(message)

    def get_messages(self, *args, **kwargs):
        return self.messages

    def get_messages_for_token_count(self):
        return self.messages

    def clear(self):
        self.messages = self.messages[:1]  # Keep system prompt


# ── MAIN ─────────────────────────────────────────────────────────────────────
async def main():
    print("\n" + "=" * 60)
    print("🗄️  VoiceQL — Voice-to-SQL with Conversational Responses")
    print("   Based on Song & Wong (2022) cascaded pipeline")
    print("   Extended with conversational response generation")
    print("=" * 60)
    print(f"   STT:  OpenAI Whisper ({WHISPER_MODEL})")
    print(f"   LLM:  GPT-4o (SQL generation + response)")
    print(f"   TTS:  ElevenLabs ({ELEVENLABS_VOICE})")
    print(f"   DB:   {SPIDER_DB_PATH}")
    print("=" * 60)

    # ── API key check ─────────────────────────────────────────────────────
    openai_key = os.getenv("OPENAI_API_KEY")
    elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")

    if not openai_key:
        print("❌ OPENAI_API_KEY missing in .env")
        sys.exit(1)
    if not elevenlabs_key:
        print("❌ ELEVENLABS_API_KEY missing in .env")
        sys.exit(1)

    # Check database exists
    if not os.path.exists(SPIDER_DB_PATH):
        print(f"❌ Database not found: {SPIDER_DB_PATH}")
        print("   Download Spider from: https://github.com/taoyds/spider")
        sys.exit(1)

    print("✅ All API keys loaded")

    # ── 1. Transport ──────────────────────────────────────────────────────
    transport = LocalAudioTransport(
        params=LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
            audio_in_channels=1,
            audio_out_channels=1, # Stereo is often required for Windows audio playback
            output_device_index=os.getenv("OUTPUT_DEVICE") if os.getenv("OUTPUT_DEVICE") else None, # Use system default unless specified
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    stop_secs=0.8,      # Allow for longer pauses (default 0.2 is too short for SQL questions)
                    min_volume=0.4      # Slightly more sensitive to low voices
                )
            ),
            vad_audio_passthrough=True,
        )
    )

    # ── 2. STT — OpenAI Whisper ───────────────────────────────────────────
    stt = OpenAISTTService(
        api_key=openai_key,
        model=WHISPER_MODEL,
        prompt="This is VoiceQL, a conversational database assistant for students and pets. Key terms: pets, pet, student, has_pet, PetID, PetType, StudentID, Fname, Lname. Always transcribe as 'pets' when referring to the database table.",
    )

    # ── 3. VoiceQL Processor (replaces LLM) ───────────────────────────────
    voiceql = VoiceQLProcessor(db_path=SPIDER_DB_PATH)

    # ── 4. TTS — ElevenLabs ───────────────────────────────────────────────
    # tts = ElevenLabsTTSService(
    #     api_key=elevenlabs_key,
    #     voice_id=ELEVENLABS_VOICE,
    # )
    # OpenAI TTS streams audio at 24kHz PCM. Change TTS_VOICE at the top of the file.
    tts = OpenAITTSService(
        api_key=openai_key,
        voice=TTS_VOICE,
        model="gpt-4o-mini-tts",
        sample_rate=24000,
    )

    # ── 5. Context aggregators ────────────────────────────────────────────
    context = VoiceQLContext()
    user_agg = LLMUserContextAggregator(context)
    assistant_agg = LLMAssistantContextAggregator(context)

    # ── 6. Pipeline ───────────────────────────────────────────────────────
    pipeline = Pipeline(
        [
            transport.input(),      # Mic audio
            stt,                    # Whisper: audio -> transcript
            voiceql,                # VoiceQL: transcript -> SQL -> response (MUST BE BEFORE AGG)
            user_agg,               # Add transcript to context
            tts,                    # ElevenLabs: text -> speech
            transport.output(),     # Speaker
            assistant_agg,          # Store response in context
        ]
    )

    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))
    runner = PipelineRunner()

    print("\n🎤 VoiceQL is ready. Ask a question about the database.")
    print("   Try: 'How many pets are in the database?'")
    print("   Try: 'What is the average age of students who have a cat?'")
    print("   Try: 'Show me the heaviest pet'")
    print("   Press Ctrl+C to stop.\n")

    await runner.run(task)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 VoiceQL offline.")