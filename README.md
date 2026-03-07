# Lette Inbox Copilot

Local Python + Streamlit MVP for property-management inbox triage.

It reads a JSON email dataset, groups by `thread_id`, classifies issue type, applies deterministic urgency scoring, and ranks threads for action. Optional LLM enrichment uses OpenAI for only:
- thread summary
- recommended next action

If OpenAI is not configured or fails, deterministic fallback text is used.

## Setup
1. Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Set environment variables for OpenAI:
```bash
export OPENAI_API_KEY=your_api_key_here
export OPENAI_MODEL=gpt-4o-mini
export OPENAI_TIMEOUT_S=20
```

4. (Optional) Override dataset path:
```bash
export DATASET_PATH=data/proptech-test-data.json
```

## Run
```bash
streamlit run app.py
```

Default dataset path used by the app:
- `data/proptech-test-data.json`

## Project Structure
- `app.py`: Streamlit UI (metrics, filters, ranked table, thread detail/timeline)
- `parsing.py`: JSON load, email flattening, properties dataframe, property enrichment
- `scoring.py`: rule-based issue classification and urgency scoring
- `llm.py`: OpenAI summary/action + deterministic fallback
- `pipeline.py`: thread-level aggregation pipeline
- `requirements.txt`: minimal dependencies
- `data/proptech-test-data.json`: sample dataset

## Environment Variables
- `OPENAI_API_KEY` (optional): enables LLM summary/action
- `OPENAI_MODEL` (optional, default `gpt-4o-mini`)
- `OPENAI_TIMEOUT_S` (optional, default `20`)
- `DATASET_PATH` (optional, default `data/proptech-test-data.json`)

## Known Limitations
- Single-file local runtime; no database/auth/background jobs.
- Keyword rules are deterministic and intentionally simple for hackathon speed.
- LLM outputs are best-effort and may vary; fallback remains deterministic.
- No automated test suite yet (sanity checks done via pipeline run + app run).

## Future Improvements
- Add unit tests for parsing/scoring/pipeline.
- Add configurable scoring weights and keyword dictionaries.
- Add richer thread explainability and SLA timers.
- Add CSV export for ranked queue.
