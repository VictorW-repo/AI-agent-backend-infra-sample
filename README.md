
# AI Agent Backend Infra — README

## Overview

This backend system powers an AI agent that converts natural language into SQL queries and handles other smart city analytics (predictions, reports, analysis). It routes queries through classification, processing, and formatting modules to produce clean responses and structured data outputs.

---

## 🔁 `middleware.py`

The **main orchestrator**. It:

* Classifies the user’s intent (`classify_task`)
* Routes the query to the correct handler (SQL, prediction, report, etc.)
* Manages token usage with summarization (`manage_conversation_history`)
* Formats inputs and responses for LLM interaction (`llm_query`, `_format_final_message`)
* Parses, downsizes, and processes LLM responses (`parse_json`, `_handle_size_reduction`)
* Handles streaming output for UI (`ask_with_streaming`)
* Main entry point is `ask(chat: Dict)`

---

## 🔍 Component Files

### `RAG.py`

* Handles retrieval-augmented generation classification (`classify_task`)
* Includes `critic_eval` for LLM response quality checking

### `query_api.py`

* Low-level wrapper for making LLM API calls (`llm_api`)

### `process_sql.py`

* Contains `query_internal_databases()` for actual SQL execution

### `middleware_helper_functions.py`

* Core utilities: timestamp parsing, string conversion, cleaning, token estimation, etc.

### `internal_report.py`

* Generates internal system-level reports (maintenance, health, KPIs)

### `report_table_formatting.py`

* Formats structured data into tables for display (`format_data_report_table`, etc.)

### `trend_analysis.py`

* Detects outliers, anomalies, and generates statistical summaries

### `light_failure.py` & `energy_consumption_prediction.py`

* Run ML predictions for streetlight failures or energy usage

### `logger_setup.py`

* Sets up logging for tracking query execution and errors

### `document_retrieval.py`

* Loads relevant documents for context-aware answering (RAG support)

---

## 🔁 Query Flow Summary

1. `ask()` → classify query intent
2. Route to specific handler (`handle_sql_request`, etc.)
3. Query SQL or ML backend
4. Format results, reduce size, attach trend info
5. Use second LLM call to generate natural language response
6. Output structured + readable data

---
