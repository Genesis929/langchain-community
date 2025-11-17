from __future__ import annotations

import asyncio
import json
import logging
import threading
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional, Union

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import AsyncCallbackHandler, BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from langchain_core.utils import guard_import

def _jsonify_safely(data: Any) -> Any:
    """Recursively converts non-serializable objects to strings."""
    if isinstance(data, dict):
        return {key: _jsonify_safely(value) for key, value in data.items()}
    if isinstance(data, list):
        return [_jsonify_safely(item) for item in data]
    try:
        json.dumps(data)
        return data
    except (TypeError, OverflowError):
        return str(data)


def import_google_cloud_bigquery() -> Any:
    """Import google-cloud-bigquery and its dependencies."""
    return (
        guard_import("google.cloud.bigquery"),
        guard_import("google.auth", pip_name="google-auth"),
        guard_import("google.api_core.gapic_v1.client_info"),
        guard_import(
            "google.cloud.bigquery_storage_v1.services.big_query_write.async_client"
        ),
        guard_import(
            "google.cloud.bigquery_storage_v1.services.big_query_write.client"
        ),
        guard_import("google.cloud.bigquery_storage_v1"),
        guard_import("pyarrow"),
    )


class AsyncBigQueryCallbackHandler(AsyncCallbackHandler):
    """
    Callback Handler that logs to Google BigQuery.

    This handler captures key events during an agent's lifecycle—such as user
    interactions, tool executions, LLM requests/responses, and errors—and
    streams them to a BigQuery table for analysis and monitoring.

    It uses the BigQuery Write API for efficient, high-throughput streaming
    ingestion. If the destination table does not exist, the handler will
    attempt to create it based on a predefined schema.
    """

    def __init__(
        self,
        project_id: str,
        dataset_id: str,
        table_id: str = "agent_events",
    ):
        """Initializes the BigQueryCallbackHandler.

        Args:
          project_id: Google Cloud project ID.
          dataset_id: BigQuery dataset ID.
          table_id: BigQuery table ID for agent events.
        """
        super().__init__()
        (
            self.bigquery,
            self.google_auth,
            self.gapic_client_info,
            self.async_client,
            self.sync_client,
            self.bq_storage,
            self.pa,
        ) = import_google_cloud_bigquery()
        self.BigQueryWriteAsyncClient = self.async_client.BigQueryWriteAsyncClient
        self.BigQueryWriteClient = self.sync_client.BigQueryWriteClient
        self._project_id, self._dataset_id, self._table_id = (
            project_id,
            dataset_id,
            table_id,
        )
        self._bq_client = None
        self._write_client = None
        self._init_lock = asyncio.Lock()
        self._arrow_schema = None
        self._schema = [
            self.bigquery.SchemaField("timestamp", "TIMESTAMP"),
            self.bigquery.SchemaField("event_type", "STRING"),
            self.bigquery.SchemaField("run_id", "STRING"),
            self.bigquery.SchemaField("parent_run_id", "STRING"),
            self.bigquery.SchemaField("content", "STRING"),
            self.bigquery.SchemaField("serialized", "STRING"),
            self.bigquery.SchemaField("tags", "STRING"),
            self.bigquery.SchemaField("metadata", "STRING"),
            self.bigquery.SchemaField("error_message", "STRING"),
        ]
        self.action_records: list = []

    async def _ensure_init(self) -> bool:
        """Ensures BigQuery clients are initialized."""
        if self._write_client:
            return True
        async with self._init_lock:
            if self._write_client:
                return True
            try:
                creds, _ = await asyncio.to_thread(
                    self.google_auth.default,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )
                client_info = self.gapic_client_info.ClientInfo(
                    user_agent="langchain-bigquery-callback"
                )
                self._bq_client = self.bigquery.Client(
                    project=self._project_id, credentials=creds, client_info=client_info
                )

                # Create dataset and table asynchronously
                if self._bq_client:
                    # Run sync methods in a thread to avoid blocking the event loop.
                    await asyncio.to_thread(
                        self._bq_client.create_dataset, self._dataset_id, exists_ok=True
                    )
                    table = self.bigquery.Table(
                        f"{self._project_id}.{self._dataset_id}.{self._table_id}",
                        schema=self._schema,
                    )
                    await asyncio.to_thread(
                        self._bq_client.create_table, table, exists_ok=True
                    )

                self._write_client = self.BigQueryWriteAsyncClient(
                    credentials=creds,  # type: ignore
                    client_info=client_info,
                )
                self._arrow_schema = self._bq_to_arrow_schema(self._schema)
                return True
            except Exception as e:  # pylint: disable=broad-exception-caught
                logging.error("BQ Init Failed: %s", e)
                return False

    def _bq_to_arrow_scalars(self, bq_scalar: str) -> Any:
        """Converts a BigQuery scalar type string to a PyArrow data type."""
        _BQ_TO_ARROW_SCALARS = {
            "BOOL": self.pa.bool_(),
            "BOOLEAN": self.pa.bool_(),
            "BYTES": self.pa.binary(),
            "DATE": self.pa.date32(),
            "DATETIME": self.pa.timestamp("us", tz=None),
            "FLOAT": self.pa.float64(),
            "FLOAT64": self.pa.float64(),
            "GEOGRAPHY": self.pa.string(),
            "INT64": self.pa.int64(),
            "INTEGER": self.pa.int64(),
            "JSON": self.pa.string(),
            "NUMERIC": self.pa.decimal128(38, 9),
            "BIGNUMERIC": self.pa.decimal256(76, 38),
            "STRING": self.pa.string(),
            "TIME": self.pa.time64("us"),
            "TIMESTAMP": self.pa.timestamp("us", tz="UTC"),
        }
        return _BQ_TO_ARROW_SCALARS.get(bq_scalar)

    def _bq_to_arrow_data_type(self, field: Any) -> Any:
        """Converts a BigQuery schema field to a PyArrow data type."""
        if field.mode == "REPEATED":
            inner = self._bq_to_arrow_data_type(
                self.bigquery.SchemaField(
                    field.name,
                    field.field_type,
                    fields=field.fields,
                    range_element_type=getattr(field, "range_element_type", None),
                )
            )
            return self.pa.list_(inner) if inner else None

        field_type_upper = field.field_type.upper() if field.field_type else ""
        if field_type_upper in ("RECORD", "STRUCT"):
            arrow_fields = [
                self._bq_to_arrow_field(subfield) for subfield in field.fields
            ]
            return self.pa.struct(arrow_fields)

        constructor = self._bq_to_arrow_scalars(field_type_upper)
        if constructor:
            return constructor
        else:
            logging.warning(
                "Failed to convert BigQuery field '%s': unsupported type '%s'.",
                field.name,
                field.field_type,
            )
            return None

    def _bq_to_arrow_field(self, bq_field: Any) -> Any:
        """Converts a BigQuery SchemaField to a PyArrow Field."""
        arrow_type = self._bq_to_arrow_data_type(bq_field)
        if arrow_type:
            return self.pa.field(
                bq_field.name,
                arrow_type,
                nullable=(bq_field.mode != "REPEATED"),
            )
        return None

    def _bq_to_arrow_schema(self, bq_schema_list: List[Any]) -> Any:
        """Converts a list of BigQuery SchemaFields to a PyArrow Schema."""
        arrow_fields = [
            af for af in (self._bq_to_arrow_field(f) for f in bq_schema_list) if af
        ]
        return self.pa.schema(arrow_fields)

    async def _log(self, data: dict) -> None:
        """Schedules a log entry to be written."""
        row = {
            "timestamp": datetime.now(UTC),
            "event_type": None,
            "run_id": None,
            "parent_run_id": None,
            "content": None,
            "serialized": None,
            "tags": None,
            "metadata": None,
            "error_message": None,
        }
        row.update(data)

        await self._perform_write(row)

    async def _perform_write(self, row: dict) -> None:
        """Actual write operation."""
        try:
            if not await self._ensure_init() or not self._write_client or not self._arrow_schema:
                return

            pydict = {field.name: [row.get(field.name)] for field in self._arrow_schema}
            batch = self.pa.RecordBatch.from_pydict(pydict, schema=self._arrow_schema)

            write_stream = f"projects/{self._project_id}/datasets/{self._dataset_id}/tables/{self._table_id}/_default"
            request = self.bq_storage.types.AppendRowsRequest(
                write_stream=write_stream,
            )
            # Correctly attach Arrow data to the `arrow_rows` field.
            request.arrow_rows.writer_schema.serialized_schema = (
                self._arrow_schema.serialize().to_pybytes()
            )
            request.arrow_rows.rows.serialized_record_batch = (
                batch.serialize().to_pybytes()
            )

            # This is an async call
            # Write with protection against immediate cancellation
            async for resp in await asyncio.shield(
                self._write_client.append_rows(iter([request]))
            ):
                if resp.error.code != 0:
                    logging.error("BQ Write Error: %s", resp.error.message)

        except RuntimeError as e:
            if "Event loop is closed" not in str(e):
                logging.exception("BQ Runtime Error: %s", e)

        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.error("BQ Write Failed: %s", e)

    async def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when LLM starts."""
        data = {
            "event_type": "LLM_START",
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id),
            "content": json.dumps({"prompts": prompts}),
            "metadata": json.dumps(_jsonify_safely(kwargs.get("metadata", {}))),
            "serialized": json.dumps(_jsonify_safely(serialized))
            if serialized
            else None,
            "tags": json.dumps(_jsonify_safely(tags or [])),
        }
        await self._log(data)

    async def on_llm_new_token(
        self,
        token: str,
        *,
        chunk: Optional[Any] = None,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when a new token is generated."""
        data = {
            "event_type": "LLM_NEW_TOKEN",
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id),
            "content": json.dumps(
                {
                    "token": token,
                }
            ),
            "metadata": json.dumps(_jsonify_safely(kwargs.get("metadata", {}))),
            "tags": json.dumps(_jsonify_safely(tags or [])),
        }
        await self._log(data)

    async def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when a chat model starts."""
        message_dicts = [[msg.dict() for msg in m] for m in messages]
        data = {
            "event_type": "CHAT_MODEL_START",
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id),
            "content": json.dumps({"messages": _jsonify_safely(message_dicts)}),
            "metadata": json.dumps(_jsonify_safely(kwargs.get("metadata", {}))),
            "serialized": json.dumps(_jsonify_safely(serialized)) if serialized else None,
            "tags": json.dumps(_jsonify_safely(tags or [])),
        }
        await self._log(data)

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when LLM ends running."""
        metadata = kwargs.get("metadata") or {}
        for generations in response.generations:
            for generation in generations:
                data = {
                    "event_type": "LLM_RESPONSE",
                    "run_id": str(run_id),
                    "parent_run_id": str(parent_run_id),
                    "content": json.dumps({"response": generation.text}),
                    "metadata": json.dumps(_jsonify_safely(metadata)),
                    "tags": json.dumps(_jsonify_safely(tags or [])),
                }
                await self._log(data)

    async def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when LLM errors."""
        data = {
            "event_type": "LLM_ERROR",
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id),
            "content": None,
            "error_message": str(error),
            "metadata": json.dumps(_jsonify_safely(kwargs.get("metadata", {}))),
            "tags": json.dumps(_jsonify_safely(tags or [])),
        }
        await self._log(data)

    async def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when chain starts running."""
        data = {
            "event_type": "CHAIN_START",
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id),
            "content": json.dumps({"inputs": _jsonify_safely(inputs)}),
            "metadata": json.dumps(_jsonify_safely(kwargs.get("metadata", {}))),
            "serialized": json.dumps(_jsonify_safely(serialized)) if serialized else None,
            "tags": json.dumps(_jsonify_safely(tags or [])),
        }
        await self._log(data)

    async def on_text(
        self,
        text: str,
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run on arbitrary text."""
        data = {
            "event_type": "TEXT",
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id),
            "content": json.dumps({"text": text}),
            "metadata": json.dumps(_jsonify_safely(kwargs.get("metadata", {}))),
            "tags": json.dumps(_jsonify_safely(tags or [])),
        }
        await self._log(data)

    async def on_retriever_start(
        self,
        serialized: Dict[str, Any],
        query: str,
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when a retriever starts."""
        data = {
            "event_type": "RETRIEVER_START",
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id),
            "content": json.dumps({"query": query}),
            "metadata": json.dumps(_jsonify_safely(kwargs.get("metadata", {}))),
            "serialized": json.dumps(_jsonify_safely(serialized)) if serialized else None,
            "tags": json.dumps(_jsonify_safely(tags or [])),
        }
        await self._log(data)

    async def on_retriever_end(
        self,
        documents: Any,
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when a retriever ends."""
        docs = [doc.dict() for doc in documents]
        data = {
            "event_type": "RETRIEVER_END",
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id),
            "content": json.dumps({"documents": _jsonify_safely(docs)}),
            "metadata": json.dumps(_jsonify_safely(kwargs.get("metadata", {}))),
            "tags": json.dumps(_jsonify_safely(tags or [])),
        }
        await self._log(data)

    async def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when a retriever errors."""
        data = {
            "event_type": "RETRIEVER_ERROR",
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id),
            "content": None,
            "error_message": str(error),
            "metadata": json.dumps(_jsonify_safely(kwargs.get("metadata", {}))),
            "tags": json.dumps(_jsonify_safely(tags or [])),
        }
        await self._log(data)

    async def close(self) -> None:
        """
        Shuts down the callback handler, ensuring all logs are flushed and clients are
        properly closed. This should be called before application exit.

        Once your Langchain application has completed its tasks, ensure that you call
        the `close` method to finalize the logging process.
        """
        logging.info("BQ Callback: Shutdown started.")

        # Use getattr for safe access in case transport is not present.
        if self._write_client and hasattr(self._write_client, "close"):
            try:
                logging.info("BQ Callback: Closing write client.")
                await self._write_client.close()
            except Exception as e:  # pylint: disable=broad-exception-caught
                logging.warning("BQ Callback: Error closing write client: %s", e)
        if self._bq_client:
            try:
                self._bq_client.close()
            except Exception as e:  # pylint: disable=broad-exception-caught
                logging.warning("BQ Callback: Error closing BQ client: %s", e)

        self._write_client = None
        self._bq_client = None
        logging.info("BQ Callback: Shutdown complete.")

    async def on_chain_end(
        self,
        outputs: Union[Dict[str, Any], Any],
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when chain ends running."""
        data = {
            "event_type": "CHAIN_END",
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id),
            "content": json.dumps({"outputs": _jsonify_safely(outputs)}),
            "metadata": json.dumps(_jsonify_safely(kwargs.get("metadata", {}))),
            "tags": json.dumps(_jsonify_safely(kwargs.get("tags", []))),
        }
        await self._log(data)

    async def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when chain errors."""
        data = {
            "event_type": "CHAIN_ERROR",
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id),
            "content": None,
            "error_message": str(error),
            "metadata": json.dumps(_jsonify_safely(kwargs.get("metadata", {}))),
            "tags": json.dumps(_jsonify_safely(tags or [])),
        }
        await self._log(data)

    async def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when tool starts running."""
        data = {
            "event_type": "TOOL_START",
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id),
            "content": json.dumps({"input": input_str}),
            "metadata": json.dumps(_jsonify_safely(kwargs.get("metadata", {}))),
            "serialized": json.dumps(_jsonify_safely(serialized)) if serialized else None,
            "tags": json.dumps(_jsonify_safely(tags or [])),
        }
        await self._log(data)

    async def on_tool_end(
        self,
        output: Any,
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when tool ends running."""
        data = {
            "event_type": "TOOL_END",
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id),
            "content": json.dumps({"output": str(output)}),
            "metadata": json.dumps(_jsonify_safely(kwargs.get("metadata", {}))),
            "tags": json.dumps(_jsonify_safely(tags or [])),
        }
        await self._log(data)

    async def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when tool errors."""
        data = {
            "event_type": "TOOL_ERROR",
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id),
            "content": None,
            "error_message": str(error),
            "metadata": json.dumps(_jsonify_safely(kwargs.get("metadata", {}))),
            "tags": json.dumps(_jsonify_safely(tags or [])),
        }
        await self._log(data)

    async def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run on agent action."""
        data = {
            "event_type": "AGENT_ACTION",
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id),
            "content": json.dumps({"tool": action.tool, "input": str(action.tool_input)}),
            "metadata": json.dumps(_jsonify_safely(kwargs.get("metadata", {}))),
            "tags": json.dumps(_jsonify_safely(tags or [])),
        }
        await self._log(data)

    async def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when agent ends running."""
        data = {
            "event_type": "AGENT_FINISH",
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id),
            "content": json.dumps({"output": _jsonify_safely(finish.return_values)}),
            "metadata": json.dumps(_jsonify_safely(kwargs.get("metadata", {}))),
            "tags": json.dumps(_jsonify_safely(tags or [])),
        }
        await self._log(data)

class BigQueryCallbackHandler(BaseCallbackHandler):
    """
    Callback Handler that logs to Google BigQuery.

    This handler captures key events during an agent's lifecycle—such as user
    interactions, tool executions, LLM requests/responses, and errors—and
    streams them to a BigQuery table for analysis and monitoring.

    It uses the BigQuery Write API for efficient, high-throughput streaming
    ingestion. If the destination table does not exist, the handler will
    attempt to create it based on a predefined schema.
    """

    def __init__(
        self,
        project_id: str,
        dataset_id: str,
        table_id: str = "agent_events",
    ):
        """Initializes the BigQueryCallbackHandler.

        Args:
          project_id: Google Cloud project ID.
          dataset_id: BigQuery dataset ID.
          table_id: BigQuery table ID for agent events.
        """
        super().__init__()
        (
            self.bigquery,
            self.google_auth,
            self.gapic_client_info,
            _,  # async_client
            self.sync_client,
            self.bq_storage,
            self.pa,
        ) = import_google_cloud_bigquery()
        self.BigQueryWriteClient = self.sync_client.BigQueryWriteClient
        self._project_id, self._dataset_id, self._table_id = (
            project_id,
            dataset_id,
            table_id,
        )
        self._bq_client = None
        self._write_client = None
        self._init_lock = threading.Lock()
        self._arrow_schema = None
        self._schema = [
            self.bigquery.SchemaField("timestamp", "TIMESTAMP"),
            self.bigquery.SchemaField("event_type", "STRING"),
            self.bigquery.SchemaField("run_id", "STRING"),
            self.bigquery.SchemaField("parent_run_id", "STRING"),
            self.bigquery.SchemaField("content", "STRING"),
            self.bigquery.SchemaField("serialized", "STRING"),
            self.bigquery.SchemaField("tags", "STRING"),
            self.bigquery.SchemaField("metadata", "STRING"),
            self.bigquery.SchemaField("error_message", "STRING"),
        ]
        self.action_records: list = []

    def _ensure_init(self) -> bool:
        """Ensures BigQuery clients are initialized."""
        if self._write_client:
            return True
        with self._init_lock:
            if self._write_client:
                return True
            try:
                creds, _ = self.google_auth.default(
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )
                client_info = self.gapic_client_info.ClientInfo(
                    user_agent="langchain-bigquery-callback"
                )
                self._bq_client = self.bigquery.Client(
                    project=self._project_id, credentials=creds, client_info=client_info
                )

                if self._bq_client:
                    self._bq_client.create_dataset(self._dataset_id, exists_ok=True)
                    table = self.bigquery.Table(
                        f"{self._project_id}.{self._dataset_id}.{self._table_id}",
                        schema=self._schema,
                    )
                    self._bq_client.create_table(table, exists_ok=True)

                self._write_client = self.BigQueryWriteClient(
                    credentials=creds,  # type: ignore
                    client_info=client_info,
                )
                self._arrow_schema = self._bq_to_arrow_schema(self._schema)
                return True
            except Exception as e:  # pylint: disable=broad-exception-caught
                logging.error("BQ Init Failed: %s", e)
                return False

    def _bq_to_arrow_scalars(self, bq_scalar: str) -> Any:
        """Converts a BigQuery scalar type string to a PyArrow data type."""
        _BQ_TO_ARROW_SCALARS = {
            "BOOL": self.pa.bool_(),
            "BOOLEAN": self.pa.bool_(),
            "BYTES": self.pa.binary(),
            "DATE": self.pa.date32(),
            "DATETIME": self.pa.timestamp("us", tz=None),
            "FLOAT": self.pa.float64(),
            "FLOAT64": self.pa.float64(),
            "GEOGRAPHY": self.pa.string(),
            "INT64": self.pa.int64(),
            "INTEGER": self.pa.int64(),
            "JSON": self.pa.string(),
            "NUMERIC": self.pa.decimal128(38, 9),
            "BIGNUMERIC": self.pa.decimal256(76, 38),
            "STRING": self.pa.string(),
            "TIME": self.pa.time64("us"),
            "TIMESTAMP": self.pa.timestamp("us", tz="UTC"),
        }
        return _BQ_TO_ARROW_SCALARS.get(bq_scalar)

    def _bq_to_arrow_data_type(self, field: Any) -> Any:
        """Converts a BigQuery schema field to a PyArrow data type."""
        if field.mode == "REPEATED":
            inner = self._bq_to_arrow_data_type(
                self.bigquery.SchemaField(
                    field.name,
                    field.field_type,
                    fields=field.fields,
                    range_element_type=getattr(field, "range_element_type", None),
                )
            )
            return self.pa.list_(inner) if inner else None

        field_type_upper = field.field_type.upper() if field.field_type else ""
        if field_type_upper in ("RECORD", "STRUCT"):
            arrow_fields = [
                self._bq_to_arrow_field(subfield) for subfield in field.fields
            ]
            return self.pa.struct(arrow_fields)

        constructor = self._bq_to_arrow_scalars(field_type_upper)
        if constructor:
            return constructor
        else:
            logging.warning(
                "Failed to convert BigQuery field '%s': unsupported type '%s'.",
                field.name,
                field.field_type,
            )
            return None

    def _bq_to_arrow_field(self, bq_field: Any) -> Any:
        """Converts a BigQuery SchemaField to a PyArrow Field."""
        arrow_type = self._bq_to_arrow_data_type(bq_field)
        if arrow_type:
            return self.pa.field(
                bq_field.name,
                arrow_type,
                nullable=(bq_field.mode != "REPEATED"),
            )
        return None

    def _bq_to_arrow_schema(self, bq_schema_list: List[Any]) -> Any:
        """Converts a list of BigQuery SchemaFields to a PyArrow Schema."""
        arrow_fields = [
            af for af in (self._bq_to_arrow_field(f) for f in bq_schema_list) if af
        ]
        return self.pa.schema(arrow_fields)

    def _log(self, data: dict) -> None:
        """Schedules a log entry to be written."""
        row = {
            "timestamp": datetime.now(UTC),
            "event_type": None,
            "run_id": None,
            "parent_run_id": None,
            "content": None,
            "serialized": None,
            "tags": None,
            "metadata": None,
            "error_message": None,
        }
        row.update(data)

        self._perform_write(row)

    def _perform_write(self, row: dict) -> None:
        """Actual write operation."""
        try:
            if not self._ensure_init() or not self._write_client or not self._arrow_schema:
                return

            pydict = {field.name: [row.get(field.name)] for field in self._arrow_schema}
            batch = self.pa.RecordBatch.from_pydict(pydict, schema=self._arrow_schema)

            write_stream = f"projects/{self._project_id}/datasets/{self._dataset_id}/tables/{self._table_id}/_default"
            request = self.bq_storage.types.AppendRowsRequest(
                write_stream=write_stream,
            )
            request.arrow_rows.writer_schema.serialized_schema = (
                self._arrow_schema.serialize().to_pybytes()
            )
            request.arrow_rows.rows.serialized_record_batch = (
                batch.serialize().to_pybytes()
            )

            # This is a sync call
            resp = self._write_client.append_rows(iter([request]))
            for r in resp:
                if r.error.code != 0:
                    logging.error("BQ Write Error: %s", r.error.message)

        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.error("BQ Write Failed: %s", e)

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when LLM starts."""
        data = {
            "event_type": "LLM_START",
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id),
            "content": json.dumps({"prompts": prompts}),
            "metadata": json.dumps(_jsonify_safely(kwargs.get("metadata", {}))),
            "serialized": json.dumps(_jsonify_safely(serialized))
            if serialized
            else None,
            "tags": json.dumps(_jsonify_safely(tags or [])),
        }
        self._log(data)

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when a chat model starts."""
        message_dicts = [[msg.dict() for msg in m] for m in messages]
        data = {
            "event_type": "CHAT_MODEL_START",
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id),
            "content": json.dumps({"messages": _jsonify_safely(message_dicts)}),
            "metadata": json.dumps(_jsonify_safely(kwargs.get("metadata", {}))),
            "serialized": json.dumps(_jsonify_safely(serialized)) if serialized else None,
            "tags": json.dumps(_jsonify_safely(tags or [])),
        }
        self._log(data)

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when LLM ends running."""
        metadata = kwargs.get("metadata") or {}
        for generations in response.generations:
            for generation in generations:
                data = {
                    "event_type": "LLM_RESPONSE",
                    "run_id": str(run_id),
                    "parent_run_id": str(parent_run_id),
                    "content": json.dumps({"response": generation.text}),
                    "metadata": json.dumps(_jsonify_safely(metadata)),
                    "tags": json.dumps(_jsonify_safely(tags or [])),
                }
                self._log(data)

    def on_llm_new_token(
        self,
        token: str,
        *,
        chunk: Optional[Any] = None,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when a new token is generated."""
        data = {
            "event_type": "LLM_NEW_TOKEN",
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id),
            "content": json.dumps(
                {
                    "token": token,
                }
            ),
            "metadata": json.dumps(_jsonify_safely(kwargs.get("metadata", {}))),
            "tags": json.dumps(_jsonify_safely(tags or [])),
        }
        self._log(data)

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when chain starts running."""
        data = {
            "event_type": "CHAIN_START",
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id),
            "content": json.dumps({"inputs": _jsonify_safely(inputs)}),
            "metadata": json.dumps(_jsonify_safely(kwargs.get("metadata", {}))),
            "serialized": json.dumps(_jsonify_safely(serialized))
            if serialized
            else None,
            "tags": json.dumps(_jsonify_safely(tags or [])),
        }
        self._log(data)

    def on_chain_end(
        self,
        outputs: Union[Dict[str, Any], Any],
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when chain ends running."""
        data = {
            "event_type": "CHAIN_END",
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id),
            "content": json.dumps({"outputs": _jsonify_safely(outputs)}),
            "metadata": json.dumps(_jsonify_safely(kwargs.get("metadata", {}))),
            "tags": json.dumps(_jsonify_safely(tags or [])),
        }
        self._log(data)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when chain errors."""
        data = {
            "event_type": "CHAIN_ERROR",
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id),
            "content": None,
            "error_message": str(error),
            "metadata": json.dumps(_jsonify_safely(kwargs.get("metadata", {}))),
            "tags": json.dumps(_jsonify_safely(tags or [])),
        }
        self._log(data)

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when tool starts running."""
        data = {
            "event_type": "TOOL_START",
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id),
            "content": json.dumps({"input": input_str}),
            "metadata": json.dumps(_jsonify_safely(kwargs.get("metadata", {}))),
            "serialized": json.dumps(_jsonify_safely(serialized))
            if serialized
            else None,
            "tags": json.dumps(_jsonify_safely(tags or [])),
        }
        self._log(data)

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when tool ends running."""
        data = {
            "event_type": "TOOL_END",
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id),
            "content": json.dumps({"output": str(output)}),
            "metadata": json.dumps(_jsonify_safely(kwargs.get("metadata", {}))),
            "tags": json.dumps(_jsonify_safely(tags or [])),
        }
        self._log(data)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when tool errors."""
        data = {
            "event_type": "TOOL_ERROR",
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id),
            "content": None,
            "error_message": str(error),
            "metadata": json.dumps(_jsonify_safely(kwargs.get("metadata", {}))),
            "tags": json.dumps(_jsonify_safely(tags or [])),
        }
        self._log(data)

    def on_text(
        self,
        text: str,
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run on arbitrary text."""
        data = {
            "event_type": "TEXT",
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id),
            "content": json.dumps({"text": text}),
            "metadata": json.dumps(_jsonify_safely(kwargs.get("metadata", {}))),
            "tags": json.dumps(_jsonify_safely(tags or [])),
        }
        self._log(data)

    def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run on agent action."""
        data = {
            "event_type": "AGENT_ACTION",
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id),
            "content": json.dumps(
                {"tool": action.tool, "input": str(action.tool_input)}
            ),
            "metadata": json.dumps(_jsonify_safely(kwargs.get("metadata", {}))),
            "tags": json.dumps(_jsonify_safely(tags or [])),
        }
        self._log(data)

    def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when agent ends running."""
        data = {
            "event_type": "AGENT_FINISH",
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id),
            "content": json.dumps({"output": _jsonify_safely(finish.return_values)}),
            "metadata": json.dumps(_jsonify_safely(kwargs.get("metadata", {}))),
            "tags": json.dumps(_jsonify_safely(tags or [])),
        }
        self._log(data)

    def on_retriever_start(
        self,
        serialized: Dict[str, Any],
        query: str,
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when a retriever starts."""
        data = {
            "event_type": "RETRIEVER_START",
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id),
            "content": json.dumps({"query": query}),
            "metadata": json.dumps(_jsonify_safely(kwargs.get("metadata", {}))),
            "serialized": json.dumps(_jsonify_safely(serialized))
            if serialized
            else None,
            "tags": json.dumps(_jsonify_safely(tags or [])),
        }
        self._log(data)

    def on_retriever_end(
        self,
        documents: Any,
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when a retriever ends."""
        docs = [doc.dict() for doc in documents]
        data = {
            "event_type": "RETRIEVER_END",
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id),
            "content": json.dumps({"documents": _jsonify_safely(docs)}),
            "metadata": json.dumps(_jsonify_safely(kwargs.get("metadata", {}))),
            "tags": json.dumps(_jsonify_safely(tags or [])),
        }
        self._log(data)

    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when a retriever errors."""
        data = {
            "event_type": "RETRIEVER_ERROR",
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id),
            "content": None,
            "error_message": str(error),
            "metadata": json.dumps(_jsonify_safely(kwargs.get("metadata", {}))),
            "tags": json.dumps(_jsonify_safely(tags or [])),
        }
        self._log(data)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when LLM errors."""
        data = {
            "event_type": "LLM_ERROR",
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id),
            "content": None,
            "error_message": str(error),
            "metadata": json.dumps(_jsonify_safely(kwargs.get("metadata", {}))),
            "tags": json.dumps(_jsonify_safely(tags or [])),
        }
        self._log(data)

    def close(self) -> None:
        """
        Shuts down the callback handler, ensuring all logs are flushed and clients are
        properly closed. This should be called before application exit.

        Once your Langchain application has completed its tasks, ensure that you call
        the `close` method to finalize the logging process.
        """
        logging.info("BQ Callback: Shutdown started.")

        if self._write_client and hasattr(self._write_client, "close"):
            try:
                logging.info("BQ Callback: Closing write client.")
                self._write_client.close()
            except Exception as e:  # pylint: disable=broad-exception-caught
                logging.warning("BQ Callback: Error closing write client: %s", e)
        if self._bq_client:
            try:
                self._bq_client.close()
            except Exception as e:  # pylint: disable=broad-exception-caught
                logging.warning("BQ Callback: Error closing BQ client: %s", e)

        self._write_client = None
        self._bq_client = None
        logging.info("BQ Callback: Shutdown complete.")
