"""Unit tests for BigQueryCallbackHandler."""

from typing import Any, Dict, Generator
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.outputs import LLMResult
from langchain_core.messages import HumanMessage

from langchain_community.callbacks.bigquery_callback import (
    AsyncBigQueryCallbackHandler,
    BigQueryCallbackHandler,
)
from langchain_core.documents import Document

@pytest.fixture
def mock_bigquery_clients() -> Generator[Dict[str, Any], None, None]:
    """Mocks the BigQuery clients and dependencies."""
    with patch(
        "langchain_community.callbacks.bigquery_callback.import_google_cloud_bigquery"
    ) as mock_import:
        mock_bigquery = MagicMock()
        mock_google_auth = MagicMock()
        mock_gapic_client_info = MagicMock()
        mock_async_client_module = MagicMock()
        mock_sync_client_module = MagicMock()
        mock_bq_storage = MagicMock()
        mock_pa = MagicMock()

        mock_import.return_value = (
            mock_bigquery,
            mock_google_auth,
            mock_gapic_client_info,
            mock_async_client_module,
            mock_sync_client_module,
            mock_bq_storage,
            mock_pa,
        )

        # Mock the async client instance
        mock_async_write_client = AsyncMock()
        mock_async_client_module.BigQueryWriteAsyncClient.return_value = (
            mock_async_write_client
        )

        # Mock the sync client instance
        mock_sync_write_client = MagicMock()
        mock_sync_client_module.BigQueryWriteClient.return_value = (
            mock_sync_write_client
        )

        # Mock the sync BigQuery client instance
        mock_bq_client = MagicMock()
        mock_bigquery.Client.return_value = mock_bq_client

        # Mock google auth to avoid real authentication
        mock_google_auth.default = MagicMock(return_value=(None, "test-project"))

        yield {
            "mock_bigquery": mock_bigquery,
            "mock_google_auth": mock_google_auth,
            "mock_async_write_client": mock_async_write_client,
            "mock_sync_write_client": mock_sync_write_client,
            "mock_bq_client": mock_bq_client,
            "mock_pa": mock_pa,
        }


@pytest.fixture
async def handler(mock_bigquery_clients: Dict[str, Any]) -> AsyncBigQueryCallbackHandler:
    """
    Returns an initialized `AsyncBigQueryCallbackHandler` with mocked clients.
    """
    handler = AsyncBigQueryCallbackHandler(
        project_id="test-project",
        dataset_id="test_dataset",
        table_id="test_table",
    )
    # Ensure initialization is run
    await handler._ensure_init()
    return handler


@pytest.fixture
def sync_handler(
    mock_bigquery_clients: Dict[str, Any]
) -> BigQueryCallbackHandler:
    """
    Returns an initialized `BigQueryCallbackHandler` with mocked clients.
    """
    handler = BigQueryCallbackHandler(
        project_id="test-project",
        dataset_id="test_dataset",
        table_id="test_table",
    )
    # Ensure initialization is run
    handler._ensure_init()
    return handler


@pytest.mark.asyncio
async def test_async_on_llm_start(
    handler: AsyncBigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that on_llm_start logs the correct event."""
    run_id = uuid4()
    parent_run_id = uuid4()
    await handler.on_llm_start(
        serialized={"name": "test_llm"},
        prompts=["test prompt"],
        run_id=run_id,
        parent_run_id=parent_run_id,
    )

    mock_write_client = mock_bigquery_clients["mock_async_write_client"]
    assert mock_write_client.append_rows.call_count == 1


def test_sync_on_llm_start(
    sync_handler: BigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that sync on_llm_start logs the correct event."""
    run_id = uuid4()
    parent_run_id = uuid4()
    sync_handler.on_llm_start(
        serialized={"name": "test_llm"},
        prompts=["test prompt"],
        run_id=run_id,
        parent_run_id=parent_run_id,
    )

    mock_write_client = mock_bigquery_clients["mock_sync_write_client"]
    assert mock_write_client.append_rows.call_count == 1


@pytest.mark.asyncio
async def test_async_on_llm_end(
    handler: AsyncBigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that on_llm_end logs the correct event."""
    response = LLMResult(generations=[], llm_output={"model_name": "test_model"})
    await handler.on_llm_end(response, run_id=uuid4())

    # on_llm_end might not log if there are no generations. Let's add one.
    response.generations.append([MagicMock(text="test generation")])
    await handler.on_llm_end(response, run_id=uuid4())

    mock_write_client = mock_bigquery_clients["mock_async_write_client"]
    assert mock_write_client.append_rows.call_count == 1


def test_sync_on_llm_end(
    sync_handler: BigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that sync on_llm_end logs the correct event."""
    response = LLMResult(generations=[], llm_output={"model_name": "test_model"})
    sync_handler.on_llm_end(response, run_id=uuid4())

    # on_llm_end might not log if there are no generations. Let's add one.
    response.generations.append([MagicMock(text="test generation")])
    sync_handler.on_llm_end(response, run_id=uuid4())

    mock_write_client = mock_bigquery_clients["mock_sync_write_client"]
    assert mock_write_client.append_rows.call_count == 1


@pytest.mark.asyncio
async def test_async_on_chain_start(
    handler: AsyncBigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that on_chain_start logs the correct event."""
    await handler.on_chain_start(
        serialized={"name": "test_chain"}, inputs={"input": "test"}, run_id=uuid4()
    )
    mock_write_client = mock_bigquery_clients["mock_async_write_client"]
    assert mock_write_client.append_rows.call_count == 1


def test_sync_on_chain_start(
    sync_handler: BigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that sync on_chain_start logs the correct event."""
    sync_handler.on_chain_start(
        serialized={"name": "test_chain"}, inputs={"input": "test"}, run_id=uuid4()
    )
    mock_write_client = mock_bigquery_clients["mock_sync_write_client"]
    assert mock_write_client.append_rows.call_count == 1


@pytest.mark.asyncio
async def test_async_on_chain_end(
    handler: AsyncBigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that on_chain_end logs the correct event."""
    await handler.on_chain_end(outputs={"output": "test"}, run_id=uuid4())
    mock_write_client = mock_bigquery_clients["mock_async_write_client"]
    assert mock_write_client.append_rows.call_count == 1


def test_sync_on_chain_end(
    sync_handler: BigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that sync on_chain_end logs the correct event."""
    sync_handler.on_chain_end(outputs={"output": "test"}, run_id=uuid4())
    mock_write_client = mock_bigquery_clients["mock_sync_write_client"]
    assert mock_write_client.append_rows.call_count == 1


@pytest.mark.asyncio
async def test_async_on_tool_start(
    handler: AsyncBigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that on_tool_start logs the correct event."""
    await handler.on_tool_start(
        serialized={"name": "test_tool"}, input_str="test", run_id=uuid4()
    )
    mock_write_client = mock_bigquery_clients["mock_async_write_client"]
    assert mock_write_client.append_rows.call_count == 1


def test_sync_on_tool_start(
    sync_handler: BigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that sync on_tool_start logs the correct event."""
    sync_handler.on_tool_start(
        serialized={"name": "test_tool"}, input_str="test", run_id=uuid4()
    )
    mock_write_client = mock_bigquery_clients["mock_sync_write_client"]
    assert mock_write_client.append_rows.call_count == 1


@pytest.mark.asyncio
async def test_async_on_agent_action(
    handler: AsyncBigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that on_agent_action logs the correct event."""
    action = AgentAction(tool="test_tool", tool_input="test", log="test log")
    await handler.on_agent_action(action, run_id=uuid4())
    mock_write_client = mock_bigquery_clients["mock_async_write_client"]
    assert mock_write_client.append_rows.call_count == 1

def test_sync_on_agent_action(
    sync_handler: BigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that sync on_agent_action logs the correct event."""
    action = AgentAction(tool="test_tool", tool_input="test", log="test log")
    sync_handler.on_agent_action(action, run_id=uuid4())
    mock_write_client = mock_bigquery_clients["mock_sync_write_client"]
    assert mock_write_client.append_rows.call_count == 1


@pytest.mark.asyncio
async def test_async_on_agent_finish(
    handler: AsyncBigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that on_agent_finish logs the correct event."""
    finish = AgentFinish(return_values={"output": "test"}, log="test log")
    await handler.on_agent_finish(finish, run_id=uuid4())
    mock_write_client = mock_bigquery_clients["mock_async_write_client"]
    assert mock_write_client.append_rows.call_count == 1

def test_sync_on_agent_finish(
    sync_handler: BigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that sync on_agent_finish logs the correct event."""
    finish = AgentFinish(return_values={"output": "test"}, log="test log")
    sync_handler.on_agent_finish(finish, run_id=uuid4())
    mock_write_client = mock_bigquery_clients["mock_sync_write_client"]
    assert mock_write_client.append_rows.call_count == 1


@pytest.mark.asyncio
async def test_async_on_llm_error(
    handler: AsyncBigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that on_llm_error logs the correct event."""
    await handler.on_llm_error(Exception("test error"), run_id=uuid4())
    mock_write_client = mock_bigquery_clients["mock_async_write_client"]
    assert mock_write_client.append_rows.call_count == 1


def test_sync_on_llm_error(
    sync_handler: BigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that sync on_llm_error logs the correct event."""
    sync_handler.on_llm_error(Exception("test error"), run_id=uuid4())
    mock_write_client = mock_bigquery_clients["mock_sync_write_client"]
    assert mock_write_client.append_rows.call_count == 1


@pytest.mark.asyncio
async def test_async_on_chat_model_start(
    handler: AsyncBigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that on_chat_model_start logs the correct event."""
    await handler.on_chat_model_start(
        serialized={"name": "test_chat_model"},
        messages=[[HumanMessage(content="test")]],
        run_id=uuid4(),
    )
    mock_write_client = mock_bigquery_clients["mock_async_write_client"]
    assert mock_write_client.append_rows.call_count == 1


def test_sync_on_chat_model_start(
    sync_handler: BigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that sync on_chat_model_start logs the correct event."""
    sync_handler.on_chat_model_start(
        serialized={"name": "test_chat_model"},
        messages=[[HumanMessage(content="test")]],
        run_id=uuid4(),
    )
    mock_write_client = mock_bigquery_clients["mock_sync_write_client"]
    assert mock_write_client.append_rows.call_count == 1


@pytest.mark.asyncio
async def test_async_on_retriever_end(
    handler: AsyncBigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that on_retriever_end logs the correct event."""
    documents = [Document(page_content="test document")]
    await handler.on_retriever_end(documents, run_id=uuid4())
    mock_write_client = mock_bigquery_clients["mock_async_write_client"]
    assert mock_write_client.append_rows.call_count == 1


def test_sync_on_retriever_end(
    sync_handler: BigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that sync on_retriever_end logs the correct event."""
    documents = [Document(page_content="test document")]
    sync_handler.on_retriever_end(documents, run_id=uuid4())
    mock_write_client = mock_bigquery_clients["mock_sync_write_client"]
    assert mock_write_client.append_rows.call_count == 1


def test_sync_ensure_init_creates_dataset_and_table(
    mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that sync _ensure_init creates dataset and table if they don't exist."""
    handler = BigQueryCallbackHandler(
        project_id="test-project",
        dataset_id="test_dataset",
        table_id="test_table",
    )
    initialized = handler._ensure_init()

    assert initialized is True
    mock_bq_client = mock_bigquery_clients["mock_bq_client"]
    mock_bq_client.create_dataset.assert_called_once_with("test_dataset", exists_ok=True)
    mock_bq_client.create_table.assert_called_once()


def test_sync_init_failure(mock_bigquery_clients: Dict[str, Any]) -> None:
    """Test that sync initialization failure is handled gracefully."""
    mock_bigquery_clients["mock_google_auth"].default.side_effect = Exception(
        "Auth failed"
    )
    handler = BigQueryCallbackHandler(project_id="test-project", dataset_id="test_dataset")
    initialized = handler._ensure_init()
    assert not initialized


@pytest.mark.asyncio
async def test_ensure_init_creates_dataset_and_table(
    mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that _ensure_init creates dataset and table if they don't exist."""
    handler = AsyncBigQueryCallbackHandler(
        project_id="test-project",
        dataset_id="test_dataset",
        table_id="test_table",
    )
    await handler._ensure_init()

    mock_bq_client = mock_bigquery_clients["mock_bq_client"]
    mock_bq_client.create_dataset.assert_called_once_with("test_dataset", exists_ok=True)
    mock_bq_client.create_table.assert_called_once()


@pytest.mark.asyncio
async def test_init_failure(mock_bigquery_clients: Dict[str, Any]) -> None:
    """Test that initialization failure is handled gracefully."""
    mock_bigquery_clients["mock_google_auth"].default.side_effect = Exception(
        "Auth failed"
    )
    handler = AsyncBigQueryCallbackHandler(
        project_id="test-project",
        dataset_id="test_dataset",
        table_id="test_table",
    )
    initialized = await handler._ensure_init()
    assert not initialized

    # Verify that no write is attempted if init failed
    await handler.on_llm_start(
        serialized={"name": "test_llm"}, prompts=["test"], run_id=uuid4()
    )
    mock_write_client = mock_bigquery_clients["mock_async_write_client"]
    mock_write_client.append_rows.assert_not_called()


@pytest.mark.asyncio
async def test_async_close(
    handler: AsyncBigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that the close method closes clients."""
    await handler.close()
    mock_async_write_client = mock_bigquery_clients["mock_async_write_client"]
    mock_async_write_client.close.assert_called_once()
    mock_bq_client = mock_bigquery_clients["mock_bq_client"]
    mock_bq_client.close.assert_called_once()


@pytest.mark.asyncio
async def test_async_on_llm_new_token(
    handler: AsyncBigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that on_llm_new_token logs the correct event."""
    await handler.on_llm_new_token(token="new", run_id=uuid4())
    mock_write_client = mock_bigquery_clients["mock_async_write_client"]
    assert mock_write_client.append_rows.call_count == 1


def test_sync_on_llm_new_token(
    sync_handler: BigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that sync on_llm_new_token logs the correct event."""
    sync_handler.on_llm_new_token(token="new", run_id=uuid4())
    mock_write_client = mock_bigquery_clients["mock_sync_write_client"]
    assert mock_write_client.append_rows.call_count == 1


@pytest.mark.asyncio
async def test_async_on_tool_end(
    handler: AsyncBigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that on_tool_end logs the correct event."""
    await handler.on_tool_end(output="test output", run_id=uuid4())
    mock_write_client = mock_bigquery_clients["mock_async_write_client"]
    assert mock_write_client.append_rows.call_count == 1


def test_sync_on_tool_end(
    sync_handler: BigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that sync on_tool_end logs the correct event."""
    sync_handler.on_tool_end(output="test output", run_id=uuid4())
    mock_write_client = mock_bigquery_clients["mock_sync_write_client"]
    assert mock_write_client.append_rows.call_count == 1


@pytest.mark.asyncio
async def test_async_on_tool_error(
    handler: AsyncBigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that on_tool_error logs the correct event."""
    await handler.on_tool_error(Exception("tool error"), run_id=uuid4())
    mock_write_client = mock_bigquery_clients["mock_async_write_client"]
    assert mock_write_client.append_rows.call_count == 1


def test_sync_on_tool_error(
    sync_handler: BigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that sync on_tool_error logs the correct event."""
    sync_handler.on_tool_error(Exception("tool error"), run_id=uuid4())
    mock_write_client = mock_bigquery_clients["mock_sync_write_client"]
    assert mock_write_client.append_rows.call_count == 1


@pytest.mark.asyncio
async def test_async_on_chain_error(
    handler: AsyncBigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that on_chain_error logs the correct event."""
    await handler.on_chain_error(Exception("chain error"), run_id=uuid4())
    mock_write_client = mock_bigquery_clients["mock_async_write_client"]
    assert mock_write_client.append_rows.call_count == 1


def test_sync_on_chain_error(
    sync_handler: BigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that sync on_chain_error logs the correct event."""
    sync_handler.on_chain_error(Exception("chain error"), run_id=uuid4())
    mock_write_client = mock_bigquery_clients["mock_sync_write_client"]
    assert mock_write_client.append_rows.call_count == 1


@pytest.mark.asyncio
async def test_async_on_retriever_start(
    handler: AsyncBigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that on_retriever_start logs the correct event."""
    await handler.on_retriever_start(
        serialized={"name": "test_retriever"}, query="test query", run_id=uuid4()
    )
    mock_write_client = mock_bigquery_clients["mock_async_write_client"]
    assert mock_write_client.append_rows.call_count == 1


def test_sync_on_retriever_start(
    sync_handler: BigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that sync on_retriever_start logs the correct event."""
    sync_handler.on_retriever_start(
        serialized={"name": "test_retriever"}, query="test query", run_id=uuid4()
    )
    mock_write_client = mock_bigquery_clients["mock_sync_write_client"]
    assert mock_write_client.append_rows.call_count == 1


@pytest.mark.asyncio
async def test_async_on_retriever_error(
    handler: AsyncBigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that on_retriever_error logs the correct event."""
    await handler.on_retriever_error(Exception("retriever error"), run_id=uuid4())
    mock_write_client = mock_bigquery_clients["mock_async_write_client"]
    assert mock_write_client.append_rows.call_count == 1


def test_sync_on_retriever_error(
    sync_handler: BigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that sync on_retriever_error logs the correct event."""
    sync_handler.on_retriever_error(Exception("retriever error"), run_id=uuid4())
    mock_write_client = mock_bigquery_clients["mock_sync_write_client"]
    assert mock_write_client.append_rows.call_count == 1


@pytest.mark.asyncio
async def test_async_on_text(
    handler: AsyncBigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that on_text logs the correct event."""
    await handler.on_text("some text", run_id=uuid4())
    mock_write_client = mock_bigquery_clients["mock_async_write_client"]
    assert mock_write_client.append_rows.call_count == 1


def test_sync_on_text(
    sync_handler: BigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that sync on_text logs the correct event."""
    sync_handler.on_text("some text", run_id=uuid4())
    mock_write_client = mock_bigquery_clients["mock_sync_write_client"]
    assert mock_write_client.append_rows.call_count == 1


def test_sync_close(
    sync_handler: BigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that the sync close method closes clients."""
    sync_handler.close()
    mock_sync_write_client = mock_bigquery_clients["mock_sync_write_client"]
    mock_sync_write_client.close.assert_called_once()
    mock_bq_client = mock_bigquery_clients["mock_bq_client"]
    mock_bq_client.close.assert_called_once()

