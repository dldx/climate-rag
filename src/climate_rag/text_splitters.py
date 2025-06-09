import logging
import re
from typing import Any, Callable, Dict, List, Optional

import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

logger = logging.getLogger(__name__)


class BaseTablePreservingTextSplitter:
    """
    Base class for table-preserving text splitters with common table extraction logic.
    """

    @staticmethod
    def extract_tables(text: str) -> List[Dict[str, Any]]:
        """
        Extract tables from the text.
        Supports different table formats (markdown, HTML, pipe-separated).
        """
        # Markdown table detection
        markdown_table_pattern = r"(\|[^\n]+\|\n)+((?:\|[-:| ]+\|\n)(\|[^\n]+\|\n)*)"

        # HTML table detection
        html_table_pattern = r"<table>.*?</table>"

        # Pipe-separated table detection
        pipe_table_pattern = r"(^.*\|.*\n)(^[-:| ]+\|\n)(^.*\|.*\n)*"

        tables = []

        # Find markdown tables
        for match in re.finditer(markdown_table_pattern, text, re.MULTILINE):
            tables.append(
                {
                    "type": "markdown",
                    "content": match.group(0),
                    "start": match.start(),
                    "end": match.end(),
                }
            )

        # Find HTML tables
        for match in re.finditer(html_table_pattern, text, re.DOTALL):
            tables.append(
                {
                    "type": "html",
                    "content": match.group(0),
                    "start": match.start(),
                    "end": match.end(),
                }
            )

        # Find pipe-separated tables
        for match in re.finditer(pipe_table_pattern, text, re.MULTILINE):
            tables.append(
                {
                    "type": "pipe",
                    "content": match.group(0),
                    "start": match.start(),
                    "end": match.end(),
                }
            )

        # Sort and deduplicate tables
        tables = sorted(tables, key=lambda x: x["start"])
        deduplicated_tables = []
        for table in tables:
            if not any(
                table["start"] == t["start"] and table["end"] == t["end"]
                for t in deduplicated_tables
            ):
                deduplicated_tables.append(table)

        return deduplicated_tables

    @staticmethod
    def _count_tokens(text: str) -> int:
        """
        Count tokens using tiktoken for accurate token measurement.
        """
        try:
            encoding = tiktoken.encoding_for_model("gpt-4o")
            return len(encoding.encode(text))
        except Exception as e:
            logger.warning(
                f"Failed to count tokens with tiktoken: {e}. Falling back to character-based estimation."
            )
            # Fallback to rough estimation if tiktoken fails
            return len(text) // 4

    @staticmethod
    def _split_large_text_segment(text: str, max_tokens: int = 25_000) -> List[str]:
        """
        Split a large text segment into smaller chunks to avoid token limits.
        Uses tiktoken for accurate token counting and character-based splitting.
        """
        if BaseTablePreservingTextSplitter._count_tokens(text) <= max_tokens:
            return [text]

        # Use a simple recursive character splitter for preprocessing
        # Estimate character count based on token count to stay well under limit
        chunk_size = max_tokens * 3  # Conservative estimate: ~3 chars per token
        temp_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        chunks = temp_splitter.split_text(text)

        # Validate that chunks are within token limits and further split if needed
        final_chunks = []
        for chunk in chunks:
            token_count = BaseTablePreservingTextSplitter._count_tokens(chunk)
            if token_count <= max_tokens:
                final_chunks.append(chunk)
            else:
                # If still too large, split more aggressively
                logger.warning(
                    f"Chunk still has {token_count} tokens, splitting further"
                )
                smaller_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size // 2,
                    chunk_overlap=100,
                    separators=["\n\n", "\n", ". ", " ", ""],
                )
                sub_chunks = smaller_splitter.split_text(chunk)
                final_chunks.extend(sub_chunks)

        logger.info(
            f"Split large text segment ({BaseTablePreservingTextSplitter._count_tokens(text)} tokens) into {len(final_chunks)} preprocessing chunks"
        )
        return final_chunks

    @classmethod
    def split_text(
        cls,
        text: str,
        base_splitter: Callable[[str], List[str]],
        chunk_size: Optional[int] = None,
        length_function: Callable[[str], int] = len,
        table_augmenter: Optional[Callable[[str], str]] = None,
        max_chunk_tokens: int = 8191,
    ) -> List[str]:
        """
        Split text while preserving tables.

        :param text: Input text to split
        :param base_splitter: The base text splitter function to use
        :param chunk_size: Optional chunk size to limit chunk length
        :param length_function: Function to calculate length of text segments
        :param table_augmenter: Optional function to augment table content
        :param max_chunk_tokens: Maximum number of tokens per chunk. Raises ValueError if exceeded.
        :return: List of text chunks
        """
        # First, identify tables in the text
        tables = cls.extract_tables(text)
        if len(tables) > 0:
            logger.info(f"{len(tables)} tables to augment with additional context")

        # Create a list of text segments that alternate between non-table text and tables
        text_segments = []
        last_end = 0
        for table in tables:
            # Add text before the table
            if table["start"] > last_end:
                text_segments.append(
                    {"type": "text", "content": text[last_end : table["start"]]}
                )

            # Add the table
            text_segments.append({"type": "table", "content": table["content"]})

            last_end = table["end"]

        # Add remaining text after the last table
        if last_end < len(text):
            text_segments.append({"type": "text", "content": text[last_end:]})

        # Split text segments
        final_chunks = []
        current_chunk = ""

        for segment in text_segments:
            if segment["type"] == "text":
                # Check if the text segment is too large for the base splitter
                # If so, split it into smaller pieces first
                text_pieces = cls._split_large_text_segment(segment["content"])

                for text_piece in text_pieces:
                    try:
                        # Split the text piece
                        text_chunks = base_splitter(text_piece)
                    except Exception as e:
                        logger.error(f"Error in base splitter: {e}")
                        # Fallback to simple character-based splitting
                        fallback_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=chunk_size or 4000, chunk_overlap=200
                        )
                        text_chunks = fallback_splitter.split_text(text_piece)
                        logger.info(
                            f"Used fallback splitter, created {len(text_chunks)} chunks"
                        )

                    for text_chunk in text_chunks:
                        # Determine if chunk can be added based on chunk_size
                        can_add_chunk = (
                            chunk_size is None
                            or length_function(current_chunk)
                            + length_function(text_chunk)
                            + 1
                            <= chunk_size
                        )

                        # Try to add the text chunk to the current chunk
                        if can_add_chunk:
                            current_chunk += (" " if current_chunk else "") + text_chunk
                        else:
                            # If adding would exceed chunk size, finalize current chunk and start a new one
                            if current_chunk:
                                final_chunks.append(current_chunk)
                            current_chunk = text_chunk

            elif segment["type"] == "table":
                # Handle table integration
                # If a table augmenter function is provided, use it to augment the table content
                # This can be used to add additional context to the table to make retrieval more accurate
                if table_augmenter:
                    segment["content"] = table_augmenter(segment["content"])
                    logger.debug(f"""Augmenting table:\n{segment["content"]}""")
                # Determine if table can be added based on chunk_size
                can_add_table = (
                    chunk_size is None
                    or length_function(current_chunk)
                    + length_function(segment["content"])
                    + 1
                    <= chunk_size
                )

                # If table fits in current chunk, add it
                if can_add_table:
                    current_chunk += ("\n\n" if current_chunk else "") + segment[
                        "content"
                    ]
                else:
                    # If current chunk exists, finalize it
                    if current_chunk:
                        final_chunks.append(current_chunk)
                    # Start a new chunk with the table
                    current_chunk = segment["content"]

        # Add the last chunk if it's not empty
        if current_chunk:
            final_chunks.append(current_chunk)

        # Validate that no chunk exceeds the maximum token limit
        for i, chunk in enumerate(final_chunks):
            token_count = cls._count_tokens(chunk)
            if token_count > max_chunk_tokens:
                logger.error(
                    f"Chunk {i+1}/{len(final_chunks)} is too large: {token_count} tokens (max: {max_chunk_tokens}). "
                    f"This is often caused by a large table that cannot be split."
                )
                raise ValueError(
                    f"A chunk is too large ({token_count} tokens > max {max_chunk_tokens}). "
                    f"Content starts with: '{chunk[:500]}...'"
                )

        return final_chunks


class TablePreservingSemanticChunker(SemanticChunker):
    def __init__(
        self,
        chunk_size,
        length_function=len,
        table_augmenter=None,
        max_chunk_tokens: int = 25_000,
        **kwargs,
    ):
        self._chunk_size = chunk_size
        self._length_function = length_function
        self._table_augmenter = table_augmenter
        self._max_chunk_tokens = max_chunk_tokens
        super().__init__(**kwargs)

    def split_text(self, text: str) -> List[str]:
        return BaseTablePreservingTextSplitter.split_text(
            text,
            base_splitter=super().split_text,
            chunk_size=self._chunk_size,
            length_function=self._length_function,
            table_augmenter=self._table_augmenter,
            max_chunk_tokens=self._max_chunk_tokens,
        )


class TablePreservingTextSplitter(RecursiveCharacterTextSplitter):
    def __init__(
        self,
        chunk_size,
        length_function=len,
        table_augmenter=None,
        max_chunk_tokens: int = 25_000,
        **kwargs,
    ):
        self._chunk_size = chunk_size
        self._length_function = length_function
        self._table_augmenter = table_augmenter
        self._max_chunk_tokens = max_chunk_tokens
        super().__init__(**kwargs)

    def split_text(self, text: str) -> List[str]:
        return BaseTablePreservingTextSplitter.split_text(
            text,
            base_splitter=super().split_text,
            chunk_size=self._chunk_size,
            length_function=self._length_function,
            table_augmenter=self._table_augmenter,
            max_chunk_tokens=self._max_chunk_tokens,
        )
