import re
from typing import List, Dict, Callable, Optional
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter

class BaseTablePreservingTextSplitter:
    """
    Base class for table-preserving text splitters with common table extraction logic.
    """
    @staticmethod
    def extract_tables(text: str) -> List[dict]:
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
            tables.append({
                "type": "markdown",
                "content": match.group(0),
                "start": match.start(),
                "end": match.end(),
            })

        # Find HTML tables
        for match in re.finditer(html_table_pattern, text, re.DOTALL):
            tables.append({
                "type": "html",
                "content": match.group(0),
                "start": match.start(),
                "end": match.end(),
            })

        # Find pipe-separated tables
        for match in re.finditer(pipe_table_pattern, text, re.MULTILINE):
            tables.append({
                "type": "pipe",
                "content": match.group(0),
                "start": match.start(),
                "end": match.end(),
            })

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

    @classmethod
    def split_text(cls,
                   text: str,
                   base_splitter: Callable[[str], List[str]],
                   chunk_size: Optional[int] = None,
                   length_function: Callable[[str], int] = len,
                   table_augmenter: Optional[Callable[[str], str]] = None
                   ) -> List[str]:
        """
        Split text while preserving tables.

        :param text: Input text to split
        :param base_splitter: The base text splitter function to use
        :param chunk_size: Optional chunk size to limit chunk length
        :param length_function: Function to calculate length of text segments
        :return: List of text chunks
        """
        # First, identify tables in the text
        tables = cls.extract_tables(text)

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
                # Split the text segment
                text_chunks = base_splitter(segment["content"])

                for text_chunk in text_chunks:
                    # Determine if chunk can be added based on chunk_size
                    can_add_chunk = (
                        chunk_size is None or
                        length_function(current_chunk) + length_function(text_chunk) + 1 <= chunk_size
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
                # Determine if table can be added based on chunk_size
                can_add_table = (
                    chunk_size is None or
                    length_function(current_chunk) + length_function(segment["content"]) + 1 <= chunk_size
                )

                # If table fits in current chunk, add it
                if can_add_table:
                    current_chunk += ("\n\n" if current_chunk else "") + segment["content"]
                else:
                    # If current chunk exists, finalize it
                    if current_chunk:
                        final_chunks.append(current_chunk)
                    # Start a new chunk with the table
                    current_chunk = segment["content"]

        # Add the last chunk if it's not empty
        if current_chunk:
            final_chunks.append(current_chunk)

        return final_chunks


class TablePreservingSemanticChunker(SemanticChunker):
    def __init__(self, chunk_size, length_function=len, table_augmenter=None, **kwargs):
        self._chunk_size = chunk_size
        self._length_function = length_function
        self._table_augmenter = table_augmenter
        super().__init__(**kwargs)

    def split_text(self, text: str) -> List[str]:
        return BaseTablePreservingTextSplitter.split_text(
            text,
            base_splitter=super().split_text,
            chunk_size=self._chunk_size,
            length_function=self._length_function,
            table_augmenter=self._table_augmenter
        )


class TablePreservingTextSplitter(RecursiveCharacterTextSplitter):
    def __init__(self, chunk_size, length_function=len, table_augmenter=None, **kwargs):
        self._chunk_size = chunk_size
        self._length_function = length_function
        self._table_augmenter = table_augmenter
        super().__init__(**kwargs)

    def split_text(self, text: str) -> List[str]:
        return BaseTablePreservingTextSplitter.split_text(
            text,
            base_splitter=super().split_text,
            chunk_size=self._chunk_size,
            length_function=self._length_function,
            table_augmenter=self._table_augmenter
        )