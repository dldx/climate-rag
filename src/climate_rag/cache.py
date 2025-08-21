import os

import redis
from dotenv import load_dotenv
from redis import ResponseError
from redis.commands.search.field import NumericField, TagField, TextField

try:
    from redis.commands.search.indexDefinition import IndexDefinition, IndexType
except ImportError:
    from redis.commands.search.index_definition import IndexDefinition, IndexType

load_dotenv()

r = redis.Redis(
    host=os.environ["REDIS_HOSTNAME"],
    port=int(os.environ["REDIS_PORT"]),
    db=0,
    decode_responses=True,
)


# Function to check if the index exists
def index_exists(r, index_name):
    try:
        r.ft(index_name).info()
        return True
    except ResponseError as e:
        if "unknown index name" in str(e).lower():
            return False
        raise e


## Create the sources index

# Define the index schema
source_schema = (
    TextField("source", sortable=True),
    TextField("key_entity", sortable=True),
    TextField("company_name", sortable=True),
    TextField("title", sortable=True),
    TextField("page_content"),
    NumericField("page_length", sortable=True),
    TagField("type_of_document", sortable=True),
    NumericField("date_added", sortable=True),
    NumericField("publishing_date", sortable=True),
    TagField("fetched_additional_metadata", sortable=True),
    TextField("key_entities"),
    TextField("raw_html"),
    TagField("project_id", sortable=True),
)

# Define the index name
source_index_name = "idx:source"

# Check if the index exists, and create it if it doesn't
if not index_exists(r, source_index_name):
    try:
        # Create the index
        r.ft(source_index_name).create_index(
            source_schema,
            definition=IndexDefinition(
                prefix=["climate-rag::source:"], index_type=IndexType.HASH
            ),
        )
        print(f"Index '{source_index_name}' created successfully.")
    except ResponseError as e:
        print(f"Error creating index: {e}")

## Create a chinese source index
# Define the index name
zh_source_index_name = "idx:source_zh"

# Check if the index exists, and create it if it doesn't
if not index_exists(r, zh_source_index_name):
    try:
        # Create the index
        r.ft(zh_source_index_name).create_index(
            source_schema,
            definition=IndexDefinition(
                prefix=["climate-rag::source:"],
                index_type=IndexType.HASH,
                language="chinese",
            ),
        )
        print(f"Index '{zh_source_index_name}' created successfully.")
    except ResponseError as e:
        print(f"Error creating index: {e}")

## Create a japanese source index
# Define the index name
ja_source_index_name = "idx:source_zh"

# Check if the index exists, and create it if it doesn't
if not index_exists(r, ja_source_index_name):
    try:
        # Create the index
        r.ft(ja_source_index_name).create_index(
            source_schema,
            definition=IndexDefinition(
                prefix=["climate-rag::source:"],
                index_type=IndexType.HASH,
                language="japanese",
            ),
        )
        print(f"Index '{ja_source_index_name}' created successfully.")
    except ResponseError as e:
        print(f"Error creating index: {e}")


## Create the answer index

# Define the index name
answer_index_name = "idx:answer"

# Define the index schema
answer_schema = (
    TextField("question"),
    TextField("answer"),
    NumericField("date_added", sortable=True),
    TextField("sources"),
    TagField("language", sortable=True),
    TagField("llm", sortable=True),
    NumericField("num_sources", sortable=True),
    NumericField("answer_length", sortable=True),
    TagField("project_id", sortable=True),
)

# Check if the index exists, and create it if it doesn't
if not index_exists(r, answer_index_name):
    try:
        # Create the index
        r.ft(answer_index_name).create_index(
            answer_schema,
            definition=IndexDefinition(
                prefix=["climate-rag::answer:"], index_type=IndexType.HASH
            ),
        )
        print(f"Index '{answer_index_name}' created successfully.")
    except ResponseError as e:
        print(f"Error creating index: {e}")
