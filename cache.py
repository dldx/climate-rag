import redis
from dotenv import load_dotenv
load_dotenv()
import os

r = redis.Redis(host=os.environ['REDIS_HOSTNAME'], port=int(os.environ['REDIS_PORT']), db=0, decode_responses=True)

from redis import ResponseError
from redis.commands.search.field import TextField, NumericField, TagField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType

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
    TextField("raw_html")
)

# Define the index name
index_name = "idx:source"

# Check if the index exists, and create it if it doesn't
if not index_exists(r, index_name):
    try:
        # Create the index
        r.ft(index_name).create_index(
            source_schema,
            definition=IndexDefinition(prefix=["climate-rag::source:"], index_type=IndexType.HASH)
        )
        print(f"Index '{index_name}' created successfully.")
    except ResponseError as e:
        print(f"Error creating index: {e}")

## Create the answer index

# Define the index name
index_name = "idx:answer"

# Define the index schema
schema = (
    TextField("question"),
    TextField("answer"),
    NumericField("date_added", sortable=True),
    TextField("sources")
)

# Check if the index exists, and create it if it doesn't
if not index_exists(r, index_name):
    try:
        # Create the index
        r.ft(index_name).create_index(
            answer_schema,
            definition=IndexDefinition(prefix=["climate-rag::answer:"], index_type=IndexType.HASH)
        )
        print(f"Index '{index_name}' created successfully.")
    except ResponseError as e:
        print(f"Error creating index: {e}")