import redis
from dotenv import load_dotenv
load_dotenv()
import os

r = redis.Redis(host=os.environ['REDIS_HOSTNAME'], port=int(os.environ['REDIS_PORT']), db=0, decode_responses=True)