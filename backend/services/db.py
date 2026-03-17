from collections import deque

try:
    from pymongo import DESCENDING, MongoClient
except Exception:
    MongoClient = None
    DESCENDING = -1


class ConversationStore:
    def __init__(self, mongo_uri, database_name):
        self._memory = deque(maxlen=32)
        self._collection = None

        if mongo_uri and MongoClient is not None:
            client = MongoClient(mongo_uri)
            database = client[database_name]
            self._collection = database["conversations"]

    def insert(self, item):
        self._memory.appendleft(item)
        if self._collection is not None:
            self._collection.insert_one(item)

    def fetch_recent(self, limit=8):
        if self._collection is not None:
            cursor = self._collection.find().sort("created_at", DESCENDING).limit(limit)
            items = []
            for document in cursor:
                document["_id"] = str(document["_id"])
                items.append(document)
            return items

        return list(self._memory)[:limit]
