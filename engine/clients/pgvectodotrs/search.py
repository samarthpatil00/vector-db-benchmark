from typing import List, Tuple

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from dataset_reader.base_reader import Query
from engine.base_client.distances import Distance
from engine.base_client.search import BaseSearcher
from engine.clients.pgvectodotrs.config import get_db_config


class PgVectoDotRSSearcher(BaseSearcher):
    conn = None
    cur = None
    distance = None
    search_params = {}

    @classmethod
    def init_client(cls, host, distance, connection_params: dict, search_params: dict):
        cls.conn = psycopg.connect(**get_db_config(host, connection_params))
        register_vector(cls.conn)
        cls.cur = cls.conn.cursor()
        cls.cur.execute(f"SET hnsw.ef_search = {search_params['config']['hnsw_ef']}")
        if distance == Distance.COSINE:
            cls.query = "SELECT id, embedding <=> %s AS _score FROM items_pgvectodotrs ORDER BY _score LIMIT %s"
        elif distance == Distance.L2:
            cls.query = "SELECT id, embedding <-> %s AS _score FROM items_pgvectodotrs ORDER BY _score LIMIT %s"
        else:
            raise NotImplementedError(f"Unsupported distance metric {cls.distance}")

    @classmethod
    def search_one(cls, query: Query, top) -> List[Tuple[int, float]]:
        # TODO: Use query.metaconditions for datasets with filtering
        cls.cur.execute(
           f"SELECT id, embedding <=> '{query.vector}' AS _score FROM items_pgvectodotrs ORDER BY _score LIMIT {top}"
        )
        return cls.cur.fetchall()

    @classmethod
    def delete_client(cls):
        if cls.cur:
            cls.cur.close()
            cls.conn.close()
