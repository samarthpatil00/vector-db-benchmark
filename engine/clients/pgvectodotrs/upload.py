import ast
import json
from typing import List

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from dataset_reader.base_reader import Record
from engine.base_client import IncompatibilityError
from engine.base_client.distances import Distance
from engine.base_client.upload import BaseUploader
from engine.clients.pgvectodotrs.config import get_db_config


class PgVectoDotRSUploader(BaseUploader):
    DISTANCE_MAPPING = {
        Distance.L2: "vector_l2_ops",
        Distance.COSINE: "vector_cos_ops",
    }
    conn = None
    cur = None
    upload_params = {}

    @classmethod
    def init_client(cls, host, distance, connection_params, upload_params):
        cls.conn = psycopg.connect(**get_db_config(host, connection_params))
        register_vector(cls.conn)
        cls.cur = cls.conn.cursor()
        cls.upload_params = upload_params

    @classmethod
    def upload_batch(cls, batch: List[Record]):
        ids, vectors = [], []
        for record in batch:
            ids.append(record.id)
            vectors.append(record.vector)

        # Copy with binary format is not yet supported with pgvecto.rs
        # So normal insert query is used
        for i, x in zip(ids, vectors):
            insert_query = "INSERT INTO items_pgvectodotrs (id, embedding) VALUES (%s, %s)"
            embedding_str = ','.join(map(str, x))
            cls.cur.execute(insert_query, (i, f'[{embedding_str}]',))

    @classmethod
    def post_upload(cls, distance):
        try:
            hnsw_distance_type = cls.DISTANCE_MAPPING[distance]
        except KeyError:
            raise IncompatibilityError(f"Unsupported distance metric: {distance}")

        print(f"pgvectodotrs building hnsw index")
        cls.conn.execute(
            f"CREATE INDEX ON items_pgvectodotrs USING vectors (embedding {hnsw_distance_type})"
        )
        print(f"pgvectodotrs Done with hnsw index")

        return {}

    @classmethod
    def delete_client(cls):
        if cls.cur:
            cls.cur.close()
            cls.conn.close()
