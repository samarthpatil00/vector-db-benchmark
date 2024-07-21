from typing import List

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from dataset_reader.base_reader import Record
from engine.base_client.upload import BaseUploader
from engine.clients.pgvectorscale.config import get_db_config


class PgVectorScaleUploader(BaseUploader):
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

        vectors = np.array(vectors)
        # Copy is faster than insert
        with cls.cur.copy(
            "COPY items_pgvectorscale (id, embedding) FROM STDIN WITH (FORMAT BINARY)"
        ) as copy:
            copy.set_types(["integer", "vector"])
            for i, embedding in zip(ids, vectors):
                copy.write_row((i, embedding))

    @classmethod
    def post_upload(cls, distance):
        print(f"pgvectorscale building disk ann index")
        cls.conn.execute(
            f"CREATE INDEX ON items_pgvectorscale USING diskann(embedding);"
        )
        print(f"pgvectorscale done with indexing")

        return {}
    
    @classmethod
    def delete_client(cls):
        if cls.cur:
            cls.cur.close()
            cls.conn.close()
    
    @classmethod
    def insert_records(cls, batch: List[Record]):
        vectors = []
        ids = []
        for record in batch:
            ids.append(record.id)
            vectors.append(record.vector)

        vectors = np.array(vectors)

        for i, embedding in zip(ids, vectors):
            # insert query for the record
            insert_query = "INSERT INTO items_pgvectorscale (id, embedding) VALUES (%s, %s)"
            embedding_str = ','.join(map(str, embedding))
            cls.cur.execute(insert_query, (i, f'[{embedding_str}]',))
            cls.conn.commit()
