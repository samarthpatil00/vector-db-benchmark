import pgvector.psycopg
import psycopg

from benchmark.dataset import Dataset
from engine.base_client import IncompatibilityError
from engine.base_client.configure import BaseConfigurator
from engine.base_client.distances import Distance
from engine.clients.pgvectorscale.config import get_db_config


class PgVectorScaleConfigurator(BaseConfigurator):
    def __init__(self, host, collection_params: dict, connection_params: dict):
        super().__init__(host, collection_params, connection_params)
        self.conn = psycopg.connect(**get_db_config(host, connection_params))
        print("Connected to the database")
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;")
        pgvector.psycopg.register_vector(self.conn)

    def clean(self):
        pass
        # self.conn.execute(
        #     "DROP TABLE IF EXISTS items_pgvectorscale CASCADE;",
        # )

    def recreate(self, dataset: Dataset, collection_params):
        if dataset.config.distance == Distance.DOT:
            raise IncompatibilityError

        self.conn.execute(
            f"""CREATE TABLE IF NOT EXISTS items_pgvectorscale (
                id SERIAL PRIMARY KEY,
                embedding vector({dataset.config.vector_size}) NOT NULL
            );"""
        )
        self.conn.close()

    def delete_client(self):
        self.conn.close()
