from sqlalchemy import create_engine, text
import logging
import pandas as pd
import psycopg


class Connection():
    def __init__(self, host: str, port: int, user: str, password: str, db: str):
        """
        Initialize the Connection object.
        """
        self.connection_string = f'postgresql://{user}:{password}@{host}:{port}/{db}'
        self.engine = None


    def establish_connection(self):
        self.engine = create_engine(self.connection_string)
        return self.engine

    def table_to_dataframe(self, table_name: str, columns: list = None) -> pd.DataFrame:
        """
        Convert a table from the database to a pandas DataFrame.

        :param table_name: The name of the table to convert
        :return: A pandas DataFrame containing the table data
        """
        if not self.engine:
            raise ValueError("No connection established. Call establish_connection() first.")
        
        with self.engine.connect() as connection:
            df = pd.read_sql_table(table_name, con=connection, columns=columns)
        
        return df
    
    def execute_sql(self, statement: str, data: list[dict]):
        for line in data: 
            engine.execute(text(statement), **line)

