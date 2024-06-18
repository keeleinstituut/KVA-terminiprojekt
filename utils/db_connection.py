from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import ResourceClosedError
import pandas as pd
from sqlalchemy.pool import NullPool

class Connection():
    def __init__(self, host: str, port: int, user: str, password: str, db: str):
        """
        Initialize the Connection object.
        """
        self.connection_string = f'postgresql://{user}:{password}@{host}:{port}/{db}'
        self.engine = create_engine(self.connection_string, poolclass=NullPool)
        self.Session = sessionmaker(bind=self.engine)
        self.session = None


    def establish_connection(self):
        self.session = self.Session()
        return self.engine

    def table_to_dataframe(self, table_name: str) -> pd.DataFrame:
        """
        Convert a table from the database to a pandas DataFrame.

        :param table_name: The name of the table to convert
        :return: A pandas DataFrame containing the table data
        """
        if not self.engine:
            raise ValueError("No connection established. Call establish_connection() first.")
        

        query = text(""" SELECT * from :table """)
        data = {'table': table_name} 

        result = self.session.execute(query, data)
        print(result)

        # Convert to DataFrame
        df = pd.DataFrame(result.fetchall())

        return df
    
    def execute_sql(self, statement: str, data: list[dict]):
        result = self.session.execute(text(statement), data)
        try:
            return result.fetchall()
        except ResourceClosedError as e:
            print(e)
            return []
    
    def commit(self):
        self.session.commit()

    def close(self):
        self.session.close()