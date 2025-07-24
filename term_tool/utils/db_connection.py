from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import ResourceClosedError
import pandas as pd
from sqlalchemy.pool import NullPool
import logging


logger = logging.getLogger("app")
logger.setLevel(logging.INFO)


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
        try:
            self.session = self.Session()
                    # Test connection
           
            connection = self.engine.connect()
            connection.close()
        
            print("Database connection established successfully")
            return self.engine
        except Exception as e:
            print(f"Failed to establish database connection: {e}")
            raise

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
    
    def execute_sql(self, statement: str, data: list[dict]) -> dict:
        result = self.session.execute(text(statement), data)
        try:
            return {'data': result.fetchall(),
                    'keys': result.keys()}
        except ResourceClosedError as e:
            return  {}
        except Exception as e:
            self.session.rollback()
            raise Exception('Failed to execute statement:', e)
    
    def statement_to_df(self, statement: str):
        try:
            result = self.execute_sql(statement, [])
            logger.info(result)
            df = pd.DataFrame(result['data'])
            df.columns = result['keys']
            return df 
        except Exception as e:
            raise Exception(e)
            

    def commit(self):
        self.session.commit()

    def close(self):
        self.session.close()