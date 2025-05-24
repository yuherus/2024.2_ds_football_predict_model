from sqlalchemy import create_engine, text
from backend.features.utils import get_pg_engine

def create_championship_probabilities_table():
    engine = get_pg_engine()
    
    # Drop table if exists
    drop_table_query = """
    DROP TABLE IF EXISTS championship_probabilities;
    """
    
    create_table_query = """
    CREATE TABLE championship_probabilities (
        id SERIAL PRIMARY KEY,
        rank INTEGER,
        team_name VARCHAR(255),
        championship_probability FLOAT,
        points INTEGER,
        form VARCHAR(5),
        league VARCHAR(255),
        season VARCHAR(255),
        remaining_matches INTEGER,
        max_possible_points INTEGER,
        prediction_model VARCHAR(255),
        simulation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    
    with engine.begin() as conn:
        conn.execute(text(drop_table_query))
        conn.execute(text(create_table_query))

if __name__ == "__main__":
    create_championship_probabilities_table() 