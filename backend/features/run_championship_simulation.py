from backend.features.create_championship_table import create_championship_probabilities_table
from backend.features.championship_probability import calculate_championship_probabilities
from tqdm import tqdm
import time

def run_championship_calculation(leagues, seasons):
    """
    Run championship probability calculations for specified leagues and seasons.
    
    Args:
        leagues (list): List of league names
        seasons (list): List of seasons
    """
    print("Creating championship probabilities table...")
    try:
        create_championship_probabilities_table()
        print("Table created successfully!")
    except Exception as e:
        print(f"Error creating table: {str(e)}")
        return
    
    total_combinations = len(leagues) * len(seasons)
    current = 0
    start_time = time.time()
    
    for league in leagues:
        for season in seasons:
            current += 1
            print(f"\nProcessing {league} {season} ({current}/{total_combinations})")
            try:
                results = calculate_championship_probabilities(league, season)
                if results:
                    print(f"\nChampionship Results for {league} {season}:")
                    print("="*80)
                    print(f"{'Team':<30} {'Championships':<15} {'Probability':<15}")
                    print("-"*80)
                    for team in results:
                        championships = int(team['championship_probability'] * 10000)  # Convert probability to number of championships
                        print(f"{team['team_name']:<30} {championships:<15} {team['championship_probability']:.2%}")
                    print("="*80)
                else:
                    print(f"No data available for {league} {season}")
            except Exception as e:
                print(f"Error processing {league} {season}: {str(e)}")
                continue
            
            # Calculate and display progress
            elapsed_time = time.time() - start_time
            avg_time_per_league = elapsed_time / current
            remaining_leagues = total_combinations - current
            estimated_remaining_time = avg_time_per_league * remaining_leagues
            
            print(f"\nProgress: {current}/{total_combinations} leagues processed")
            print(f"Estimated time remaining: {estimated_remaining_time/60:.1f} minutes")
    
    total_time = time.time() - start_time
    print(f"\nChampionship probability calculation completed in {total_time/60:.1f} minutes!")

if __name__ == "__main__":
    # Example usage with correct league names from database
    leagues = ["premierleague", "bundesliga", "seriea"]  # Add other leagues as they appear in the database
    seasons = ["2023-2024"]  # Match the season format in the database
    
    run_championship_calculation(leagues, seasons) 