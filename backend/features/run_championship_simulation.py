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
    
    prediction_models = ['lstm', 'xgboost']
    total_combinations = len(leagues) * len(seasons) * len(prediction_models)
    current = 0
    start_time = time.time()
    
    for league in leagues:
        for season in seasons:
            for model in prediction_models:
                current += 1
                print(f"\nProcessing {league} {season} with {model.upper()} model ({current}/{total_combinations})")
                try:
                    results = calculate_championship_probabilities(league, season, model)
                    if results:
                        print(f"\nChampionship Results for {league} {season} ({model.upper()}):")
                        print("="*100)
                        print(f"{'Rank':<6} {'Team':<30} {'Championships':<15} {'Probability':<15} {'Points':<10} {'Remaining':<10} {'Form':<10}")
                        print("-"*100)
                        for team in results:
                            championships = int(team['championship_probability'] * 1000)  # Convert probability to number of championships
                            print(f"{team['rank']:<6} {team['team_name']:<30} {championships:<15} {team['championship_probability']:.2%} {team['points']:<10} {team['remaining_matches']:<10} {team['form']:<10}")
                        print("="*100)
                    else:
                        print(f"No data available for {league} {season} with {model.upper()} model")
                except Exception as e:
                    print(f"Error processing {league} {season} with {model.upper()} model: {str(e)}")
                    continue
                
                # Calculate and display progress
                elapsed_time = time.time() - start_time
                avg_time_per_combination = elapsed_time / current
                remaining_combinations = total_combinations - current
                estimated_remaining_time = avg_time_per_combination * remaining_combinations
                
                print(f"\nProgress: {current}/{total_combinations} combinations processed")
                print(f"Estimated time remaining: {estimated_remaining_time/60:.1f} minutes")
    
    total_time = time.time() - start_time
    print(f"\nChampionship probability calculation completed in {total_time/60:.1f} minutes!")

if __name__ == "__main__":
    # Example usage with correct league names from database
    leagues = ["premierleague", "bundesliga", "seriea"]  # Add other leagues as they appear in the database
    seasons = ["2023-2024"]  # Match the season format in the database
    
    run_championship_calculation(leagues, seasons) 