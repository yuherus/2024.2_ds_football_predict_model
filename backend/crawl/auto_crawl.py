import os
import schedule
import time
import pandas as pd
from datetime import datetime
from craw_data import crawl_match_results
from crawl_match_results import crawl_fbref_match_results
from crawl_player_value import crawl_league_teams, crawl_league_team_values
from crawl_standing import crawl_standings

def get_current_season():
    """Get the current season based on the current date"""
    current_date = datetime.now()
    return current_date.year

def check_and_create_data_directory():
    """Check if data directory exists, if not create it"""
    if not os.path.exists("data"):
        os.makedirs("data")

def check_season_data(league, season):
    """Check if we have data for a specific league and season"""
    data_files = {
        "match_results": f"data/match_results_{league}_{season}.csv",
        "match_details": f"data/match_details_{league}_{season}.csv",
        "player_values": f"data/player_values_{league}_{season}.csv",
        "team_values": f"data/team_values_{league}_{season}.csv",
        "standings": f"data/standings_{league}_{season}.csv"
    }
    
    # Check if all files exist and are not empty
    for file_path in data_files.values():
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            return False
    return True

def crawl_new_season_data():
    """Crawl data for the new season"""
    current_season = get_current_season()
    leagues = ["premierleague", "laliga", "seriea", "bundesliga", "ligue1"]
    
    # Create data directory if it doesn't exist
    check_and_create_data_directory()
    
    for league in leagues:
        print(f"\nCrawling data for {league} season {current_season}...")
        
        try:
            # Crawl match results
            match_results = crawl_match_results(league, current_season)
            pd.DataFrame(match_results).to_csv(f"data/match_results_{league}_{current_season}.csv", index=False)
            
            # Crawl match details from FBref
            match_details = crawl_fbref_match_results(league, current_season)
            pd.DataFrame(match_details).to_csv(f"data/match_details_{league}_{current_season}.csv", index=False)
            
            # Crawl player values
            player_values = crawl_league_teams(league, current_season)
            pd.DataFrame(player_values).to_csv(f"data/player_values_{league}_{current_season}.csv", index=False)
            
            # Crawl team values
            team_values = crawl_league_team_values(league, current_season)
            pd.DataFrame(team_values).to_csv(f"data/team_values_{league}_{current_season}.csv", index=False)
            
            # Crawl standings
            standings = crawl_standings(league, current_season)
            pd.DataFrame(standings).to_csv(f"data/standings_{league}_{current_season}.csv", index=False)
            
            print(f"Successfully crawled all data for {league} season {current_season}")
            
        except Exception as e:
            print(f"Error crawling data for {league} season {current_season}: {str(e)}")
            continue

def job():
    """Main job to run the crawl at July 1st midnight"""
    print(f"Starting crawl for new season at {datetime.now()}")
    crawl_new_season_data()

def main():
    # Schedule to run at midnight on July 1st
    schedule.every().year.at("00:00").do(job)
    
    print("Auto-crawl service started. Will run at 00:00 on July 1st each year...")
    
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    main() 