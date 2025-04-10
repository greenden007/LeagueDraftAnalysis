import requests
import pandas as pd
from collections import defaultdict
import os
import time
import logging

# Riot API Key
# Only I (Idan) has this key
API_KEY = os.getenv("RIOT_API_KEY")
if not API_KEY:
    raise ValueError("API key not found! Please set the RIOT_API_KEY environment variable.") 

# Base URL for Riot API
BASE_URL = "https://{region}.api.riotgames.com/lol"

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_latest_patches():
    """Fetch the latest two patch versions from Riot's Data Dragon API."""
    url = "https://ddragon.leagueoflegends.com/api/versions.json"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Failed to fetch patch versions")
    
    versions = response.json()  # List of patch versions
    latest_patches = versions[:2]  # Get the first two (latest) patches
    logging.info(f"Latest patches: {latest_patches}")
    return latest_patches

def rate_limit_delay(requests_made, start_time):
    """Ensure we don't exceed Riot's API rate limits."""
    # Riot API limits: 20 requests per second, 100 requests per minute
    if requests_made >= 20:  # Check if we’ve hit the per-second limit
        elapsed_time = time.time() - start_time
        if elapsed_time < 1:  # If less than a second has passed, wait
            time.sleep(1 - elapsed_time)
        return 0, time.time()  # Reset the counter and start time

    return requests_made, start_time

def get_challenger_win_rates():
    """Fetch match data and calculate champion win rates for Challenger Solo Queue."""
    regions = ["na1", "euw1", "kr", "eun1", "jp1", "oc1", "br1", "ru", "tr1", "la1", "la2"]
    roles = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
    champion_stats = defaultdict(lambda: {role: {"wins": 0, "games": 0} for role in roles})
    
    # Create output folder if it doesn't exist
    output_folder = "soloq_challenger_winrates"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        logging.info(f"Directory '{output_folder}' created successfully!")
    
    # Two latest patches
    latest_patches = get_latest_patches()

    # Initialize rate-limiting variables
    requests_made = 0
    start_time = time.time()
    
    for region in regions:
        logging.info(f"Processing region: {region}")
        
        try:
            # Step 1: Get Challenger players' PUUIDs
            url = f"{BASE_URL}/league/v4/challengerleagues/by-queue/RANKED_SOLO_5x5"
            response = requests.get(url.format(region=region), headers={"X-Riot-Token": API_KEY})
            requests_made += 1
            requests_made, start_time = rate_limit_delay(requests_made, start_time)
            
            if response.status_code == 429:
                logging.error("Rate limit exceeded while fetching Challenger data. Pausing for 60 seconds...")
                time.sleep(60)  # Wait for a minute before retrying
                continue
            
            if response.status_code != 200:
                logging.warning(f"Failed to fetch challenger data for region {region}: {response.status_code}")
                continue
            
            players = response.json()["entries"]
            puuids = []
            for player in players:
                summoner_id = player["summonerId"]
                summoner_url = f"{BASE_URL}/summoner/v4/summoners/{summoner_id}"
                summoner_response = requests.get(summoner_url.format(region=region), headers={"X-Riot-Token": API_KEY})
                requests_made += 1
                requests_made, start_time = rate_limit_delay(requests_made, start_time)
                
                if summoner_response.status_code == 429:
                    logging.error("Rate limit exceeded while fetching PUUIDs. Pausing for 60 seconds...")
                    time.sleep(60)
                    continue
                
                if summoner_response.status_code == 200:
                    puuids.append(summoner_response.json()["puuid"])
            
            patch_counts = defaultdict(int)
            
            # Step 2: Get match history for each player
            for puuid in puuids:
                start_index = 0
                while True:  # Paginate through all matches
                    match_url = f"{BASE_URL}/match/v5/matches/by-puuid/{puuid}/ids?start={start_index}&count=100"
                    match_response = requests.get(match_url.format(region=region), headers={"X-Riot-Token": API_KEY})
                    requests_made += 1
                    requests_made, start_time = rate_limit_delay(requests_made, start_time)
                    
                    if match_response.status_code == 429:
                        logging.error("Rate limit exceeded while fetching match history. Pausing for 60 seconds...")
                        time.sleep(60)
                        continue
                    
                    if match_response.status_code != 200 or not match_response.json():
                        break
                    
                    match_ids = match_response.json()
                    start_index += 100
                    
                    # Step 3: Process each match
                    for match_id in match_ids:
                        match_data_url = f"{BASE_URL}/match/v5/matches/{match_id}"
                        match_data_response = requests.get(match_data_url.format(region=region), headers={"X-Riot-Token": API_KEY})
                        requests_made += 1
                        requests_made, start_time = rate_limit_delay(requests_made, start_time)
                        
                        if match_data_response.status_code == 429:
                            logging.error("Rate limit exceeded while fetching match data. Pausing for 60 seconds...")
                            time.sleep(60)
                            continue
                        
                        if match_data_response.status_code != 200:
                            continue
                        
                        match_data = match_data_response.json()
                        
                        # Filter by ranked Solo Queue matches only and patch version
                        if match_data["info"]["queueId"] != 420:
                            continue
                        
                        game_version = match_data["info"]["gameVersion"].split(".")
                        patch_version = f"{game_version[0]}.{game_version[1]}"
                        if patch_version not in latest_patches:
                            logging.info(f"Skipping match ID {match_id} from older patch: {patch_version}")
                            continue
                        
                        logging.info(f"Match ID: {match_id}, Patch Version: {patch_version}")
                        patch_counts[patch_version] += 1
                        
                        participants = match_data["info"]["participants"]
                        
                        # Step 4: Aggregate stats by champion and role
                        for participant in participants:
                            champion_name = participant["championName"]
                            role = participant["teamPosition"]
                            win = participant["win"]
                            
                            if role in roles:
                                champion_stats[champion_name][role]["games"] += 1
                                if win:
                                    champion_stats[champion_name][role]["wins"] += 1

            logging.info("Matches per patch:")
            for patch, count in patch_counts.items():
                logging.info(f"Patch {patch}: {count} matches")
        
        except Exception as e:
            logging.error(f"An unexpected error occurred while processing region {region}: {e}")
                
    # Step 5: Calculate win rates and save to CSV
    rows = []
    for champion, stats in champion_stats.items():
        for role, data in stats.items():
            if data["games"] > 0:
                win_rate = (data["wins"] / data["games"]) * 100
                rows.append({"Champion": champion, "Role": role, "Win Rate": win_rate, "Games Played": data["games"]})
    
    df = pd.DataFrame(rows)
    output_path = os.path.join(output_folder, "challenger_win_rates.csv")
    df.to_csv(output_path, index=False)
    logging.info(f"Win rate data saved to {output_path}")

    
if __name__ == "__main__":
    # Call the main function when this script is executed directly
    get_challenger_win_rates()
