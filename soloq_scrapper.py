import os
import time
import logging
from datetime import datetime
from threading import Lock
from collections import defaultdict, deque
import pandas as pd
import requests

# ========================
# Configuration Classes
# ========================
class Config:
    """Centralized configuration"""
    API_KEY = os.getenv("RIOT_API_KEY")
    BASE_URL = "https://{region}.api.riotgames.com/lol"
    RATE_LIMITS = {
        "per_second": 18,  # 20 * 0.9 buffer
        "per_two_minutes": 90  # 100 * 0.9 buffer
    }
    REGIONS = ["na1", "euw1", "kr", "eun1", "br1", "la1", "la2", "oc1", "ru", "tr1", "jp1"]
    OUTPUT_DIR = "soloq_stats"
    
    @classmethod
    def validate(cls):
        if not cls.API_KEY:
            raise ValueError("RIOT_API_KEY environment variable not set")

# ========================
# Core Components
# ========================
class RateLimiter:
    """Thread-safe rate limiting"""
    def __init__(self):
        self.requests_second = deque(maxlen=Config.RATE_LIMITS["per_second"])
        self.requests_two_min = deque(maxlen=Config.RATE_LIMITS["per_two_minutes"])
        self.lock = Lock()

    def wait(self):
        with self.lock:
            now = time.time()
            self._clean_old_requests(now)
            
            if self._needs_throttle(now):
                sleep_time = self._calculate_sleep(now)
                logging.warning(f"Rate limit approaching. Sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)
            
            self.requests_second.append(now)
            self.requests_two_min.append(now)

    def _clean_old_requests(self, now):
        """Remove expired request timestamps"""
        while self.requests_second and now - self.requests_second[0] > 1:
            self.requests_second.popleft()
        while self.requests_two_min and now - self.requests_two_min[0] > 120:
            self.requests_two_min.popleft()

    def _needs_throttle(self, now):
        """Check if we're approaching limits"""
        return (len(self.requests_second) >= Config.RATE_LIMITS["per_second"] or
                len(self.requests_two_min) >= Config.RATE_LIMITS["per_two_minutes"])

    def _calculate_sleep(self, now):
        """Determine required sleep duration"""
        sleep_times = []
        if self.requests_second:
            sleep_times.append((self.requests_second[0] + 1) - now)
        if self.requests_two_min:
            sleep_times.append((self.requests_two_min[0] + 120) - now)
        return max(max(sleep_times) if sleep_times else 0, 0.1)

class RiotAPI:
    """Handles all Riot API communication"""
    def __init__(self, rate_limiter):
        self.rate_limiter = rate_limiter

    def get_challenger_league(self, region):
        url = f"{Config.BASE_URL.format(region=region)}/league/v4/challengerleagues/by-queue/RANKED_SOLO_5x5"
        return self._make_request(url)

    def get_summoner(self, region, summoner_id):
        url = f"{Config.BASE_URL.format(region=region)}/summoner/v4/summoners/{summoner_id}"
        return self._make_request(url)

    def _make_request(self, url):
        """Generic request handler"""
        self.rate_limiter.wait()
        
        try:
            response = requests.get(
                url,
                headers={"X-Riot-Token": Config.API_KEY},
                timeout=10
            )
            
            if response.status_code == 403:
                logging.error(f"403 Forbidden - Check API key permissions for: {url}")
                return None
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed: {str(e)}")
            return None

# ========================
# Data Processing
# ========================
class DataProcessor:
    @staticmethod
    def process_region(api_client, region):
        league_data = api_client.get_challenger_league(region)
        if not league_data:
            return None

        stats = []
        for player in league_data["entries"][:50]:  # Top 50 players
            stats.append({
                "summoner_id": player["summonerId"],
                "league_points": player["leaguePoints"],
                "wins": player["wins"],
                "losses": player["losses"],
                "veteran": player["veteran"],
                "inactive": player["inactive"],
                "hot_streak": player["hotStreak"],
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M")
            })
            time.sleep(0.2)
        return stats

    @classmethod
    def save_to_csv(cls, data, region):
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        filename = f"{Config.OUTPUT_DIR}/{region}_challenger_players.csv"
        pd.DataFrame(data).to_csv(filename, index=False)

# ========================
# Main Execution
# ========================
def main():
    """Clean main function"""
    # Setup
    Config.validate()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("scraper.log"),
            logging.StreamHandler()
        ]
    )

    # Initialize components
    rate_limiter = RateLimiter()
    api_client = RiotAPI(rate_limiter)
    processor = DataProcessor()

    # Process all regions
    for region in Config.REGIONS:
        try:
            region_stats = processor.process_region(api_client, region)
            if region_stats:
                processor.save_to_csv(region_stats, region)
        except Exception as e:
            logging.error(f"Failed processing {region}: {str(e)}")

    logging.info("✨ All regions processed!")

if __name__ == "__main__":
    main()