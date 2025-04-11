import requests
import pandas as pd
from collections import defaultdict, deque
import os
import time
import logging
from datetime import datetime
from threading import Lock

# --- Constants ---
API_KEY = os.getenv("RIOT_API_KEY")
if not API_KEY:
    raise ValueError("API key not found in environment variables")

BASE_URL = "https://{region}.api.riotgames.com/lol"

# YOUR EXACT RATE LIMITS (from your API key)
RATE_LIMITS = {
    "league": {
        "per_second": 20,
        "per_two_minutes": 100
    },
    "match": {
        "per_second": 20,
        "per_two_minutes": 100
    }
}

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class StrictRateLimiter:
    """Enforces your exact rate limits with buffer"""
    def __init__(self):
        self.queues = {
            "league": {
                "second": deque(maxlen=RATE_LIMITS["league"]["per_second"]),
                "two_minute": deque(maxlen=RATE_LIMITS["league"]["per_two_minutes"])
            },
            "match": {
                "second": deque(maxlen=RATE_LIMITS["match"]["per_second"]),
                "two_minute": deque(maxlen=RATE_LIMITS["match"]["per_two_minutes"])
            }
        }
        self.lock = Lock()
        self.BUFFER = 0.9  # 10% safety buffer

    def wait(self, endpoint_type):
        with self.lock:
            now = time.time()
            queues = self.queues[endpoint_type]

            # Clean old entries
            while queues["second"] and now - queues["second"][0] > 1:
                queues["second"].popleft()
            while queues["two_minute"] and now - queues["two_minute"][0] > 120:
                queues["two_minute"].popleft()

            # Calculate maximum allowed with buffer
            max_per_second = RATE_LIMITS[endpoint_type]["per_second"] * self.BUFFER
            max_per_two_min = RATE_LIMITS[endpoint_type]["per_two_minutes"] * self.BUFFER

            # Determine required wait
            required_wait = 0

            # Per-second limit check
            if len(queues["second"]) >= max_per_second:
                oldest = queues["second"][0]
                required_wait = max(required_wait, (oldest + 1) - now)

            # Per-two-minute limit check
            if len(queues["two_minute"]) >= max_per_two_min:
                oldest = queues["two_minute"][0]
                required_wait = max(required_wait, (oldest + 120) - now)

            if required_wait > 0:
                logger.warning(f"Rate limit approaching. Sleeping {required_wait:.2f}s")
                time.sleep(required_wait)

            # Record this request
            now = time.time()
            queues["second"].append(now)
            queues["two_minute"].append(now)

rate_limiter = StrictRateLimiter()

def make_request(url, endpoint_type):
    """Make a request with strict rate limiting"""
    rate_limiter.wait(endpoint_type)
    
    try:
        logger.debug(f"Requesting {url}")
        response = requests.get(
            url,
            headers={"X-Riot-Token": API_KEY},
            timeout=15
        )

        if response.status_code == 403:
            logger.error(f"403 Forbidden - Key may be invalid or expired")
            logger.error(f"Response: {response.text}")
            return None
        elif response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 10))
            logger.warning(f"Rate limited. Waiting {retry_after}s")
            time.sleep(retry_after)
            return make_request(url, endpoint_type)
        
        response.raise_for_status()
        return response.json()
    
    except Exception as e:
        logger.error(f"Request failed: {str(e)}")
        return None

# --- Data Collection Functions ---
def get_challenger_players(region):
    """Fetch challenger ladder (League-v4)"""
    url = f"{BASE_URL.format(region=region)}/league/v4/challengerleagues/by-queue/RANKED_SOLO_5x5"
    data = make_request(url, "league")
    
    if not data:
        logger.error(f"Failed to get challenger ladder for {region}")
        return None
    
    logger.info(f"Found {len(data['entries'])} challenger players in {region}")
    return data["entries"]

def get_puuid(region, summoner_id):
    """Get PUUID (Summoner-v4)"""
    url = f"{BASE_URL.format(region=region)}/summoner/v4/summoners/{summoner_id}"
    data = make_request(url, "league")
    return data.get("puuid") if data else None

def get_match_history(region, puuid, count=20):
    """Get match IDs (Match-v5)"""
    url = f"{BASE_URL.format(region=region)}/match/v5/matches/by-puuid/{puuid}/ids?count={count}"
    return make_request(url, "match")

def process_match(region, match_id):
    """Process a single match (Match-v5)"""
    url = f"{BASE_URL.format(region=region)}/match/v5/matches/{match_id}"
    return make_request(url, "match")

# --- Main Workflow ---
def scrape_region(region, max_players=50, matches_per_player=10):
    """Scrape data for one region"""
    logger.info(f"\n{'='*50}")
    logger.info(f"Starting scrape for {region.upper()}")
    
    players = get_challenger_players(region)
    if not players:
        return None
    
    champion_stats = defaultdict(lambda: {"wins": 0, "games": 0})
    processed_players = 0
    
    for player in players[:max_players]:
        puuid = get_puuid(region, player["summonerId"])
        if not puuid:
            continue
            
        logger.info(f"Processing {player['summonerName']} ({processed_players+1}/{max_players})")
        
        match_ids = get_match_history(region, puuid, matches_per_player)
        if not match_ids:
            continue
            
        for match_id in match_ids:
            match = process_match(region, match_id)
            if not match or match["info"]["queueId"] != 420:
                continue
                
            for p in match["info"]["participants"]:
                if p["puuid"] == puuid:
                    champ = p["championName"]
                    champion_stats[champ]["games"] += 1
                    champion_stats[champ]["wins"] += int(p["win"])
        
        processed_players += 1
        time.sleep(0.5)  # Small delay between players
    
    logger.info(f"Finished {region.upper()} with {processed_players} players processed")
    return champion_stats

if __name__ == "__main__":
    logger.info("Starting scraper with STRICT rate limiting")
    
    regions = ["na1"]  # Start with one region
    all_stats = defaultdict(lambda: {"wins": 0, "games": 0})
    
    for region in regions:
        region_stats = scrape_region(region)
        if region_stats:
            for champ, stats in region_stats.items():
                all_stats[champ]["wins"] += stats["wins"]
                all_stats[champ]["games"] += stats["games"]
    
    # Save results
    df = pd.DataFrame([
        {
            "Champion": champ,
            "WinRate": round((stats["wins"] / stats["games"]) * 100, 2),
            "Games": stats["games"],
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        for champ, stats in all_stats.items()
        if stats["games"] > 0
    ]).sort_values("WinRate", ascending=False)
    
    df.to_csv("challenger_win_rates.csv", index=False)
    logger.info("Saved results to challenger_win_rates.csv")