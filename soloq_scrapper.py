#!/usr/bin/env python3
"""
Riot API Challenger Scraper (With KDA Stats)
- Collects champion win rates AND performance metrics from all regions
- Requires Match-v5 API access
- Production-ready with comprehensive logging
"""

import os
import time
import logging
from datetime import datetime
from threading import Lock
from collections import defaultdict, deque
import pandas as pd
import requests

# ========================
# Configuration
# ========================
class Config:
    # API Settings
    API_KEY = os.getenv("RIOT_API_KEY")
    BASE_URL = "https://{region}.api.riotgames.com/lol"
    
    # Match-v5 Regional Routing
    MATCH_REGION_MAP = {
        "na1": "americas",
        "br1": "americas",
        "la1": "americas",
        "la2": "americas",
        "oc1": "americas",
        "euw1": "europe",
        "eun1": "europe",
        "tr1": "europe",
        "ru": "europe",
        "kr": "asia",
        "jp1": "asia"
    }
    
    # Rate Limits (with 10% buffer)
    RATE_LIMITS = {
        "league": {"per_second": 18, "per_two_minutes": 90},
        "match": {"per_second": 18, "per_two_minutes": 90}
    }
    
    # Execution Parameters
    REGIONS = list(MATCH_REGION_MAP.keys())
    MAX_PLAYERS = 50  # Players per region to process
    MAX_MATCHES = 20  # Matches per player
    OUTPUT_DIR = "soloq_stats"
    REQUEST_TIMEOUT = 15  # seconds
    MIN_GAMES_THRESHOLD = 3  # Minimum games to include champion in stats
    
    @classmethod
    def validate(cls):
        if not cls.API_KEY:
            raise ValueError("RIOT_API_KEY environment variable not set")
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)

# ========================
# Logger Setup (FIXED)
# ========================
def configure_logging():
    """Configures root logger with file and console output"""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler("scraper.log")
    fh.setLevel(logging.DEBUG)
    
    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    
    logger.addHandler(ch)
    logger.addHandler(fh)
    
    return logger

logger = configure_logging()

# ========================
# Core Components
# ========================
class RateLimiter:
    """Thread-safe rate limiter with separate buckets per endpoint type"""
    def __init__(self):
        self.buckets = {
            "league": {
                "second": deque(maxlen=Config.RATE_LIMITS["league"]["per_second"]),
                "minute": deque(maxlen=Config.RATE_LIMITS["league"]["per_two_minutes"])
            },
            "match": {
                "second": deque(maxlen=Config.RATE_LIMITS["match"]["per_second"]),
                "minute": deque(maxlen=Config.RATE_LIMITS["match"]["per_two_minutes"])
            }
        }
        self.lock = Lock()
        self.last_request = 0

    def wait(self, endpoint_type):
        with self.lock:
            now = time.time()
            bucket = self.buckets[endpoint_type]
            
            # Clean old requests
            for ts in list(bucket["second"]):
                if now - ts > 1:
                    bucket["second"].remove(ts)
            for ts in list(bucket["minute"]):
                if now - ts > 120:
                    bucket["minute"].remove(ts)
            
            # Calculate required delay
            delay = max(
                self._calculate_delay(bucket["second"], now, 1),
                self._calculate_delay(bucket["minute"], now, 120),
                Config.REQUEST_TIMEOUT - (now - self.last_request)
            )
            
            if delay > 0:
                logger.debug(f"Rate limit pacing: Sleeping {delay:.2f}s")
                time.sleep(delay)
            
            # Record request
            bucket["second"].append(now)
            bucket["minute"].append(now)
            self.last_request = now

    def _calculate_delay(self, queue, now, window):
        if len(queue) >= queue.maxlen:
            return (queue[0] + window) - now
        return 0

class RiotAPI:
    """Handles all Riot API communication with automatic region routing"""
    def __init__(self, rate_limiter):
        self.rate_limiter = rate_limiter
        self.session = requests.Session()
        self.session.headers.update({
            "X-Riot-Token": Config.API_KEY,
            "User-Agent": "ChallengerAnalytics/1.0",
            "Accept-Language": "en-US"
        })

    # --- League-v4 Endpoints ---
    def get_challenger_league(self, region):
        url = f"{Config.BASE_URL.format(region=region)}/league/v4/challengerleagues/by-queue/RANKED_SOLO_5x5"
        return self._make_request(url, "league")

    def get_summoner(self, region, summoner_id):
        url = f"{Config.BASE_URL.format(region=region)}/summoner/v4/summoners/{summoner_id}"
        return self._make_request(url, "league")

    # --- Match-v5 Endpoints ---
    def get_match_history(self, puuid, region):
        match_region = Config.MATCH_REGION_MAP[region]
        url = f"https://{match_region}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?queue=420&count={Config.MAX_MATCHES}"
        return self._make_request(url, "match")

    def get_match_details(self, match_id, region):
        match_region = Config.MATCH_REGION_MAP[region]
        url = f"https://{match_region}.api.riotgames.com/lol/match/v5/matches/{match_id}"
        return self._make_request(url, "match")

    def _make_request(self, url, endpoint_type):
        """Universal request handler with retry logic"""
        self.rate_limiter.wait(endpoint_type)
        
        try:
            logger.debug(f"Requesting {url}")
            response = self.session.get(url, timeout=Config.REQUEST_TIMEOUT)
            
            if response.status_code == 403:
                logger.error(f"403 Forbidden - Verify API key permissions for: {url}")
                return None
            elif response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 10))
                logger.warning(f"Rate limited. Waiting {retry_after}s")
                time.sleep(retry_after)
                return self._make_request(url, endpoint_type)
            elif response.status_code != 200:
                logger.warning(f"Unexpected status {response.status_code} for {url}")
                return None
                
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            return None

# ========================
# Data Processing (Enhanced with KDA)
# ========================
class ChampionStats:
    """Tracks comprehensive champion performance metrics"""
    def __init__(self):
        self.games = 0
        self.wins = 0
        self.kills = 0
        self.deaths = 0
        self.assists = 0
    
    def add_game(self, win, kills, deaths, assists):
        self.games += 1
        self.wins += int(win)
        self.kills += kills
        self.deaths += deaths
        self.assists += assists
    
    @property
    def win_rate(self):
        return (self.wins / self.games) * 100 if self.games > 0 else 0
    
    @property
    def kda(self):
        return (self.kills + self.assists) / max(1, self.deaths)

class ChampionAnalyzer:
    """Processes match data into champion statistics"""
    @staticmethod
    def process_player(player_data, matches):
        """Analyzes a player's match history with KDA tracking"""
        stats = defaultdict(ChampionStats)
        
        for match in matches:
            if not match:
                continue
                
            for p in match["info"]["participants"]:
                if p["puuid"] == player_data["puuid"]:
                    champ = p["championName"]
                    stats[champ].add_game(
                        win=p["win"],
                        kills=p["kills"],
                        deaths=p["deaths"],
                        assists=p["assists"]
                    )
        
        return stats

    @classmethod
    def aggregate_region_stats(cls, all_players):
        """Combines data from multiple players"""
        region_stats = defaultdict(ChampionStats)
        for player_stats in all_players:
            for champ, stats in player_stats.items():
                region_stats[champ].games += stats.games
                region_stats[champ].wins += stats.wins
                region_stats[champ].kills += stats.kills
                region_stats[champ].deaths += stats.deaths
                region_stats[champ].assists += stats.assists
        return region_stats

class DataExporter:
    """Handles data storage and formatting with KDA stats"""
    @staticmethod
    def save_region_stats(region, stats):
        """Saves region data to CSV with enhanced metrics"""
        filename = f"{Config.OUTPUT_DIR}/{region}_champion_stats.csv"
        
        records = []
        for champ, data in stats.items():
            if data.games >= Config.MIN_GAMES_THRESHOLD:
                records.append({
                    "champion": champ,
                    "games": data.games,
                    "wins": data.wins,
                    "win_rate": round(data.win_rate, 2),
                    "avg_kills": round(data.kills / data.games, 2),
                    "avg_deaths": round(data.deaths / data.games, 2),
                    "avg_assists": round(data.assists / data.games, 2),
                    "kda": round(data.kda, 2),
                    "region": region,
                    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
        
        df = pd.DataFrame(records).sort_values("win_rate", ascending=False)
        df.to_csv(filename, index=False)
        logger.info(f"Saved {filename} ({len(df)} champions)")
        return filename

# ========================
# Workflow Orchestration
# ========================
class Scraper:
    """Main scraping orchestrator"""
    def __init__(self):
        self.rate_limiter = RateLimiter()
        self.api = RiotAPI(self.rate_limiter)
        self.processed_regions = 0

    def process_region(self, region):
        """Full pipeline for one region"""
        logger.info(f"\n{'='*50}")
        logger.info(f"🌍 Processing {region.upper()} (Region {self.processed_regions + 1}/{len(Config.REGIONS)})")
        
        try:
            # Step 1: Get challenger ladder
            league_data = self._get_challenger_league(region)
            if not league_data:
                return False

            # Step 2: Process players
            player_stats = self._process_players(region, league_data["entries"])
            if not player_stats:
                return False

            # Step 3: Aggregate and save
            region_stats = ChampionAnalyzer.aggregate_region_stats(player_stats)
            DataExporter.save_region_stats(region, region_stats)
            
            self.processed_regions += 1
            return True
            
        except Exception as e:
            logger.error(f"Failed processing {region}: {str(e)}", exc_info=True)
            return False

    def _get_challenger_league(self, region):
        """Retrieves challenger ladder with retry logic"""
        for attempt in range(3):
            data = self.api.get_challenger_league(region)
            if data:
                logger.info(f"Found {len(data['entries'])} challenger players in {region} - analyzing top {Config.MAX_PLAYERS} by LP")
                return data
            time.sleep(2 ** attempt)
        logger.error(f"Failed to get challenger league after 3 attempts")
        return None

    def _process_players(self, region, players):
        """Processes all players in a region"""
        all_player_stats = []
        players = sorted(players, key=lambda x: x["leaguePoints"], reverse=True)[:Config.MAX_PLAYERS]
        
        for i, player in enumerate(players, 1):
            player_stats = self._process_player(region, player, i)
            if player_stats:
                all_player_stats.append(player_stats)
            time.sleep(0.5)  # Delay between players
            
        return all_player_stats

    def _process_player(self, region, player, player_index):
        """Processes a single player's match history"""
        logger.info(f"Processing player {player_index}/{Config.MAX_PLAYERS} (LP: {player['leaguePoints']})")
        
        # Get PUUID
        summoner = self.api.get_summoner(region, player["summonerId"])
        if not summoner or "puuid" not in summoner:
            logger.warning(f"Skipping player {player['summonerId'][:6]} (no PUUID)")
            return None

        # Get match history
        match_ids = self.api.get_match_history(summoner["puuid"], region)
        if not match_ids:
            logger.warning(f"No matches found for {summoner['puuid'][:6]}")
            return None

        # Get match details
        matches = []
        for match_id in match_ids:
            match = self.api.get_match_details(match_id, region)
            if match and match["info"]["queueId"] == 420:  # Ranked SoloQ only
                matches.append(match)
            time.sleep(0.1)  # Small delay between match requests

        return ChampionAnalyzer.process_player({
            "puuid": summoner["puuid"],
            "summoner_id": player["summonerId"]
        }, matches)

# ========================
# Main Execution
# ========================
def main():
    """Entry point with error handling"""
    logger.info("🚀 Starting Riot API Challenger Scraper (With KDA Stats)")
    
    try:
        Config.validate()
        scraper = Scraper()
        
        for region in Config.REGIONS:
            start_time = time.time()
            success = scraper.process_region(region)
            elapsed = time.time() - start_time
            
            if success:
                logger.info(f"✅ Completed {region.upper()} in {elapsed:.1f}s")
            else:
                logger.warning(f"⚠️ Partial completion for {region.upper()}")

            # Region cooldown
            if region != Config.REGIONS[-1]:
                cooldown = max(10, 30 - elapsed)
                logger.info(f"Cooling down for {cooldown:.1f}s...")
                time.sleep(cooldown)
                
    except Exception as e:
        logger.critical(f"💥 Critical failure: {str(e)}", exc_info=True)
    finally:
        logger.info("✨ Scraping completed")

if __name__ == "__main__":
    main()