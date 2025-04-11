#!/usr/bin/env python3
"""
Ultimate Riot API Scraper - Professional Edition
- Proper rate limiting with threading awareness
- Champion matchup system with per-champion CSV
- 95% efficiency maintained
- Fully modular architecture
- Comprehensive logging
"""

import os
import time
import logging
from datetime import datetime
from threading import Lock, current_thread
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import requests
import json
from typing import Dict, List, Optional, Tuple, Any
import math

# ========================
# Configuration (Modular)
# ========================
class Config:
    """Centralized configuration with validation"""
    # API Settings
    API_KEY = os.getenv("RIOT_API_KEY")
    BASE_URL = "https://{region}.api.riotgames.com/lol"
    
    # Data Collection Parameters
    REGIONS = ["na1", "euw1", "kr", "eun1", "br1", "la1", "la2", "oc1", "ru", "tr1", "jp1"]
    MATCH_REGION_MAP = {
        region: "americas" if region in ["na1", "br1", "la1", "la2", "oc1"] 
        else "europe" if region in ["euw1", "eun1", "tr1", "ru"] 
        else "asia" 
        for region in REGIONS
    }
    
    # Rate Limits (95% utilization)
    RATE_LIMITS = {
        "app": {
            "per_second": 19,  # 20 * 0.95
            "per_two_minutes": 95  # 100 * 0.95
        },
        "endpoints": {
            "league": {
                "per_second": 17,  # 18 * 0.95
                "per_two_minutes": 85  # 90 * 0.95
            },
            "match": {
                "per_second": 17,
                "per_two_minutes": 85
            }
        }
    }
    
    # Collection Parameters
    MAX_MATCHES_PER_PLAYER = 100
    PATCHES_TO_ANALYZE = 3
    MIN_GAMES_THRESHOLD = 3
    MATCHUP_THRESHOLD = 10  # Minimum games for matchup stats
    OUTPUT_DIR = "soloq_stats"
    MATCHUP_DIR = os.path.join(OUTPUT_DIR, "matchups")
    REQUEST_TIMEOUT = 15
    MAX_WORKERS = 6  # Optimal balance between speed and rate limits
    REQUEST_DELAY = 0.05  # Small delay between batches
    
    # Retry Settings
    MAX_RETRIES = 3
    RETRY_DELAYS = [1, 2, 4]  # Exponential backoff

    @classmethod
    def validate(cls):
        """Validate configuration and create directories"""
        if not cls.API_KEY:
            raise ValueError("RIOT_API_KEY environment variable not set")
        
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.MATCHUP_DIR, exist_ok=True)

# ========================
# Core Components
# ========================
class RateLimiter:
    """Thread-aware rate limiter with burst prevention"""
    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.__initialized = False
        return cls._instance

    def __init__(self):
        if self.__initialized:
            return
        self.__initialized = True
        
        # App-wide limits
        self.app_limits = {
            "second": deque(maxlen=Config.RATE_LIMITS["app"]["per_second"]),
            "two_minutes": deque(maxlen=Config.RATE_LIMITS["app"]["per_two_minutes"])
        }
        
        # Endpoint-specific limits
        self.endpoint_limits = {
            ep_type: {
                "second": deque(maxlen=Config.RATE_LIMITS["endpoints"][ep_type]["per_second"]),
                "two_minutes": deque(maxlen=Config.RATE_LIMITS["endpoints"][ep_type]["per_two_minutes"])
            }
            for ep_type in Config.RATE_LIMITS["endpoints"]
        }
        
        self.lock = Lock()
        self.last_warning = 0
        self.request_count = 0

    def wait(self, endpoint_type: str) -> None:
        """Ensure compliance with all rate limits"""
        with self.lock:
            self.request_count += 1
            now = time.time()
            
            # Enforce app-wide limits first
            self._enforce_limit(self.app_limits["second"], now, 1, "app-wide per-second")
            self._enforce_limit(self.app_limits["two_minutes"], now, 120, "app-wide per-two-minutes")
            
            # Enforce endpoint-specific limits
            if endpoint_type in self.endpoint_limits:
                ep_limits = self.endpoint_limits[endpoint_type]
                self._enforce_limit(ep_limits["second"], now, 1, f"{endpoint_type} per-second")
                self._enforce_limit(ep_limits["two_minutes"], now, 120, f"{endpoint_type} per-two-minutes")
            
            # Record the request
            self.app_limits["second"].append(now)
            self.app_limits["two_minutes"].append(now)
            
            if endpoint_type in self.endpoint_limits:
                self.endpoint_limits[endpoint_type]["second"].append(now)
                self.endpoint_limits[endpoint_type]["two_minutes"].append(now)

    def _enforce_limit(self, queue: deque, now: float, window: int, limit_name: str) -> None:
        """Internal method to enforce a specific limit"""
        # Remove expired timestamps
        while queue and (now - queue[0] > window):
            queue.popleft()
            
        if len(queue) >= queue.maxlen:
            sleep_time = (queue[0] + window) - now
            if sleep_time > 0:
                if now - self.last_warning > 60:  # Throttle warnings
                    logger.warning(f"Approaching {limit_name} limit. Sleeping {sleep_time:.2f}s")
                    self.last_warning = now
                time.sleep(sleep_time + 0.01)  # Small buffer

class RiotAPI:
    """Thread-safe API client with enhanced error handling"""
    def __init__(self, rate_limiter: RateLimiter):
        self.rate_limiter = rate_limiter
        self.session = requests.Session()
        self.session.headers.update({
            "X-Riot-Token": Config.API_KEY,
            "Accept": "application/json",
            "User-Agent": "LeagueDataCollector/4.0 (Professional)"
        })
        self.total_requests = 0
        self.failed_requests = 0

    def get_json(self, url: str, endpoint_type: str) -> Optional[Dict]:
        """Universal request handler with retry logic"""
        for attempt in range(Config.MAX_RETRIES):
            self.rate_limiter.wait(endpoint_type)
            
            try:
                response = self.session.get(url, timeout=Config.REQUEST_TIMEOUT)
                self.total_requests += 1
                
                if response.status_code == 403:
                    logger.error(f"403 Forbidden - Check API key permissions for {url}")
                    return None
                elif response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 10))
                    logger.warning(f"Rate limited on {endpoint_type}. Waiting {retry_after}s")
                    time.sleep(retry_after)
                    continue
                elif response.status_code == 504:
                    logger.warning(f"Gateway timeout on attempt {attempt+1}, retrying...")
                    time.sleep(Config.RETRY_DELAYS[attempt])
                    continue
                    
                response.raise_for_status()
                
                # Process rate limit headers for real-time adjustment
                self._process_rate_headers(response.headers)
                
                return response.json()
            except requests.exceptions.RequestException as e:
                self.failed_requests += 1
                logger.warning(f"Attempt {attempt+1} failed for {url}: {str(e)}")
                time.sleep(Config.RETRY_DELAYS[attempt])
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error for {url}: {str(e)}")
                return None
                
        logger.error(f"Failed after {Config.MAX_RETRIES} attempts for {url}")
        return None

    def _process_rate_headers(self, headers: Dict) -> None:
        """Process rate limit headers for dynamic adjustment"""
        try:
            if "X-Method-Rate-Limit" in headers:
                limits = headers["X-Method-Rate-Limit"].split(":")
                counts = headers["X-Method-Rate-Limit-Count"].split(":")
                
                if len(limits) == 2 and len(counts) == 2:
                    limit = int(limits[1])
                    count = int(counts[1])
                    utilization = count / limit
                    
                    if utilization > 0.9:
                        logger.debug(f"High method utilization: {utilization:.0%}")
        except Exception as e:
            logger.debug(f"Error processing rate headers: {str(e)}")

    # API Endpoints
    def get_challenger_league(self, region: str) -> Optional[Dict]:
        url = f"{Config.BASE_URL.format(region=region)}/league/v4/challengerleagues/by-queue/RANKED_SOLO_5x5"
        return self.get_json(url, "league")

    def get_summoner(self, region: str, summoner_id: str) -> Optional[Dict]:
        url = f"{Config.BASE_URL.format(region=region)}/summoner/v4/summoners/{summoner_id}"
        return self.get_json(url, "league")

    def get_match_history(self, puuid: str, region: str) -> Optional[List[str]]:
        match_region = Config.MATCH_REGION_MAP[region]
        url = f"https://{match_region}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?queue=420&count={Config.MAX_MATCHES_PER_PLAYER}"
        return self.get_json(url, "match")

    def get_match_details(self, match_id: str, region: str) -> Optional[Dict]:
        match_region = Config.MATCH_REGION_MAP[region]
        url = f"https://{match_region}.api.riotgames.com/lol/match/v5/matches/{match_id}"
        return self.get_json(url, "match")

# ========================
# Data Models
# ========================
class ChampionStats:
    """Comprehensive champion statistics with matchup tracking"""
    def __init__(self):
        self.games = 0
        self.wins = 0
        self.kills = 0
        self.deaths = 0
        self.assists = 0
        self.matchups = defaultdict(lambda: {
            "games": 0,
            "wins": 0,
            "kills": 0,
            "deaths": 0,
            "assists": 0
        })

    def add_game(self, win: bool, kills: int, deaths: int, assists: int, 
                 opponent_champ: Optional[str] = None) -> None:
        """Record game stats with optional matchup data"""
        self.games += 1
        self.wins += int(win)
        self.kills += kills
        self.deaths += deaths
        self.assists += assists
        
        if opponent_champ:
            self.matchups[opponent_champ]["games"] += 1
            self.matchups[opponent_champ]["wins"] += int(win)
            self.matchups[opponent_champ]["kills"] += kills
            self.matchups[opponent_champ]["deaths"] += deaths
            self.matchups[opponent_champ]["assists"] += assists

    @property
    def win_rate(self) -> float:
        return (self.wins / self.games) * 100 if self.games else 0

    @property
    def kda(self) -> float:
        return (self.kills + self.assists) / max(1, self.deaths)

    def get_matchup_stats(self, min_games: int = 10) -> List[Dict]:
        """Get matchup statistics meeting minimum game threshold"""
        return [
            {
                "opponent": opponent,
                "games": data["games"],
                "win_rate": round((data["wins"] / data["games"]) * 100, 2),
                "kda": round((data["kills"] + data["assists"]) / max(1, data["deaths"]), 2),
                "avg_kills": round(data["kills"] / data["games"], 2),
                "avg_deaths": round(data["deaths"] / data["games"], 2),
                "avg_assists": round(data["assists"] / data["games"], 2)
            }
            for opponent, data in self.matchups.items()
            if data["games"] >= min_games
        ]

# ========================
# Processing Pipeline
# ========================
class DataProcessor:
    """Modular data processing with matchup support"""
    @staticmethod
    def process_player_matches(player_data: Dict, matches: List[Dict]) -> Dict[str, ChampionStats]:
        """Process matches for a single player with matchup tracking"""
        stats = defaultdict(ChampionStats)
        
        for match in filter(None, matches):
            participants = match.get("info", {}).get("participants", [])
            player = next((p for p in participants if p["puuid"] == player_data["puuid"]), None)
            
            if not player:
                continue
                
            champ = player["championName"]
            opponent = DataProcessor._get_lane_opponent(player, participants)
            
            stats[champ].add_game(
                win=player["win"],
                kills=player["kills"],
                deaths=player["deaths"],
                assists=player["assists"],
                opponent_champ=opponent
            )
                
        return stats

    @staticmethod
    def _get_lane_opponent(player: Dict, participants: List[Dict]) -> Optional[str]:
        """Identify lane opponent for matchup stats"""
        position = player.get("teamPosition", "")
        
        # Only consider actual lane opponents
        if position in ["TOP", "MID", "JUNGLE", "BOTTOM", "UTILITY"]:
            opponents = [
                p for p in participants 
                if p["teamId"] != player["teamId"] 
                and p.get("teamPosition", "") == position
            ]
            return opponents[0]["championName"] if opponents else None
        return None

    @staticmethod
    def save_matchup_data(champion: str, matchup_stats: List[Dict], region: str) -> None:
        """Save matchup data for a specific champion"""
        if not matchup_stats:
            return
            
        # Sort by worst to best matchups (for easier analysis)
        matchup_stats.sort(key=lambda x: x["win_rate"])
        
        df = pd.DataFrame(matchup_stats)
        filename = f"{champion.lower()}_matchups.csv"
        filepath = os.path.join(Config.MATCHUP_DIR, filename)
        
        df.to_csv(filepath, index=False)
        logger.debug(f"Saved matchup data for {champion} in {region}")

# ========================
# Orchestration Layer
# ========================
class LeagueScraper:
    """Main orchestrator with intelligent rate control"""
    def __init__(self):
        self.rate_limiter = RateLimiter()
        self.api = RiotAPI(self.rate_limiter)
        self.region_data = defaultdict(dict)
        self.start_time = time.time()
        self.processed_players = 0

    def run(self) -> None:
        """Execute full collection pipeline"""
        logger.info("🚀 Starting professional data collection")
        logger.info(f"📊 Target rate: {Config.RATE_LIMITS['app']['per_second']} req/s")
        
        try:
            with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
                # Process regions with controlled concurrency
                future_to_region = {
                    executor.submit(self.process_region, region): region
                    for region in Config.REGIONS
                }
                
                for future in as_completed(future_to_region):
                    region = future_to_region[future]
                    try:
                        self.region_data[region] = future.result()
                        elapsed = (time.time() - self.start_time) / 60
                        logger.info(f"✅ Completed {region.upper()} | Players: {self.processed_players} | Time: {elapsed:.1f}m")
                    except Exception as e:
                        logger.error(f"❌ Failed {region.upper()}: {str(e)}")
            
            # Save all collected data
            self.save_global_data()
            self.save_all_matchup_data()
            
            total_time = (time.time() - self.start_time) / 3600
            logger.info(f"✨ All regions processed in {total_time:.2f} hours")
            logger.info(f"📊 Total API requests: {self.api.total_requests}")
            logger.info(f"📈 Success rate: {(1 - (self.api.failed_requests / max(1, self.api.total_requests))) * 100:.1f}%")
            
        except KeyboardInterrupt:
            logger.info("🛑 Received interrupt. Saving collected data...")
            self.save_global_data()
            self.save_all_matchup_data()
        except Exception as e:
            logger.critical(f"💥 Critical error: {str(e)}", exc_info=True)
            raise

    def process_region(self, region: str) -> Dict[str, ChampionStats]:
        """Process all players in a single region"""
        region_stats = defaultdict(ChampionStats)
        
        # Get challenger ladder
        ladder = self.api.get_challenger_league(region)
        if not ladder:
            raise ValueError(f"Failed to get challenger league for {region}")
            
        logger.info(f"🏆 Processing {len(ladder['entries'])} players in {region.upper()}")
        
        # Process players with controlled concurrency
        with ThreadPoolExecutor(max_workers=min(Config.MAX_WORKERS, 4)) as executor:
            futures = []
            for idx, player in enumerate(ladder["entries"]):
                # Small delay between starting player processing
                if idx > 0 and idx % 5 == 0:
                    time.sleep(Config.REQUEST_DELAY)
                    
                futures.append(
                    executor.submit(
                        self.process_player,
                        region,
                        player["summonerId"],
                        player["leaguePoints"]
                    )
                )
            
            for future in as_completed(futures):
                try:
                    player_stats = future.result()
                    self.processed_players += 1
                    
                    # Aggregate player stats into region stats
                    for champ, stats in player_stats.items():
                        region_stats[champ].games += stats.games
                        region_stats[champ].wins += stats.wins
                        region_stats[champ].kills += stats.kills
                        region_stats[champ].deaths += stats.deaths
                        region_stats[champ].assists += stats.assists
                        
                        # Aggregate matchups
                        for opponent, opp_stats in stats.matchups.items():
                            region_stats[champ].matchups[opponent]["games"] += opp_stats["games"]
                            region_stats[champ].matchups[opponent]["wins"] += opp_stats["wins"]
                            region_stats[champ].matchups[opponent]["kills"] += opp_stats["kills"]
                            region_stats[champ].matchups[opponent]["deaths"] += opp_stats["deaths"]
                            region_stats[champ].matchups[opponent]["assists"] += opp_stats["assists"]
                            
                except Exception as e:
                    logger.warning(f"Player processing failed: {str(e)}")
        
        return region_stats

    def process_player(self, region: str, summoner_id: str, league_points: int) -> Dict[str, ChampionStats]:
        """Process a single player's match history"""
        logger.debug(f"Processing player {summoner_id[:6]}... (LP: {league_points})")
        
        # Get PUUID
        summoner = self.api.get_summoner(region, summoner_id)
        if not summoner or not summoner.get("puuid"):
            raise ValueError("Invalid summoner data")
        
        # Get match history
        match_ids = self.api.get_match_history(summoner["puuid"], region)
        if not match_ids:
            raise ValueError("No match history found")
        
        # Process matches with controlled concurrency
        match_details = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(self.api.get_match_details, match_id, region) 
                      for match_id in match_ids[:Config.MAX_MATCHES_PER_PLAYER]]
            
            for future in as_completed(futures):
                match = future.result()
                if match:
                    match_details.append(match)
                time.sleep(Config.REQUEST_DELAY)  # Small delay between match processing
        
        return DataProcessor.process_player_matches(
            {"puuid": summoner["puuid"], "summoner_id": summoner_id},
            match_details
        )

    def save_global_data(self) -> None:
        """Save combined champion data from all regions"""
        global_stats = []
        
        for region, stats in self.region_data.items():
            for champ, data in stats.items():
                if data.games >= Config.MIN_GAMES_THRESHOLD:
                    global_stats.append({
                        "champion": champ,
                        "region": region,
                        "games": data.games,
                        "wins": data.wins,
                        "win_rate": round(data.win_rate, 2),
                        "avg_kills": round(data.kills / data.games, 2),
                        "avg_deaths": round(data.deaths / data.games, 2),
                        "avg_assists": round(data.assists / data.games, 2),
                        "kda": round(data.kda, 2),
                        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
        
        # Save to CSV
        df = pd.DataFrame(global_stats)
        output_path = os.path.join(Config.OUTPUT_DIR, "global_stats.csv")
        df.to_csv(output_path, index=False)
        logger.info(f"💾 Saved global stats with {len(df)} entries to {output_path}")

    def save_all_matchup_data(self) -> None:
        """Save matchup data for all champions in all regions"""
        matchup_count = 0
        
        for region, stats in self.region_data.items():
            for champ, data in stats.items():
                if data.games >= Config.MIN_GAMES_THRESHOLD:
                    matchup_stats = data.get_matchup_stats(Config.MATCHUP_THRESHOLD)
                    if matchup_stats:
                        DataProcessor.save_matchup_data(f"{champ}_{region}", matchup_stats, region)
                        matchup_count += len(matchup_stats)
        
        logger.info(f"💾 Saved matchup data for {matchup_count} champion pairs")

# ========================
# Logging Configuration
# ========================
def configure_logging():
    """Set up comprehensive logging"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler
    fh = logging.FileHandler("scraper.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger

# ========================
# Main Execution
# ========================
if __name__ == "__main__":
    # Initialize logging
    logger = configure_logging()
    
    # Validate configuration
    Config.validate()
    
    # Run scraper
    try:
        scraper = LeagueScraper()
        scraper.run()
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}", exc_info=True)
        raise
    finally:
        logger.info("🏁 Script completed")