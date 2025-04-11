#!/usr/bin/env python3
"""
Ultimate Riot API Scraper - Fixed ID Edition
- Corrected summoner ID usage
- Proper PUUID handling
- Enhanced validation
- Detailed error tracking
"""

import os
import time
import logging
from datetime import datetime
from collections import defaultdict, deque
import pandas as pd
import requests
import json
from typing import Dict, List, Optional, Tuple

# ========================
# Configuration
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
    
    # Rate Limits
    RATE_LIMITS = {
        "app": {
            "per_two_minutes": 95,
            "window_seconds": 120
        },
        "endpoints": {
            "league": {
                "per_two_minutes": 85,
                "window_seconds": 120
            },
            "match": {
                "per_two_minutes": 85,
                "window_seconds": 120
            },
            "summoner": {
                "per_two_minutes": 85,
                "window_seconds": 120
            }
        }
    }
    
    # Collection Parameters
    MAX_MATCHES_PER_PLAYER = 100
    MIN_GAMES_THRESHOLD = 3
    MATCHUP_THRESHOLD = 10
    
    # Output Directories
    OUTPUT_DIR = "soloq_stats"
    MATCHUP_DIR = os.path.join(OUTPUT_DIR, "matchups")
    
    # Request Parameters
    REQUEST_TIMEOUT = 15
    MAX_RETRIES = 3
    RETRY_DELAYS = [1, 2, 3]
    PROGRESS_LOG_INTERVAL = 10

    @classmethod
    def validate(cls):
        """Validate configuration and create directories"""
        if not cls.API_KEY:
            raise ValueError("RIOT_API_KEY environment variable not set")
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.MATCHUP_DIR, exist_ok=True)

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

    def add_game(self, win: bool, kills: int, deaths: int, assists: int, opponent_champ: Optional[str] = None) -> None:
        """Record game stats with matchup data"""
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

    def get_matchup_stats(self) -> List[Dict]:
        """Get matchup statistics meeting threshold"""
        return [
            {
                "opponent": opponent,
                "games": data["games"],
                "win_rate": round((data["wins"] / data["games"]) * 100, 2),
                "kda": round((data["kills"] + data["assists"]) / max(1, data["deaths"]), 2),
                "avg_kills": round(data["kills"] / data["games"], 2),
                "avg_deaths": round(data["deaths"] / data["games"], 2),
                "avg_assists": round(data["assists"] / data["games"], 2),
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            for opponent, data in self.matchups.items()
            if data["games"] >= Config.MATCHUP_THRESHOLD
        ]

# ========================
# Rate Limiter
# ========================
class PrecisionRateLimiter:
    """Maintains exact 95% utilization through precise request pacing"""
    def __init__(self):
        self.request_times = {
            "app": deque(maxlen=Config.RATE_LIMITS["app"]["per_two_minutes"]),
            "league": deque(maxlen=Config.RATE_LIMITS["endpoints"]["league"]["per_two_minutes"]),
            "match": deque(maxlen=Config.RATE_LIMITS["endpoints"]["match"]["per_two_minutes"]),
            "summoner": deque(maxlen=Config.RATE_LIMITS["endpoints"]["summoner"]["per_two_minutes"])
        }
        self.last_request_time = 0
        self.total_requests = 0

    def wait(self, endpoint_type: str) -> None:
        """Ensure perfect request pacing for 95% utilization"""
        now = time.time()
        self.total_requests += 1
        
        # Calculate delays for all limits
        delays = []
        for limit_type in ["app", endpoint_type]:
            queue = self.request_times[limit_type]
            max_requests = Config.RATE_LIMITS["app"]["per_two_minutes"] if limit_type == "app" else Config.RATE_LIMITS["endpoints"][endpoint_type]["per_two_minutes"]
            window = Config.RATE_LIMITS["app"]["window_seconds"] if limit_type == "app" else Config.RATE_LIMITS["endpoints"][endpoint_type]["window_seconds"]
            
            if len(queue) >= max_requests:
                elapsed = now - queue[0]
                if elapsed < window:
                    delays.append((queue[0] + window) - now)
        
        # Apply the longest required delay
        if delays:
            time.sleep(max(delays))
        
        # Record the request
        current_time = time.time()
        self.request_times["app"].append(current_time)
        self.request_times[endpoint_type].append(current_time)
        self.last_request_time = current_time

# ========================
# API Client
# ========================
class RiotAPI:
    """Fixed API client with proper ID handling"""
    def __init__(self, rate_limiter: PrecisionRateLimiter):
        self.rate_limiter = rate_limiter
        self.session = requests.Session()
        self.session.headers.update({
            "X-Riot-Token": Config.API_KEY,
            "Accept": "application/json",
            "User-Agent": "LeagueDataCollector/10.0 (FixedIDs)"
        })
        self.total_requests = 0
        self.failed_requests = 0

    def get_json(self, url: str, endpoint_type: str) -> Optional[Dict]:
        """Make API request with retry logic"""
        for attempt in range(Config.MAX_RETRIES):
            self.rate_limiter.wait(endpoint_type)
            
            try:
                response = self.session.get(url, timeout=Config.REQUEST_TIMEOUT)
                self.total_requests += 1
                
                if response.status_code == 404:
                    return None  # Summoner not found
                elif response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 10))
                    logger.warning(f"Rate limited. Waiting {retry_after}s")
                    time.sleep(retry_after)
                    continue
                    
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                self.failed_requests += 1
                logger.debug(f"Attempt {attempt+1} failed: {str(e)}")
                time.sleep(Config.RETRY_DELAYS[attempt])
                
        logger.error(f"Failed after {Config.MAX_RETRIES} attempts for {url}")
        return None

    def get_challenger_league(self, region: str) -> Optional[Dict]:
        url = f"{Config.BASE_URL.format(region=region)}/league/v4/challengerleagues/by-queue/RANKED_SOLO_5x5"
        return self.get_json(url, "league")

    def get_summoner_by_id(self, region: str, summoner_id: str) -> Optional[Dict]:
        """Get summoner by summonerId (v4 API)"""
        url = f"{Config.BASE_URL.format(region=region)}/summoner/v4/summoners/{summoner_id}"
        return self.get_json(url, "summoner")

    def get_summoner_by_puuid(self, region: str, puuid: str) -> Optional[Dict]:
        """Get summoner by puuid (v4 API)"""
        url = f"{Config.BASE_URL.format(region=region)}/summoner/v4/summoners/by-puuid/{puuid}"
        return self.get_json(url, "summoner")

    def get_match_history(self, puuid: str, region: str) -> Optional[List[str]]:
        match_region = Config.MATCH_REGION_MAP[region]
        url = f"https://{match_region}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?queue=420&count={Config.MAX_MATCHES_PER_PLAYER}"
        return self.get_json(url, "match")

    def get_match_details(self, match_id: str, region: str) -> Optional[Dict]:
        match_region = Config.MATCH_REGION_MAP[region]
        url = f"https://{match_region}.api.riotgames.com/lol/match/v5/matches/{match_id}"
        return self.get_json(url, "match")

# ========================
# Main Scraper
# ========================
class LeagueScraper:
    """Main scraper with fixed ID handling"""
    def __init__(self):
        self.rate_limiter = PrecisionRateLimiter()
        self.api = RiotAPI(self.rate_limiter)
        self.region_data = defaultdict(dict)
        self.start_time = time.time()
        self.processed_players = 0
        self.skipped_players = 0
        self.failed_summoner_lookups = 0

    def run(self) -> None:
        """Execute the scraping process"""
        logger.info("🚀 Starting data collection with fixed ID handling")
        
        try:
            for region in Config.REGIONS:
                self.process_region(region)
            
            self.save_data()
            
        except KeyboardInterrupt:
            logger.info("🛑 Received interrupt. Saving data...")
            self.save_data()
        except Exception as e:
            logger.critical(f"💥 Fatal error: {str(e)}", exc_info=True)
            raise
        finally:
            self.log_final_stats()

    def process_region(self, region: str) -> None:
        """Process a single region with proper ID handling"""
        logger.info(f"🏆 Processing {region.upper()}")
        
        ladder = self.api.get_challenger_league(region)
        if not ladder or not isinstance(ladder, dict) or "entries" not in ladder:
            logger.error(f"Invalid ladder data for {region}")
            return
            
        players = ladder["entries"]
        logger.info(f"Found {len(players)} players in {region.upper()}")
        
        for i, player in enumerate(players):
            if i % Config.PROGRESS_LOG_INTERVAL == 0:
                logger.info(f"⏳ Processing player {i+1}/{len(players)} in {region.upper()}")
            
            result = self.process_player(region, player["summonerId"])
            if result[0]:
                self.processed_players += 1
            else:
                self.skipped_players += 1
                if "summoner not found" in result[1].lower():
                    self.failed_summoner_lookups += 1
            
            time.sleep(0.1)

    def process_player(self, region: str, summoner_id: str) -> Tuple[bool, str]:
        """Process a single player with detailed status reporting"""
        # Step 1: Get summoner by summonerId
        summoner = self.api.get_summoner_by_id(region, summoner_id)
        if not summoner:
            return (False, f"Summoner not found by ID {summoner_id[:6]} (may have changed name)")
        
        # Step 2: Verify we have PUUID
        puuid = summoner.get("puuid")
        if not puuid:
            return (False, f"Summoner {summoner_id[:6]} has no PUUID")
        
        # Step 3: Get match history by PUUID
        match_ids = self.api.get_match_history(puuid, region)
        if not match_ids:
            return (False, f"No match history for {summoner_id[:6]} (PUUID: {puuid[:8]}...)")
        
        # Step 4: Process matches
        valid_matches = []
        for match_id in match_ids[:Config.MAX_MATCHES_PER_PLAYER]:
            if not self.validate_match_id(match_id, region):
                logger.debug(f"Invalid match ID format: {match_id[:12]}...")
                continue
                
            match = self.api.get_match_details(match_id, region)
            if match and self.validate_match_data(match, puuid):
                valid_matches.append(match)
        
        if not valid_matches:
            return (False, f"No valid matches for {summoner_id[:6]} (PUUID: {puuid[:8]}...)")
        
        # Step 5: Process stats
        try:
            stats = self.process_matches(summoner, valid_matches)
            self.aggregate_stats(region, stats)
            return (True, f"Processed {len(valid_matches)} matches")
        except Exception as e:
            return (False, f"Processing error: {str(e)}")

    def validate_match_id(self, match_id: str, region: str) -> bool:
        """Validate match ID structure and region"""
        try:
            parts = match_id.split('_')
            return (len(parts) == 2 
                    and parts[0] == Config.MATCH_REGION_MAP[region].upper()
                    and parts[1].isdigit()
                    and len(parts[1]) == 10)
        except Exception:
            return False

    def validate_match_data(self, match: Dict, puuid: str) -> bool:
        """Validate match contains the player"""
        try:
            return any(p["puuid"] == puuid for p in match["info"]["participants"])
        except Exception:
            return False

    def process_matches(self, summoner: Dict, matches: List[Dict]) -> Dict[str, ChampionStats]:
        """Process validated matches into stats"""
        stats = defaultdict(ChampionStats)
        puuid = summoner["puuid"]
        
        for match in matches:
            try:
                participants = match["info"]["participants"]
                player = next(p for p in participants if p["puuid"] == puuid)
                opponent = self.get_lane_opponent(player, participants)
                
                stats[player["championName"]].add_game(
                    win=player["win"],
                    kills=player["kills"],
                    deaths=player["deaths"],
                    assists=player["assists"],
                    opponent_champ=opponent
                )
            except Exception as e:
                logger.debug(f"Match processing error: {str(e)}")
                continue
                
        return stats

    def get_lane_opponent(self, player: Dict, participants: List[Dict]) -> Optional[str]:
        """Identify lane opponent if possible"""
        try:
            position = player["teamPosition"]
            if position in ["TOP", "MID", "JUNGLE", "BOTTOM", "UTILITY"]:
                opponents = [p for p in participants 
                            if p["teamId"] != player["teamId"] 
                            and p.get("teamPosition") == position]
                return opponents[0]["championName"] if opponents else None
        except Exception:
            return None
        return None

    def save_data(self) -> None:
        """Save all collected data"""
        self.save_global_stats()
        self.save_matchup_stats()

    def save_global_stats(self) -> None:
        """Save global champion statistics"""
        global_stats = []
        
        for region, stats in self.region_data.items():
            for champ, champ_stats in stats.items():
                if champ_stats.games >= Config.MIN_GAMES_THRESHOLD:
                    global_stats.append({
                        "champion": champ,
                        "region": region,
                        "games": champ_stats.games,
                        "wins": champ_stats.wins,
                        "win_rate": round(champ_stats.win_rate, 2),
                        "avg_kills": round(champ_stats.kills / champ_stats.games, 2),
                        "avg_deaths": round(champ_stats.deaths / champ_stats.games, 2),
                        "avg_assists": round(champ_stats.assists / champ_stats.games, 2),
                        "kda": round(champ_stats.kda, 2),
                        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
        
        df = pd.DataFrame(global_stats)
        output_path = os.path.join(Config.OUTPUT_DIR, "global_stats.csv")
        df.to_csv(output_path, index=False)
        logger.info(f"💾 Saved global stats with {len(df)} entries")

    def save_matchup_stats(self) -> None:
        """Save per-champion matchup statistics"""
        matchup_count = 0
        
        for region, stats in self.region_data.items():
            for champ, champ_stats in stats.items():
                if champ_stats.games >= Config.MIN_GAMES_THRESHOLD:
                    matchups = champ_stats.get_matchup_stats()
                    if matchups:
                        filename = f"{champ.lower()}_{region}_matchups.csv"
                        filepath = os.path.join(Config.MATCHUP_DIR, filename)
                        
                        matchups.sort(key=lambda x: x["win_rate"])
                        pd.DataFrame(matchups).to_csv(filepath, index=False)
                        matchup_count += len(matchups)
        
        logger.info(f"💾 Saved matchup data for {matchup_count} champion pairs")

    def log_final_stats(self) -> None:
        """Log detailed final statistics"""
        total_time = (time.time() - self.start_time) / 60
        logger.info("\n📊 Final Statistics:")
        logger.info(f"⏱️  Total runtime: {total_time:.1f} minutes")
        logger.info(f"✅ Players processed: {self.processed_players}")
        logger.info(f"⚠️  Players skipped: {self.skipped_players}")
        logger.info(f"🔍 Failed summoner lookups: {self.failed_summoner_lookups}")
        logger.info(f"📡 API requests: {self.api.total_requests}")
        logger.info(f"❌ Failed requests: {self.api.failed_requests}")
        logger.info(f"🏁 Script completed")

# ========================
# Logging Setup
# ========================
def setup_logging():
    """Configure comprehensive logging"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    fh = logging.FileHandler("scraper.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger

# ========================
# Main Execution
# ========================
if __name__ == "__main__":
    logger = setup_logging()
    Config.validate()
    
    try:
        scraper = LeagueScraper()
        scraper.run()
    except Exception as e:
        logger.critical(f"💥 Fatal error: {str(e)}", exc_info=True)
        raise