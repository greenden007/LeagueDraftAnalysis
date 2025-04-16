
"""
Riot Matchup Aggregator
"""

import os
import glob
import logging
from datetime import datetime
from collections import defaultdict
import pandas as pd

# ========================
# Configuration
# ========================
class Config:
    """Centralized configuration for file paths"""
    INPUT_DIR = "soloq_stats/matchups"
    GLOBAL_STATS_FILE = "soloq_stats/global_stats.csv"
    OUTPUT_DIR = "aggregated_matchups"
    GLOBAL_STATS_OUTPUT = "aggregated_global_stats.csv"
    MATCHUP_THRESHOLD = 10
    
    @classmethod
    def validate(cls):
        """Create output directory if needed"""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)

# ========================
# Data Processing
# ========================
class MatchupAggregator:
    """Handles both matchup and global stats aggregation"""
    def __init__(self):
        self.matchup_data = defaultdict(list)
        self.global_stats = defaultdict(lambda: {
            'games': 0, 'wins': 0,
            'kills': 0.0, 'deaths': 0.0, 'assists': 0.0
        })
        self.processed_files = 0

    def run(self) -> None:
        """Main processing pipeline"""
        logger.info("🚀 Starting aggregation process")
        Config.validate()
        
        try:
            self.load_global_stats()
            self.load_matchup_data()
            self.save_aggregated_stats()
            self.save_global_stats()
            self.log_completion_stats()
        except Exception as e:
            logger.critical(f"💥 Aggregation failed: {str(e)}", exc_info=True)
            raise

    def load_global_stats(self) -> None:
        """Load and aggregate global stats CSV"""
        if not os.path.exists(Config.GLOBAL_STATS_FILE):
            logger.warning("⚠️ Global stats file not found")
            return

        try:
            df = pd.read_csv(Config.GLOBAL_STATS_FILE)
            for _, row in df.iterrows():
                champ = row['champion'].lower()
                self.global_stats[champ]['games'] += row['games']
                self.global_stats[champ]['wins'] += row['wins']
                self.global_stats[champ]['kills'] += row['avg_kills'] * row['games']
                self.global_stats[champ]['deaths'] += row['avg_deaths'] * row['games']
                self.global_stats[champ]['assists'] += row['avg_assists'] * row['games']
            
            self.processed_files += 1
            logger.info(f"🌍 Loaded global stats from {Config.GLOBAL_STATS_FILE}")
        except Exception as e:
            logger.error(f"❌ Failed to process global stats: {str(e)}")

    def load_matchup_data(self) -> None:
        """Process matchup CSV files"""
        file_pattern = os.path.join(Config.INPUT_DIR, "*_matchups.csv")
        
        for filepath in glob.glob(file_pattern):
            try:
                filename = os.path.basename(filepath)
                champion = filename.split('_')[0].lower()
                
                df = pd.read_csv(filepath)
                for _, row in df.iterrows():
                    self.matchup_data[champion].append({
                        "opponent": row["opponent"].lower(),
                        "games": row["games"],
                        "wins": int(row["games"] * row["win_rate"] / 100),
                        "kills": row["avg_kills"] * row["games"],
                        "deaths": row["avg_deaths"] * row["games"],
                        "assists": row["avg_assists"] * row["games"],
                    })
                
                self.processed_files += 1
                if self.processed_files % 50 == 0:
                    logger.info(f"📂 Processed {self.processed_files} files")
                    
            except Exception as e:
                logger.error(f"❌ Failed to process {filename}: {str(e)}")

    def save_aggregated_stats(self) -> None:
        """Save champion matchup stats"""
        total_matchups = 0
        
        for champion, matchups in self.matchup_data.items():
            combined = defaultdict(lambda: {
                "games": 0, "wins": 0,
                "kills": 0, "deaths": 0, "assists": 0
            })
            
            for m in matchups:
                opp = m["opponent"]
                combined[opp]["games"] += m["games"]
                combined[opp]["wins"] += m["wins"]
                combined[opp]["kills"] += m["kills"]
                combined[opp]["deaths"] += m["deaths"]
                combined[opp]["assists"] += m["assists"]
            
            matchup_stats = []
            for opponent, data in combined.items():
                if data["games"] >= Config.MATCHUP_THRESHOLD:
                    matchup_stats.append({
                        "opponent": opponent.title(),
                        "games": data["games"],
                        "win_rate": round((data["wins"] / data["games"]) * 100, 2),
                        "avg_kills": round(data["kills"] / data["games"], 2),
                        "avg_deaths": round(data["deaths"] / data["games"], 2),
                        "avg_assists": round(data["assists"] / data["games"], 2),
                        "kda": round((data["kills"] + data["assists"]) / max(1, data["deaths"]), 2),
                        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
            
            if matchup_stats:
                output_file = os.path.join(Config.OUTPUT_DIR, f"{champion}_matchups.csv")
                pd.DataFrame(matchup_stats).to_csv(output_file, index=False)
                total_matchups += len(matchup_stats)
        
        logger.info(f"💾 Saved {total_matchups} champion matchups")

    def save_global_stats(self) -> None:
        """Save aggregated global champion stats"""
        stats = []
        
        for champion, data in self.global_stats.items():
            if data['games'] > 0:
                stats.append({
                    'champion': champion.title(),
                    'games': data['games'],
                    'wins': data['wins'],
                    'win_rate': round((data['wins'] / data['games']) * 100, 2),
                    'avg_kills': round(data['kills'] / data['games'], 2),
                    'avg_deaths': round(data['deaths'] / data['games'], 2),
                    'avg_assists': round(data['assists'] / data['games'], 2),
                    'kda': round((data['kills'] + data['assists']) / max(1, data['deaths']), 2),
                    'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
        
        if stats:
            output_path = os.path.join(Config.OUTPUT_DIR, Config.GLOBAL_STATS_OUTPUT)
            pd.DataFrame(stats).to_csv(output_path, index=False)
            logger.info(f"🌐 Saved global stats for {len(stats)} champions")

    def log_completion_stats(self) -> None:
        """Log final processing statistics"""
        logger.info("\n✅ Aggregation Complete:")
        logger.info(f"Processed files: {self.processed_files}")
        logger.info(f"Champions with matchups: {len(self.matchup_data)}")
        logger.info(f"Champions with global stats: {len(self.global_stats)}")

# ========================
# Logging Setup
# ========================
def setup_logging():
    """Configure logging system"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

# ========================
# Main Execution
# ========================
if __name__ == "__main__":
    logger = setup_logging()
    try:
        aggregator = MatchupAggregator()
        aggregator.run()
    except Exception as e:
        logger.critical(f"💥 Fatal error: {str(e)}", exc_info=True)
