"""Bot manager for running multiple bots."""
import subprocess
import time
import logging
import signal
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BOT_SCRIPTS = [
    "bots/bots/bot.py",
]

processes = []


def signal_handler(sig, frame):
    """Handle shutdown signals."""
    logger.info("Shutting down bots...")
    for p in processes:
        try:
            p.terminate()
        except Exception as e:
            logger.error(f"Error terminating process: {e}")
    sys.exit(0)


def main():
    """Start all bots."""
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Get the bots directory
    bots_dir = Path(__file__).parent.parent
    
    logger.info(f"Starting {len(BOT_SCRIPTS)} bot(s)...")
    
    for script in BOT_SCRIPTS:
        script_path = bots_dir / script
        if not script_path.exists():
            logger.error(f"Bot script not found: {script_path}")
            continue
        
        logger.info(f"Starting bot: {script}")
        try:
            # Run bot script
            p = subprocess.Popen(
                [sys.executable, str(script_path)],
                cwd=str(bots_dir),
            )
            processes.append(p)
            logger.info(f"Bot {script} started (PID: {p.pid})")
            time.sleep(2)  # Stagger bot starts
        except Exception as e:
            logger.error(f"Error starting bot {script}: {e}")
    
    if not processes:
        logger.error("No bots started successfully")
        return
    
    logger.info(f"All bots started. Monitoring {len(processes)} process(es)...")
    
    # Wait for all processes
    try:
        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        signal_handler(None, None)


if __name__ == "__main__":
    main()
