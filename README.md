# ML Bot

A machine learning-based stock trading bot for backtesting and live trading.

## Features
- Backtesting trading strategies using historical data
- Machine learning model integration (model.pkl)
- Customizable trading logic in `main.py`
- Data files for account, trades, settlements, and holidays

## Project Structure
- `main.py` - Main entry point for running the bot
- `backtest.py` - Script for backtesting trading strategies
- `model.pkl` - Pre-trained machine learning model
- `account.csv`, `trades.csv`, `settlement.csv` - Data files for account, trades, and settlements
- `dataset.csv` - Historical stock data
- `holidays.json` - List of market holidays
- `requirements.txt` - Python dependencies
- `.env` - Environment variables (not included in version control)

## Setup
1. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```
2. **Configure environment variables:**
   - Create a `.env` file with your API keys and settings as needed.

3. **Run backtest:**
   ```powershell
   python backtest.py
   ```

4. **Run the bot:**
   ```powershell
   python main.py
   ```

## Notes
- Ensure all required data files are present in the project directory.
- The bot uses a pre-trained model (`model.pkl`). Retrain or update as needed.

## License
See `LICENSE` for details.
