import pandas as pd
from datetime import datetime, timedelta
import random


def generate_stock_data():
    """Generate sample OHLCV stock market data"""
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    data = []
    
    base_prices = {
        'AAPL': 175.0,
        'GOOGL': 140.0,
        'MSFT': 380.0,
        'TSLA': 250.0,
        'AMZN': 185.0
    }
    
    base_date = datetime(2024, 1, 1)
    
    for symbol in symbols:
        price = base_prices[symbol]
        for i in range(60):  # 60 days of data
            date = base_date + timedelta(days=i)
            if date.weekday() >= 5:  # Skip weekends
                continue
            
            # Simulate price movement
            change_pct = random.uniform(-0.03, 0.03)
            open_price  = round(price, 2)
            close_price = round(price * (1 + change_pct), 2)
            high_price  = round(max(open_price, close_price) * random.uniform(1.001, 1.02), 2)
            low_price   = round(min(open_price, close_price) * random.uniform(0.98, 0.999), 2)
            volume      = random.randint(10_000_000, 100_000_000)
            
            data.append({
                'date':   date.strftime('%Y-%m-%d'),
                'symbol': symbol,
                'open':   open_price,
                'high':   high_price,
                'low':    low_price,
                'close':  close_price,
                'volume': volume
            })
            
            price = close_price  # Next day opens near previous close
    
    df = pd.DataFrame(data)
    df.to_csv('stock_data.csv', index=False)
    print(f"✅ Generated {len(df)} rows of stock data")
    print(df.head(10).to_string())
    return df
