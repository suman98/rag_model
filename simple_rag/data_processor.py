import pandas as pd


class StockDataProcessor:
    """Process stock price and financial data and create text chunks for RAG"""

    def __init__(self, stock_prices_df: pd.DataFrame, financials_df: pd.DataFrame = None):
        self.stock_prices_df = stock_prices_df
        self.financials_df = financials_df
        self.chunks = []
        self.metadata = []

    def create_text_chunks(self):
        """Convert stock data rows into meaningful text chunks"""

        # Chunk Type 1: Daily Stock Price Summary
        for _, row in self.stock_prices_df.iterrows():
            daily_change = round(
                ((row['close'] - row['open']) / row['open']) * 100, 2
            )
            direction = "UP 📈" if daily_change > 0 else "DOWN 📉"

            chunk = (
                f"Stock: {row['symbol']} | Date: {row['date']} | "
                f"Open: ${row['open']} | High: ${row['high']} | "
                f"Low: ${row['low']} | Close: ${row['close']} | "
                f"Volume: {row['volume']:,} | "
                f"Daily Change: {daily_change}% ({direction}) | "
                f"Price Range: ${round(row['high'] - row['low'], 2)}"
            )
            self.chunks.append(chunk)
            self.metadata.append({
                'type':   'daily_price',
                'symbol': row['symbol'],
                'date':   row['date']
            })

        # Chunk Type 2: Stock Price Statistics per Symbol
        stock_prices_df_copy = self.stock_prices_df.copy()
        stock_prices_df_copy['date'] = pd.to_datetime(stock_prices_df_copy['date'])
        
        for symbol in stock_prices_df_copy['symbol'].unique():
            sym_df = stock_prices_df_copy[stock_prices_df_copy['symbol'] == symbol]
            if len(sym_df) == 0:
                continue
                
            total_change = round(
                ((sym_df['close'].iloc[-1] - sym_df['close'].iloc[0])
                 / sym_df['close'].iloc[0]) * 100, 2
            )
            chunk = (
                f"Price Stats | Stock: {symbol} | "
                f"Period: {sym_df['date'].min().date()} to {sym_df['date'].max().date()} | "
                f"Starting Price: ${sym_df['close'].iloc[0]} | "
                f"Latest Price: ${sym_df['close'].iloc[-1]} | "
                f"All-Time High: ${sym_df['high'].max()} | "
                f"All-Time Low: ${sym_df['low'].min()} | "
                f"Avg Daily Volume: {int(sym_df['volume'].mean()):,} | "
                f"Total Change: {total_change}% | "
                f"Trading Days: {len(sym_df)}"
            )
            self.chunks.append(chunk)
            self.metadata.append({
                'type':   'price_stats',
                'symbol': symbol
            })

        # Chunk Type 3: Financial Data Summary
        if self.financials_df is not None and len(self.financials_df) > 0:
            for _, row in self.financials_df.iterrows():
                chunk = (
                    f"Financial Data | Stock: {row['symbol']} | "
                    f"Quarter: {row['quarter']} | "
                    f"Fiscal Year: {row['fiscal_year']} | "
                    f"Financial Date: {row['financial_date']} | "
                    f"Value: {row['value']}"
                )
                self.chunks.append(chunk)
                self.metadata.append({
                    'type':   'financial',
                    'symbol': row['symbol'],
                    'quarter': str(row['quarter']),
                    'fiscal_year': str(row['fiscal_year'])
                })

        print(f"\n✅ Created {len(self.chunks)} text chunks")
        print(f"   - Daily Price Chunks: {sum(1 for m in self.metadata if m['type'] == 'daily_price')}")
        print(f"   - Price Stats Chunks: {sum(1 for m in self.metadata if m['type'] == 'price_stats')}")
        print(f"   - Financial Chunks: {sum(1 for m in self.metadata if m['type'] == 'financial')}")
        
        return self.chunks, self.metadata
