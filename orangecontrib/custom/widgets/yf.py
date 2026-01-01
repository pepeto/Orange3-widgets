import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

from AnyQt.QtWidgets import QFormLayout, QLineEdit
from orangewidget.utils.widgetpreview import WidgetPreview
from Orange.widgets import widget, gui, settings
from Orange.widgets.widget import Output
from Orange.data.pandas_compat import table_from_frame
from Orange.data import Table
from Orange.widgets.utils.concurrent import ThreadExecutor, FutureWatcher
import pandas as pd
import yfinance as yf


def download_full_yahoo_history(ticker: str) -> pd.DataFrame:
    # Fecha de hoy en formato YYYY-MM-DD
    #today = datetime.today().strftime("%Y-%m-%d")
    
    # Fecha de hoy
    today = datetime.today()

    # Fecha de mañana
    tomorrow = today + timedelta(days=1)

    # Formateamos como YYYY-MM-DD
    end_date = tomorrow.strftime("%Y-%m-%d")	
	
	
    df = yf.download(
        ticker,
        start="1900-01-01",
        end=end_date,
        auto_adjust=True,
        progress=False
    )
    if df.empty:
        return df
    return df


class OWCustYahooFinance(widget.OWWidget):
    name = "Yahoo Finance"
    description = "Download full historical stock data from Yahoo Finance."
    icon = "icons/StockData.svg"
    priority = 10
    category = "Custom"

    class Outputs:
        data = Output("Data", Table)

    ticker = settings.Setting("AAPL")
    want_main_area = False
    resizing_enabled = False

    class Error(widget.OWWidget.Error):
        download_error = widget.Msg(
            "Failed to download data.\n"
            "Check your internet connection or ticker symbol."
        )

    def __init__(self):
        super().__init__()
        layout = QFormLayout()
        gui.widgetBox(self.controlArea, True, orientation=layout)

        self.ticker_input = QLineEdit(self.ticker)
        self.ticker_input.editingFinished.connect(self._on_ticker_changed)
        layout.addRow("Ticker:", self.ticker_input)

        gui.button(self.controlArea, self, "Download", callback=self.download)
        
        self.executor = ThreadExecutor()
        self.task = None

    def _on_ticker_changed(self):
        self.ticker = self.ticker_input.text().strip().upper()

    def download(self):
        self.Error.clear()
        symbol = self.ticker.strip().upper()
        if not symbol:
            return

        # Cancel previous task
        if self.task:
            self.task.cancel()
            self.task = None

        self.Outputs.data.send(None)
        
        self.progressBarInit()
        self.setBlocking(True) # Optional: block inputs while downloading or handle gracefully
        
        # Start Task
        self.task = Task(symbol)
        self.task.watcher.finished.connect(self._on_download_finished)
        self.task.start()

    @gui.deferred
    def _on_download_finished(self):
        self.progressBarFinished()
        self.setBlocking(False)
        
        if not self.task:
            return
            
        try:
            df = self.task.future.result()
            if df is None or df.empty:
                raise ValueError("Empty data returned")
                
            table = self._dataframe_to_table(df)
            self.Outputs.data.send(table)
            
        except Exception as e:
            self.Error.download_error()
            print(f"Download error: {e}")
        
        self.task = None

    def _dataframe_to_table(self, df: pd.DataFrame) -> Table:
        # conversión automática
        return table_from_frame(df)


class Task:
    def __init__(self, ticker):
        self.ticker = ticker
        self.executor = ThreadExecutor()
        self.watcher = FutureWatcher()
        self.future = None
        
    def start(self):
        self.future = self.executor.submit(run_download, self.ticker)
        self.watcher.bind(self.future)
    
    def cancel(self):
        if self.future:
            self.future.cancel()

def run_download(ticker):
    # 1) bajar todo el histórico
    today = datetime.today()
    tomorrow = today + timedelta(days=1)
    end_date = tomorrow.strftime("%Y-%m-%d")    
    
    df = yf.download(
        ticker,
        start="1900-01-01",
        end=end_date,
        auto_adjust=True,
        progress=False
    )
    if df.empty:
        return df

    # 2) ordenar, resetear índice
    df = df.sort_index().reset_index()
    
    # Handle MultiIndex columns (Price, Ticker) -> just Price
    if isinstance(df.columns, pd.MultiIndex):
        # usually level 0 is Price (Open, Close...), level 1 is Ticker
        # We want level 0. 
        # But yfinance changed this recently. 
        # Safest is to flatten or drop.
        df.columns = df.columns.get_level_values(0)

    # 3) Normalize and filter robustly
    # Map all to lower case
    df.columns = [str(c).lower().strip().replace(" ", "_") for c in df.columns]
    
    # Select columns if they exist
    wanted = ["date", "open", "high", "low", "close", "volume", "adj_close"]
    # "date" might be "datetime" or "index" depending on reset_index()
    
    # Helper to find column regardless of exact name
    has_date = False
    for c in df.columns:
        if "date" in c or "time" in c:
            has_date = True
            break
            
    final_cols = [c for c in wanted if c in df.columns]
    
    # If no date found in cols, maybe it's still index (shouldn't be after reset_index)
    # Just return what we have
    if final_cols:
        return df[final_cols]
    else:
        return df # Fallback: return everything if matching fails


if __name__ == "__main__":
    WidgetPreview(OWCustYahooFinance).run()
