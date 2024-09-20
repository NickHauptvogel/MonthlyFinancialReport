import os
import shutil
import tempfile
import datetime
import calendar
import pandas as pd
from abc import ABC, abstractmethod


class Statement(ABC):

    def __init__(self, data_dir, month, year, name):
        self.name = name
        self.file = None
        self.month = month
        self.year = year
        self.data_dir = data_dir
        self.data = None

    @abstractmethod
    def __read_statement(self):
        pass

    @staticmethod
    def convert_amount(row, name):
        if row[name] < 0:
            row["Type"] = "Expense"
            row[name] = -row[name]
        else:
            row["Type"] = "Income"
        return row


class LunarStatement(Statement):

    def __init__(self, data_dir, month, year, name):
        super().__init__(data_dir, month, year, name)
        for file in os.listdir(data_dir):
            # Format month and year into August 1, 2024 for example
            date_begin_str = datetime.datetime.strptime(f"{month:02d} 1, {year}", "%m %d, %Y").strftime("%B %d, %Y")
            date_begin_str = date_begin_str.replace(" 0", " ")
            # Get last day of month
            last_day = calendar.monthrange(year, month)[1]
            date_end_str = datetime.datetime.strptime(f"{month:02d} {last_day}, {year}", "%m %d, %Y").strftime("%B %d, %Y")
            date_end_str = date_end_str.replace(" 0", " ")
            # Check if file is in the format of month and year
            if date_begin_str in file and date_end_str in file:
                self.file = file
                print(f"Found statement for {date_begin_str} to {date_end_str}")
                break

        self.__read_statement()

    def __read_statement(self):
        if self.file:
            with open(f"{self.data_dir}/{self.file}", "r") as f:
                # Read from csv
                self.data = pd.read_csv(f, thousands='.', decimal=',')
                # Select columns
                self.data = self.data[["Date", "Text", "Amount"]]
                # Convert date to datetime
                self.data["Date"] = pd.to_datetime(self.data["Date"])
                # Rename and add columns
                self.data.columns = ["Date_Time", "Title", self.name]
                # Remove payer name from title (appears as " - Name" in Lunar)
                self.data["Title"] = self.data["Title"].apply(lambda x: x.split(" - ")[0])
                self.data = self.data.apply(Statement.convert_amount, axis=1, name=self.name)
                self.data["Payer/Payee"] = self.name
                self.data["Special Currency"] = "DKK"


class TradeRepublicStatement(Statement):

    def __init__(self, data_dir, month, year, name):
        super().__init__(data_dir, month, year, name)
        self.file = f"{year}-{month:02d}-account_transactions.csv"
        if not os.path.exists(os.path.join(data_dir, self.file)):
            # Get all documents from Trade Republic
            from pytr.dl import DL
            from pytr.account import login
            from pytr.portfolio import Portfolio
            import asyncio
            timestamp = datetime.datetime(year, month, 1).timestamp()
            with tempfile.TemporaryDirectory() as tmp_dir:
                dl = DL(
                    login(),
                    output_path=tmp_dir,
                    filename_fmt='{iso_date}{time} {title}{doc_num}',
                    since_timestamp=timestamp
                )
                try:
                    asyncio.get_event_loop().run_until_complete(dl.dl_loop())
                except SystemExit:
                    # Expected to exit
                    pass

                # Move account transactions to data_dir
                shutil.move(os.path.join(tmp_dir, "account_transactions.csv"),
                            os.path.join(data_dir, self.file))

                p = Portfolio(login())
                try:
                    p.get()
                except SystemExit:
                    # Expected to exit
                    pass
                p.portfolio_to_csv(os.path.join(data_dir, f"{year}-{month:02d}-portfolio.csv"))

        self.__read_statement()

    def __read_statement(self):
        if self.file:
            with open(f"{self.data_dir}/{self.file}", "r") as f:
                # Read from csv
                self.data = pd.read_csv(f, sep=";", thousands=',', decimal='.')
                # Select columns
                self.data = self.data[["CSVColumn_Date", "CSVColumn_Note", "CSVColumn_Value"]]
                # Convert date to datetime
                self.data["CSVColumn_Date"] = pd.to_datetime(self.data["CSVColumn_Date"])
                # Rename and add columns
                self.data.columns = ["Date_Time", "Title", self.name]
                # Filter and remove "card_successful_transaction -" from title
                self.data = self.data[self.data["Title"].str.contains("card_successful_transaction -")]
                self.data["Title"] = self.data["Title"].str.replace("card_successful_transaction - ", "")
                # Filter date
                self.data = self.data[self.data["Date_Time"].dt.month == self.month]
                self.data = self.data[self.data["Date_Time"].dt.year == self.year]

                self.data = self.data.apply(Statement.convert_amount, axis=1, name=self.name)
                self.data["Payer/Payee"] = self.name

