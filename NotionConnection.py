import copy
import re
import requests
import os
from datetime import datetime
import pandas as pd
import logging
from currency_converter import CurrencyConverter

from statements import LunarStatement, TradeRepublicStatement


def get_previous_month(month, year):
    if month == 1:
        return 12, year - 1
    else:
        return month - 1, year

def get_next_month(month, year):
    if month == 12:
        return 1, year + 1
    else:
        return month + 1, year


def increase_date_in_title(row, offset):
    m_y = re.search(r'\d{2}/\d{2}', row['Title'])
    if m_y is not None:
        new_date = datetime.strptime(m_y.group(), '%m/%y')
        new_date = new_date + offset
        row['Title'] = re.sub(r'\d{2}/\d{2}', new_date.strftime('%m/%y'), row['Title'])
    return row


class NotionConnection:
    def __init__(self, token, database_entries_id, database_networth_id):
        self.token = token
        self.database_entries_id = database_entries_id
        self.database_networth_id = database_networth_id
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "Notion-Version": "2022-06-28"
        }
        self.url = "https://api.notion.com/v1/"
        self.all_entries = None
        self.networth = None

    def query_database(self, id, start_cursor=None):
        payload = {
            "page_size": 100
        }
        if start_cursor is not None:
            payload["start_cursor"] = start_cursor
        response = requests.post(
            self.url + f"databases/{id}/query",
            json=payload,
            headers=self.headers
        )
        return response.json()

    def create_database_entry(self, id, properties):
        payload = {
            "parent": {"database_id": id},
            "properties": properties
        }
        response = requests.post(
            self.url + "pages",
            json=payload,
            headers=self.headers
        )
        return response.json()

    def patch_page(self, page_id, properties):
        payload = {
            "properties": properties
        }
        response = requests.patch(
            self.url + f"pages/{page_id}",
            json=payload,
            headers=self.headers
        )
        return response.json()

    def delete_page(self, page_id):
        response = requests.patch(
            self.url + f"pages/{page_id}",
            json={"archived": True},
            headers=self.headers
        )
        return response.json()

    def create_database_entries_from_df(self, df, df_to_check_for_duplicates=None):
        # If a dataframe is provided, check for duplicates
        if df_to_check_for_duplicates is not None:
            # Inner merge, disregard Amount columns Tabea and Nick
            duplicates = pd.merge(df, df_to_check_for_duplicates, how='inner', indicator=True, on=["Title", "Category", "Payer/Payee", "Date_Time"])
            if not duplicates.empty:
                logging.warning(f"Found {len(duplicates)} duplicates already in the database")
                logging.warning(duplicates[["Title"]].to_string())
                df = df[~df.index.isin(duplicates.index)].reset_index(drop=True)

        total = len(df)
        logging.info(f"Creating {total} new entries in the database")
        for i, row in df.iterrows():
            properties = {
                "Type": {"select": {"name": row["Type"]}},
                "Title": {"title": [{"text": {"content": row["Title"]}}]},
                "Tabea": {"number": row["Tabea"]},
                "Nick": {"number": row["Nick"]},
                "Category": {"select": {"name": row["Category"]}},
                "Payer/Payee": {"select": {"name": row["Payer/Payee"]}},
                "Date": {"date": {"start": row["Date_Time"].isoformat()}}
            }
            if row["Reoccurrence"] is not None:
                properties["Reoccurrence"] = {"select": {"name": row["Reoccurrence"]}}
            if row["Special Currency"] is not None:
                properties["Special Currency"] = {"select": {"name": row["Special Currency"]}}
            if "Receipt" in row and row["Receipt"] is not None and row["Receipt"] != '' and pd.notna(row["Receipt"]):
                properties["Receipt"] = {"files": [{"name": "receipt", "external": {"url": row["Attachment URL"]}}]}
            if row["Special Tag"] is not None and len(row["Special Tag"]) > 0:
                properties["Special Tag"] = {"multi_select": [{"name": tag} for tag in row["Special Tag"]]}

            self.create_database_entry(self.database_entries_id, properties)
            print(f"Entry created ({i+1}/{total})")

    def get_all_entries(self, month=None, year=None):
        # Read the response into a pandas dataframe
        logging.info("Reading all entries from the database")
        entries = []
        start_cursor = None
        batch_nr = 1
        while True:
            response = self.query_database(self.database_entries_id, start_cursor)
            logging.info(f"Reading batch {batch_nr}")
            for raw_entry in response["results"]:
                if len(raw_entry["properties"]["Title"]["title"]) == 0:
                    print("Found an entry without a title, skipping it. Created: ", raw_entry["created_time"])
                    continue
                if raw_entry["properties"]["Title"]["title"][0]["plain_text"] == "Expense":
                    print("Found an entry with the title 'Expense', skipping it (probably from template). Created: ", raw_entry["created_time"])
                    continue
                entry = {
                    "Type": raw_entry["properties"]["Type"]["select"]["name"],
                    "Title": raw_entry["properties"]["Title"]["title"][0]["plain_text"],
                    "Tabea": raw_entry["properties"]["Tabea"]["number"],
                    "Nick": raw_entry["properties"]["Nick"]["number"],
                    "Category": raw_entry["properties"]["Category"]["select"]["name"],
                    "Payer/Payee": raw_entry["properties"]["Payer/Payee"]["select"]["name"],
                    "Month": raw_entry["properties"]["Month"]["formula"]["number"],
                    "Year": raw_entry["properties"]["Year"]["formula"]["number"],
                    "Total": raw_entry["properties"]["Total"]["formula"]["number"],
                    "Date_Time": datetime.fromisoformat(raw_entry["properties"]["Date"]["date"]["start"]),
                    "Reoccurrence": raw_entry["properties"]["Reoccurrence"]["select"]["name"] if raw_entry["properties"]["Reoccurrence"]["select"] is not None else None,
                    "Special Currency": raw_entry["properties"]["Special Currency"]["select"]["name"] if raw_entry["properties"]["Special Currency"]["select"] is not None else None,
                    "Page_ID": raw_entry["id"]
                }
                if len(raw_entry["properties"]["Receipt"]["files"]) > 0:
                    if "external" in raw_entry["properties"]["Receipt"]["files"][0]:
                        entry["Receipt"] = raw_entry["properties"]["Receipt"]["files"][0]["external"]["url"]
                    elif "file" in raw_entry["properties"]["Receipt"]["files"][0]:
                        entry["Receipt"] = raw_entry["properties"]["Receipt"]["files"][0]["file"]["url"]
                    else:
                        entry["Receipt"] = None
                if len(raw_entry["properties"]["Special Tag"]["multi_select"]) > 0:
                    entry["Special Tag"] = [tag["name"] for tag in raw_entry["properties"]["Special Tag"]["multi_select"]]
                else:
                    entry["Special Tag"] = None
                entries.append(entry)
            if not response["has_more"]:
                break
            start_cursor = response["next_cursor"]
            batch_nr += 1

        existing_entries = pd.DataFrame(entries)
        existing_entries['Date_Time'] = pd.to_datetime(existing_entries['Date_Time'], utc=True)
        existing_entries['Date_Time'] = existing_entries['Date_Time'].apply(lambda x: x.replace(tzinfo=None))

        # Filter to only entries from the current year and possibly month
        if year is not None:
            logging.info(f"Filtering entries for year {year}")
            existing_entries = existing_entries[existing_entries['Date_Time'].dt.year == year]
            if month is not None:
                logging.info(f"Filtering entries for month {month}")
                existing_entries = existing_entries[existing_entries['Date_Time'].dt.month == month]

        logging.info(f"Found {len(existing_entries)} entries")
        self.all_entries = existing_entries

    def get_networth(self, month=None, year=None):
        # Read the response into a pandas dataframe
        logging.info("Reading networth from the database")
        entries = []
        start_cursor = None
        batch_nr = 1
        while True:
            response = self.query_database(self.database_networth_id, start_cursor)
            logging.info(f"Reading batch {batch_nr}")
            for raw_entry in response["results"]:
                entry = {
                    "Position": raw_entry["properties"]["Position"]["title"][0]["plain_text"],
                    "Value": raw_entry["properties"]["Value"]["number"],
                    "Category": raw_entry["properties"]["Category"]["select"]["name"],
                    "Person": raw_entry["properties"]["Person"]["select"]["name"],
                    "Date_Time": datetime.fromisoformat(raw_entry["properties"]["Date"]["date"]["start"]),
                    "Month": raw_entry["properties"]["Month"]["formula"]["number"],
                    "Year": raw_entry["properties"]["Year"]["formula"]["number"],
                    "Status": raw_entry["properties"]["Status"]["select"]["name"],
                    "Page_ID": raw_entry["id"]
                }
                entries.append(entry)
            if not response["has_more"]:
                break
            start_cursor = response["next_cursor"]
            batch_nr += 1

        networth_entries = pd.DataFrame(entries)
        networth_entries['Date_Time'] = pd.to_datetime(networth_entries['Date_Time'], utc=True)
        networth_entries['Date_Time'] = networth_entries['Date_Time'].apply(lambda x: x.replace(tzinfo=None))

        # Filter to only entries from the current year and possibly month
        if year is not None:
            logging.info(f"Filtering entries for year {year}")
            networth_entries = networth_entries[networth_entries['Date_Time'].dt.year == year]
            if month is not None:
                logging.info(f"Filtering entries for month {month}")
                networth_entries = networth_entries[networth_entries['Date_Time'].dt.month == month]

        logging.info(f"Found {len(networth_entries)} networth entries")
        self.networth = networth_entries

    def convert_special_currencies_to_euro(self):
        # Get all entries with special currencies
        special_currencies = self.all_entries[self.all_entries['Special Currency'].notnull()]
        c = CurrencyConverter(fallback_on_wrong_date=True)

        row_counter = 1
        for _, row in special_currencies.iterrows():
            old_currency = row["Special Currency"]
            properties = {
                "Tabea": {"number": round(c.convert(row['Tabea'], row['Special Currency'], 'EUR', date=row['Date_Time']), 2)},
                "Nick": {"number": round(c.convert(row['Nick'], row['Special Currency'], 'EUR', date=row['Date_Time']), 2)},
                "Special Currency": {"select": None}
            }
            self.patch_page(row["Page_ID"], properties)
            print(f"Converted special currency {old_currency} to EUR ({row_counter}/{len(special_currencies)})")
            row_counter += 1

    def convert_special_tags(self):
        # SPECIAL TAG: "Yearly Subscription"
        # Get all entries with special tag "Yearly Subscription"
        yearly_subscriptions = self.all_entries.loc[self.all_entries['Special Tag'].apply(lambda x: 'Yearly Subscription' in x if x is not None else False)]
        insert_rows = []
        for _, row in yearly_subscriptions.iterrows():
            # Divide the amount by 12
            monthly_amount_nick = round(row['Nick'] / 12, 2)
            monthly_amount_tabea = round(row['Tabea'] / 12, 2)
            # If current title does not contain MM/YY, add it
            m_y = re.search(r'\d{2}/\d{2}', row['Title'])
            if m_y is None:
                row['Title'] = row['Title'] + f" {row['Date_Time'].strftime('%m/%y')}"
            # Create 12 new entries
            for i in range(12):
                new_row = copy.deepcopy(row)
                new_row['Nick'] = monthly_amount_nick
                new_row['Tabea'] = monthly_amount_tabea
                new_row['Date_Time'] = row['Date_Time'] + pd.DateOffset(months=i)
                new_row['Month'] = new_row['Date_Time'].month
                new_row['Year'] = new_row['Date_Time'].year
                new_row['Special Tag'] = [tag for tag in row['Special Tag'] if tag != 'Yearly Subscription']
                new_row['Total'] = monthly_amount_nick + monthly_amount_tabea

                new_row = increase_date_in_title(new_row, pd.DateOffset(months=i))
                insert_rows.append(new_row)

        insert_df = pd.DataFrame(insert_rows)
        self.create_database_entries_from_df(insert_df, self.all_entries)

        for _, row in yearly_subscriptions.iterrows():
            result = self.delete_page(row["Page_ID"])
            print(f"Deleted yearly subscription entry {row['Title']} with result {result}")

    def insert_reoccurring_expenses(self, insert_month, insert_year):
        last_month, last_year = get_previous_month(insert_month, insert_year)

        last_month_data = self.all_entries[(self.all_entries['Date_Time'].dt.month == last_month) & (self.all_entries['Date_Time'].dt.year == last_year)]
        # Get monthly entries from last month
        monthly_occurrences = copy.deepcopy(last_month_data[last_month_data['Reoccurrence'] == 'Monthly'])
        monthly_occurrences = monthly_occurrences.reset_index(drop=True)
        # Set Tabea and Nick to 0
        monthly_occurrences['Tabea'] = 0
        monthly_occurrences['Nick'] = 0
        # Increase the date by one month
        monthly_occurrences['Date_Time'] = monthly_occurrences['Date_Time'] + pd.DateOffset(months=1)

        # Increase date in Title. Search for format MM/YY (could be any month, e.g. for paid in advance)
        monthly_occurrences = monthly_occurrences.apply(increase_date_in_title, axis=1, offset=pd.DateOffset(months=1))
        self.create_database_entries_from_df(monthly_occurrences, self.all_entries)

        # Get yearly entries from last year
        yearly_occurrences = copy.deepcopy(self.all_entries[(self.all_entries['Date_Time'].dt.month == insert_month) & (self.all_entries['Date_Time'].dt.year == insert_year - 1)])
        yearly_occurrences = yearly_occurrences.reset_index(drop=True)
        # Set Tabea and Nick to 0
        yearly_occurrences['Tabea'] = 0
        yearly_occurrences['Nick'] = 0
        # Increase the date by one year
        yearly_occurrences['Date_Time'] = yearly_occurrences['Date_Time'] + pd.DateOffset(years=1)
        # Increase date in Title. Search for format MM/YY (could be any month, e.g. for paid in advance)
        yearly_occurrences = yearly_occurrences.apply(increase_date_in_title, axis=1, offset=pd.DateOffset(years=1))
        self.create_database_entries_from_df(yearly_occurrences, self.all_entries)

    def prefill_account_statements(self, month, year, data_dir):
        for name in ['Tabea', 'Nick']:
            # Look in data_dir/statements/name
            statements_folder = os.path.join(data_dir, "statements", name)
            for folder in os.listdir(statements_folder):
                if folder == "lunar":
                    statement = LunarStatement(os.path.join(statements_folder, "lunar"), month, year, name)
                elif folder == "traderepublic":
                    statement = TradeRepublicStatement(os.path.join(statements_folder, "traderepublic"), month, year, name)
                else:
                    print(f"Automatic statement for folder {folder} not supported!")
                    statement = None

                if statement is not None and statement.data is not None:
                    # Create new entries in the database
                    if self.all_entries is None:
                        print("Reading all entries from the database, as they are not yet loaded")
                        self.get_all_entries(month, year)
                    self.create_database_entries_from_df(statement.data, self.all_entries)


if __name__ == '__main__':
    month = 9
    year = 2024

    last_month, last_year = get_previous_month(month, year)

    data_dir = "/mnt/c/Users/NHaup/OneDrive/Dokumente/Persönlich/Buntentor/02_Financials/01_monthly/data"

    logging.basicConfig(level=logging.INFO)

    notion_ = NotionConnection(os.environ.get("NOTION_TOKEN"), os.environ.get("NOTION_DATABASE_ID"), os.environ.get("NOTION_NETWORTH_ID"))

    #notion_.get_networth()
    notion_.get_all_entries()
    notion_.convert_special_tags()
    #notion_.insert_reoccurring_expenses(month, year)
    #notion_.convert_special_currencies_to_euro()
    #notion_.prefill_account_statements(last_month, last_year, data_dir)