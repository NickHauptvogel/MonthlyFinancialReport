from datetime import datetime
import tempfile
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

from docxtpl import DocxTemplate, InlineImage
from docx.shared import Mm


def hex_to_rgba(hex: str, opacity=1.0):
    hex = hex.lstrip('#')
    r, g, b = tuple(int(hex[i:i + 2], 16) for i in (0, 2, 4))
    return f'rgba({r}, {g}, {b}, {opacity})'


def get_incomes(data, column='Total'):
    return data[data[column] > 0]


def get_expenses(data, column='Total'):
    return data[data[column] < 0]


def get_month_data(data, month, year):
    data = data[data['Year'] == year]
    return data[data['Month'] == month]


def get_year_data(data, year):
    return data[data['Year'] == year]


def datetime_to_date(row):
    row['Date'] = row['Date_Time'].strftime("%d.%m.%Y")
    return row


class FinancialReportCreator:

    def __init__(self,
                 data_dir,
                 tricount_csv_name,
                 networth_xlsx_name,
                 color_mapping_xlsx_name,
                 template_name,
                 month,
                 year,
                 reports_dir,
                 excel_dir,
                 people,
                 parent_category_sheet="parent_category",
                 tricount_sheet="tricount_category",
                 networth_sheet="networth_category"):

        self.month = month
        self.year = year
        self.data_dir = data_dir
        self.reports_dir = reports_dir
        self.excel_dir = excel_dir
        # Make sure all directories exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.excel_dir, exist_ok=True)
        self.people = people
        self.analysis_columns = people + [('Total', 'tot')]
        self.tmpdirname = None
        self.context = {'current_month': f"{month:02}",
                        'current_year': f"{year}",
                        'person1_name': people[0][0],
                        'person2_name': people[1][0]}
        self.doc = DocxTemplate(template_name)
        self.norm_color_map, self.tricount_color_map, self.networth_color_map = \
            self.read_color_map(color_mapping_xlsx_name, parent_category_sheet, tricount_sheet, networth_sheet)
        self.networth = self.read_networth_data(networth_xlsx_name)
        self.data = self.read_tricount_data(tricount_csv_name)

        # Get the earliest date
        earliest_date = self.data['Date_Time'].min()
        # Counts the number of months between the earliest date and the current date
        count_months = (self.year - earliest_date.year) * 12 + (self.month - earliest_date.month)
        self.context['count_months'] = count_months

        # Remove future expenses for current month and year
        next_year = int(self.year)
        next_month = int(self.month) + 1
        if next_month > 12:
            next_month = next_month % 12
            next_year += 1

        last_date = datetime(next_year, next_month, 1)
        self.data = self.data[self.data['Date_Time'] < last_date]

        # Normalize categories
        self.data['Category_Norm'] = self.normalize_categories()
        # Turn datetime into date string
        self.data = self.data.apply(datetime_to_date, axis=1)
        # Store month and year data
        self.month_data = get_month_data(self.data, month, year)
        self.year_data = get_year_data(self.data, year)

        try:
            self.month_networth = self.networth.loc[year, month]
            self.month_networth = self.month_networth.unstack(level=-1).T
            self.context['display_networth'] = True
        except KeyError:
            self.context['display_networth'] = False

    def read_color_map(self, file_name, parent_category_sheet, tricount_sheet, networth_sheet):
        norm_color_map = pd.read_excel(os.path.join(self.data_dir, file_name), sheet_name=parent_category_sheet)
        norm_color_map = norm_color_map.set_index('parent_category')
        tricount_color_map = pd.read_excel(os.path.join(self.data_dir, file_name), sheet_name=tricount_sheet)
        tricount_color_map = tricount_color_map.set_index('tricount_category')
        networth_color_map = pd.read_excel(os.path.join(self.data_dir, file_name), sheet_name=networth_sheet)
        networth_color_map = networth_color_map.set_index('networth_category')

        return norm_color_map, tricount_color_map, networth_color_map

    def get_trend_cats(self):
        return list(self.norm_color_map[self.norm_color_map['show_trend'] == True].index)

    def get_cleaned_data(self, data):
        clean_cats = self.tricount_color_map[self.tricount_color_map['consider_in_cleaned'] == True].index
        return data[data['Category'].isin(clean_cats)]

    def read_networth_data(self, file_name):
        dfs = []
        names = [name for name, _ in self.people]
        for name in names:
            df = pd.read_excel(os.path.join(self.data_dir, file_name), sheet_name=name)
            df = df.rename(columns={"Value": name})
            dfs.append(df)

        # Stack dataframes
        df = pd.concat(dfs).reset_index(drop=True)
        df = df.fillna(0)
        df['Total'] = df[names].sum(axis=1)

        group_dict = {'Total': 'sum'}
        group_dict.update({name: 'sum' for name, _ in self.people})
        cats_total = df.groupby(['Year', 'Month', 'Category']).agg(group_dict)
        cats_total = cats_total.unstack(level=-1)

        return cats_total

    def read_tricount_data(self, filename) -> pd.DataFrame:
        """
        Reads and returns dataframe in following format:
        Title: Title of expense / income
        Total: Total amount spent (negative) or received (positive)
        Category: Category of expense
        Person1: Person 1s amount spent (negative) or received (positive)
        Person2: Person 2s amount spent (negative) or received (positive)
        Date_Time: Timestamp
        Month: Number of month
        Year: Number of year

        :param filename: filename
        :return: pd.Dataframe
        """

        df = pd.read_csv(os.path.join(self.data_dir, filename))
        # Replace &amp; with & in column names and data
        df.columns = df.columns.str.replace('&amp;', '&')
        df = df.replace('&amp;', '&', regex=True)
        # Delete last row (Tricount export info)
        df = df[:-1]

        df['Date_Time'] = pd.to_datetime(df['Date & time'])
        df['Month'] = df['Date_Time'].dt.month
        df['Year'] = df['Date_Time'].dt.year

        names = [name for name, _ in self.people]

        # Drop unnecessary columns
        df = df.drop(
            columns=['Amount', 'Currency', 'Exchange rate', 'Attachment URL', 'Paid by', 'Date & time']
                    + ['Paid by ' + name for name in names])
        # Normalize column names
        df = df.rename(columns={"Amount in default currency (EUR)": "Total"})
        df['Total'] = df['Total'] * -1  # Tricount shows expenses as positive values

        for name in names:
            df = df.rename(columns={f"Impacted to {name}": f"{name}"})

        # Handle direct transactions
        def f(row):
            if row['Transaction type'] == 'Money transfer':
                for name in names:
                    if row[name] == 0.0:
                        row[name] = -1 * row['Total']
                        break
                row['Total'] = 0.0

            return row

        df = df.apply(f, axis=1)
        df = df.drop(columns=['Transaction type'])

        return df

    def normalize_categories(self):
        urlaub_set = set()
        misc_set = set()

        def insert_norm_category(cell):
            if 'Urlaub' in str(cell):
                value = 'Urlaub'
                urlaub_set.add(cell)
            elif pd.isnull(cell):
                value = 'Misc'
            else:
                try:
                    # Get value from dataframe
                    value = self.tricount_color_map.loc[cell]['parent_category']
                except KeyError:
                    misc_set.add(cell)
                    value = 'Misc'
            cell = value
            return cell

        s_categories_norm = self.data['Category'].apply(insert_norm_category)

        # Get distinct parent categories from norm_color_map
        parent_categories = self.norm_color_map.index.unique()
        for parent_category in parent_categories:
            # All values in tricount_color_map that have parent_category as parent_category
            self.context[f'{parent_category.lower()}_list'] = self.tricount_color_map[
                self.tricount_color_map['parent_category'] == parent_category].index.tolist()

        self.context['urlaub_list'] = list(urlaub_set)
        self.context['misc_list'] = list(misc_set)

        return s_categories_norm

    def plot_pie_chart(self, column, cats, color_map, plot_name, plot_percentages=True):
        cats_single = cats.sort_values(by=[column], ascending=False)
        # Get colors as list
        colors = list(cats_single.index.map(lambda x: color_map.loc[x, 'color']))
        autopct = '%1.1f%%' if plot_percentages else None
        cats_single.plot.pie(subplots=True, figsize=(5, 5), legend=True, title=column,
                             labeldistance=None, ylabel='', colors=colors, fontsize=15, autopct=autopct)
        plt.tight_layout()
        img_name = os.path.join(self.tmpdirname, f'{plot_name}.png')
        plt.savefig(img_name)
        self.context[plot_name] = InlineImage(self.doc, img_name, width=Mm(50))

    def plot_trend_regression(self, data, title, plot_name, color='#000000'):
        points = data.values.tolist()
        X = np.array([[i, points[i]] for i in range(len(points))])
        X_clean = X[~np.isnan(points), :]
        X = np.nan_to_num(X)

        indices = list(data.index)
        indices = [str(elem) for elem in indices]
        indices_clean = np.array(indices)[~np.isnan(points)]

        reg = LinearRegression()
        reg.fit(X_clean[:, 0].reshape(-1, 1), X_clean[:, 1].reshape(-1, 1))
        y_pred = np.array(reg.predict(X_clean[:, 0].reshape(-1, 1)))

        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(111)
        plt.grid()
        ax.plot(indices, X[:, 1], label='Expenses', color=color)
        ax.plot(indices_clean, y_pred, label=f'Lin Reg. m = {str(np.round(reg.coef_[0][0], 2))} €/month',
                color='#000000', linestyle='--')
        ax.patch.set_facecolor(color)
        ax.patch.set_alpha(0.05)
        plt.legend()
        plt.title(title)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        fig.savefig(os.path.join(self.tmpdirname, f'{plot_name}.png'), facecolor=fig.get_facecolor(), edgecolor='none')

    def render_sankey(self, column, field_abbr, incomes, expenses):
        income_sum_cats = incomes.groupby(['Category']).agg({column: 'sum'})
        income_sum_cats = np.round(income_sum_cats, 2)
        income_sum_cats = income_sum_cats.sort_values(by=[column], ascending=False)
        income_sum_cats = income_sum_cats[income_sum_cats[column] > 0]
        income_sum_cats['Category'] = income_sum_cats.index

        expense_sum_cats = expenses.groupby(['Category']).agg({column: 'sum'})
        expense_sum_cats = np.round(expense_sum_cats, 2)
        expense_sum_cats *= -1
        expense_sum_cats = expense_sum_cats.sort_values(by=[column], ascending=False)
        expense_sum_cats = expense_sum_cats[expense_sum_cats[column] > 0]
        expense_sum_cats['Category'] = expense_sum_cats.index

        # Combine income and expense categories with their values
        income_labels = [f'{label} ({income_sum_cats.loc[label][column]}€)' for label in income_sum_cats.index]
        income_names = [label for label in income_sum_cats.index]
        expense_labels = [f'{label} ({expense_sum_cats.loc[label][column]}€)' for label in expense_sum_cats.index]
        expense_names = [label for label in expense_sum_cats.index]

        labels_list = income_labels + expense_labels
        names_list = income_names + expense_names
        # Use grey for as default
        node_colors = [
            self.tricount_color_map.loc[label]['color'] if label in self.tricount_color_map.index else '#909090'
            for label in names_list]
        link_colors = [hex_to_rgba(c, opacity=0.5) for c in node_colors]

        total_node = len(labels_list)
        saldo = incomes[column].sum() + expenses[column].sum()
        labels_list.append('Balance (' + str(round(saldo, 2)) + '€)')
        node_colors.append('#F0F0F0')

        # Create Sankey
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.0),
                label=labels_list,
                color=node_colors
            ),
            link=dict(
                source=[i for i in range(len(income_labels))] + [total_node] * len(expense_labels),
                target=[total_node] * len(income_labels) + [i + len(income_labels) for i in
                                                            range(len(expense_labels))],
                value=income_sum_cats[column].tolist() + expense_sum_cats[column].tolist(),
                color=link_colors
            ))])

        filename = f'{self.tmpdirname}/sankey_{field_abbr}.png'
        fig.write_image(filename, width=800, height=600, scale=2)
        self.context[f'sankey_{field_abbr}'] = InlineImage(self.doc, filename, width=Mm(160))

    def render_expenses(self, column, field_abbr, expenses):
        cats_total = expenses.groupby(['Category_Norm']).agg({column: 'sum'})
        cats_total_dict = dict(cats_total[column])
        cats_total_dict['Expense'] = expenses[column].sum()

        for k, v in cats_total_dict.items():
            self.context[f'{str.lower(k)}_{field_abbr}_m'] = np.round(v, 2)

        cats_total = cats_total * -1  # Convert expenses to positive
        self.plot_pie_chart(column, cats_total, self.norm_color_map, f'expense_pie_charts_{field_abbr}',
                            plot_percentages=False)

    def render_networth(self, column, field_abbr):
        self.context[f'networth_{field_abbr}'] = np.round(self.month_networth[column].sum(), 2)
        self.plot_pie_chart(column, self.month_networth[[column]], self.networth_color_map,
                            f'networth_pie_{field_abbr}', plot_percentages=True)
        plot_name = f'networth_trend_{field_abbr}'

        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(111)
        self.networth[column].plot.area(ax=ax)
        plt.legend(loc='upper left')
        plt.xticks(rotation=45, ha='right')
        plt.title(column)
        plt.tight_layout()
        fig.savefig(os.path.join(self.tmpdirname, f'{plot_name}.png'))
        self.context[plot_name] = InlineImage(self.doc, os.path.join(self.tmpdirname, f'{plot_name}.png'),
                                              width=Mm(160))

    def render_trends(self, column, field_abbr):
        total = get_expenses(self.data, column=column)
        cats_total = total.groupby(['Year', 'Month', 'Category_Norm']).agg({column: 'sum'})
        cats_total = cats_total.unstack(level=-1)
        cats_total = cats_total * -1
        cats_total_column = cats_total[column]
        if self.context.get('trends') is None:
            self.context['trends'] = {}

        def add_plot_to_context(arg_cat, arg_plot_name):
            if self.context['trends'].get(str.lower(arg_cat)) is None:
                self.context['trends'][str.lower(arg_cat)] = {}
                self.context['trends'][str.lower(arg_cat)]['title'] = f'{arg_cat} Cost Trend'
                self.context['trends'][str.lower(arg_cat)]['list'] = "TODO"
            self.context['trends'][str.lower(arg_cat)][field_abbr] = \
                InlineImage(self.doc, os.path.join(self.tmpdirname, f'{arg_plot_name}.png'), width=Mm(160))

        # Total trend
        cat = 'Total'
        plot_name = f'trend_{field_abbr}_{str.lower(cat)}'
        self.plot_trend_regression(cats_total_column.sum(axis=1), f'{column} {cat}', plot_name)
        add_plot_to_context(cat, plot_name)

        # Category trends
        for cat in self.get_trend_cats():
            plot_name = f'trend_{field_abbr}_{str.lower(cat)}'
            color = self.norm_color_map.loc[cat, 'color']
            self.plot_trend_regression(cats_total_column[cat], f'{column} {cat}', plot_name, color=color)
            add_plot_to_context(cat, plot_name)

    def render_trend_balance(self, column, field_abbr):
        def render_trend_balance_separate(costs_merged, suffix):
            fig = plt.figure(figsize=(15, 10))
            ax = fig.add_subplot(111)
            costs_merged[column + '_expense'].plot.bar(ax=ax, color=['#F7CAAC'])
            costs_merged[column + '_income'].plot.bar(ax=ax, color=['#ACB9CA'])
            costs_merged[column].plot.bar(ax=ax, grid=True, color=['#303030'], alpha=0.4)

            patches_per_cat = len(ax.patches) * 2 // 3
            for p in ax.patches[patches_per_cat:]:
                if p.get_height() > 0:
                    ax.annotate(str(round(p.get_height(), 2)), (p.get_x(), p.get_height() + 80))
                else:
                    ax.annotate(str(round(p.get_height(), 2)), (p.get_x(), p.get_height() - 200))

            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            name = f'trend_balance_{field_abbr}_{suffix}'
            path = os.path.join(self.tmpdirname, name + '.png')
            plt.savefig(path)
            self.context[name] = InlineImage(self.doc, path, width=Mm(160))

        expenses_m = get_expenses(self.data, column=column).groupby(['Year', 'Month']).agg({column: 'sum'})
        incomes_m = get_incomes(self.data, column=column).groupby(['Year', 'Month']).agg({column: 'sum'})
        net_m = self.data.groupby(['Year', 'Month']).agg({column: 'sum'})
        costs_m_merged = expenses_m.merge(incomes_m, how='outer', left_index=True, right_index=True,
                                          suffixes=("_expense", "_income")) \
            .merge(net_m, how='outer', left_index=True, right_index=True)

        expenses_y = get_expenses(self.data, column=column).groupby(['Year']).agg({column: 'sum'})
        incomes_y = get_incomes(self.data, column=column).groupby(['Year']).agg({column: 'sum'})
        net_y = self.data.groupby(['Year']).agg({column: 'sum'})
        costs_y_merged = expenses_y.merge(incomes_y, how='outer', left_index=True, right_index=True,
                                          suffixes=("_expense", "_income")) \
            .merge(net_y, how='outer', left_index=True, right_index=True)

        render_trend_balance_separate(costs_m_merged, 'm')
        render_trend_balance_separate(costs_y_merged, 'y')

    def render_top_expenses(self, column, field_abbr, n=10):
        def sort_drop_dup_take_first_n_to_dict(df: pd.DataFrame):
            def mark_rows(row):
                if row['Number'] > 1:
                    row['Date'] = str(row['Number']) + ' times'
                return row

            df_number = df.groupby([column, 'Title']).size().reset_index().rename(columns={0: 'Number'})

            df = df.drop_duplicates(subset=[column, 'Title'], keep='last')
            df = df.merge(df_number)
            df = df.apply(mark_rows, axis=1)
            df = df.reset_index(drop=True)
            df = df.sort_values(column)
            # Filter out rows containing "Miete"
            df = df[~df['Title'].str.contains('Miete')]

            # Rename columns in analysis_columns
            cols = df.columns
            for col, abbr in self.analysis_columns:
                if col in cols:
                    df = df.rename(columns={col: abbr})

            return df[:n].to_dict(orient='records')

        self.context[f'top_expenses_{field_abbr}_tot'] = sort_drop_dup_take_first_n_to_dict(self.data)
        self.context[f'top_expenses_{field_abbr}_y'] = sort_drop_dup_take_first_n_to_dict(self.year_data)
        self.context[f'top_expenses_{field_abbr}_m'] = sort_drop_dup_take_first_n_to_dict(self.month_data)

    def create_report(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.tmpdirname = tmpdirname

            self.data.to_excel(os.path.join(self.excel_dir, f"{self.year}{self.month:02}.xlsx"))

            # Render total income (independent of column)
            income_month_data = get_incomes(self.month_data)
            income_month_sum_cats = income_month_data.groupby(['Category']).agg(
                {column: 'sum' for column, _ in self.analysis_columns})
            income_month_sum_cats = income_month_sum_cats.sort_values('Total', ascending=False)
            income_month_sum_cats = np.round(income_month_sum_cats, 2)
            income_month_sum_cats['Category'] = income_month_sum_cats.index
            income_month_sum_cats = income_month_sum_cats.rename(columns={self.people[0][0]: 'person1'})
            income_month_sum_cats = income_month_sum_cats.rename(columns={self.people[1][0]: 'person2'})
            self.context['income_m'] = income_month_sum_cats.to_dict(orient='records')

            for column, field_abbr in self.analysis_columns:
                income_month_data = get_incomes(self.month_data, column=column)
                income_year_data = get_incomes(self.year_data, column=column)
                income_data = get_incomes(self.data, column=column)
                income_month_cleaned = self.get_cleaned_data(income_month_data)
                income_year_cleaned = self.get_cleaned_data(income_year_data)
                income_cleaned = self.get_cleaned_data(income_data)
                expense_month_data = get_expenses(self.month_data, column=column)
                expense_year_data = get_expenses(self.year_data, column=column)
                expense_data = get_expenses(self.data, column=column)

                self.context[f'income_{field_abbr}_m'] = np.round(income_month_data[column].sum(), 2)
                self.context[f'income_{field_abbr}_y'] = np.round(income_year_data[column].sum(), 2)
                self.context[f'income_{field_abbr}_tot'] = np.round(income_data[column].sum(), 2)
                self.context[f'income_cleaned_{field_abbr}_m'] = np.round(income_month_cleaned[column].sum(), 2)
                self.context[f'income_cleaned_{field_abbr}_y'] = np.round(income_year_cleaned[column].sum(), 2)
                self.context[f'income_cleaned_{field_abbr}_tot'] = np.round(income_cleaned[column].sum(), 2)
                self.context[f'balance_{field_abbr}_m'] = np.round(income_month_data[column].sum()+expense_month_data[column].sum(), 2)
                self.context[f'balance_{field_abbr}_y'] = np.round(income_year_data[column].sum()+expense_year_data[column].sum(), 2)
                self.context[f'balance_{field_abbr}_tot'] = np.round(income_data[column].sum()+expense_data[column].sum(), 2)
                self.context[f'balance_cleaned_{field_abbr}_m'] = np.round(income_month_cleaned[column].sum()+expense_month_data[column].sum(), 2)
                self.context[f'balance_cleaned_{field_abbr}_y'] = np.round(income_year_cleaned[column].sum()+expense_year_data[column].sum(), 2)
                self.context[f'balance_cleaned_{field_abbr}_tot'] = np.round(income_cleaned[column].sum()+expense_data[column].sum(), 2)
                self.context[f'expense_{field_abbr}_m'] = np.round(expense_month_data[column].sum(), 2)
                self.context[f'expense_{field_abbr}_y'] = np.round(expense_year_data[column].sum(), 2)
                self.context[f'expense_{field_abbr}_tot'] = np.round(expense_data[column].sum(), 2)

                self.render_sankey(column, field_abbr, income_month_data, expense_month_data)
                self.render_expenses(column, field_abbr, expense_month_data)
                if self.context['display_networth']:
                    self.render_networth(column, field_abbr)
                self.render_trends(column, field_abbr)
                self.render_trend_balance(column, field_abbr)
                self.render_top_expenses(column, field_abbr)

            # Turn trends into a list of dicts
            self.context['trends'] = [v for k, v in self.context['trends'].items()]

            self.doc.render(self.context)
            self.doc.save(os.path.join(self.reports_dir, f"Report_{self.year}_{self.month:02}.docx"))


if __name__ == '__main__':
    base_dir = '/mnt/c/Users/NHaup/OneDrive/Dokumente/Persönlich/Buntentor/02_Financials/01_monthly/'
    reports_dir = base_dir + 'reports/'
    data_dir = base_dir + 'data/'
    excel_dir = base_dir + 'excel_exports/'
    csv_name = 'Tricount_BuntentorAxiom.csv'
    networth_name = 'networth.xlsx'
    color_mapping_name = 'color_mapping.xlsx'
    template_name = 'Report_Template.docx'
    month = 2
    year = 2024
    people = [('Tabea', 'ta'), ('Nick', 'ni')]

    fr_creator = FinancialReportCreator(data_dir, csv_name, networth_name, color_mapping_name, template_name, month,
                                        year, reports_dir, excel_dir, people)
    fr_creator.create_report()
