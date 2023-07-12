from datetime import datetime
import tempfile
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from matplotlib.backends.backend_pdf import PdfPages
from docxtpl import DocxTemplate, InlineImage
from docx.shared import Mm


def filter_past_only(month, year, df):
    # Include only past expenses
    next_year = int(year)
    next_month = int(month) + 1
    if next_month > 12:
        next_month = next_month % 12
        next_year += 1

    last_date = datetime(next_year, next_month, 1)
    df = df[df['Date_Time'] < last_date]

    return df


def read_tricount_data(data_dir, filename="data.csv") -> pd.DataFrame:
    """
    Reads and returns dataframe in following format:
    Title: Title of expense / income
    Total: Total amount spent (negative) or received (positive)
    Category: Category of expense
    Nick: Nicks amount spent (negative) or received (positive)
    Tabea: Tabeas amount spent (negative) or received (positive)
    Date_Time: Timestamp
    Month: Number of month
    Year: Number of year

    :param data_dir: path to data
    :param filename: filename
    :return: pd.Dataframe
    """

    df = pd.read_csv(data_dir + filename)
    # Delete last row (Tricount export info)
    df = df[:-1]

    df['Date_Time'] = pd.to_datetime(df['Date & time'])
    df['Month'] = df['Date_Time'].dt.month
    df['Year'] = df['Date_Time'].dt.year

    # Drop unnecessary columns
    df = df.drop(
        columns=['Amount', 'Currency', 'Exchange rate', 'Attachment URL', 'Paid by',
                 'Paid by Nick', 'Paid by Tabea', 'Date & time'])
    # Normalize column names
    df = df.rename(columns={"Title": "Title"})
    df = df.rename(columns={"Amount in default currency (EUR)": "Total"})
    df['Total'] = df['Total'] * -1
    df = df.rename(columns={"Category": "Category"})
    df = df.rename(columns={"Impacted to Nick": "Nick"})
    df = df.rename(columns={"Impacted to Tabea": "Tabea"})

    # Handle direct transactions
    def f(row):
        if row['Transaction type'] == 'Money transfer':
            if row['Nick'] == 0.0:
                row['Nick'] = -1 * row['Total']
            elif row['Tabea'] == 0.0:
                row['Tabea'] = -1 * row['Total']
            row['Total'] = 0.0

        return row

    df = df.apply(f, axis=1)
    df = df.drop(columns=['Transaction type'])

    return df


def get_trend_cats():
    return ['Charges', 'Groceries', 'Shopping', 'Leisure', 'Urlaub']


def get_num_categories():
    return 9  # Change to automatic


def normalize_categories(context, s_categories):
    normalize_dict = {
        'Charges': ['Rent & Charges', 'Abonnements', 'Geschäftsreise', 'Healthcare', 'Internet & Telefon',
                    'Shares & Fonds'],
        'Groceries': ['Groceries', 'Penny', 'Rossmann', 'Rewe'],
        'Shopping': ['Accommodation', 'Shopping', 'Hobby'],
        'Presents': ['Presents'],
        'Leisure': ['Leisure', 'Date', 'Eating out', 'Sinnlos Aktivitäten'],
        'Education': ['Education', 'Denmark Preparation', 'Studies'],
        'Transport': ['Transport'],
        'Salary': ['Salary'],
        'Child Support': ['Child Support']
    }
    normalize_dict_reverse = {}
    for k, vs in normalize_dict.items():
        for v in vs:
            normalize_dict_reverse[v] = k
    normalize_dict['Misc'] = set()
    normalize_dict['Urlaub'] = set()

    def insert_norm_category(cell):
        if 'Urlaub' in str(cell):
            value = 'Urlaub'
            normalize_dict['Urlaub'].add(cell)
        elif pd.isnull(cell):
            value = 'Misc'
        else:
            value = normalize_dict_reverse.get(cell)
            if value is None:
                normalize_dict['Misc'].add(cell)
                value = 'Misc'
        cell = value
        return cell

    s_categories_norm = s_categories.apply(insert_norm_category)

    for key in normalize_dict.keys():
        context[f'{key.lower()}_list'] = "{0}".format(', '.join(map(str, normalize_dict[key])))

    return context, s_categories_norm


def get_incomes(data, column='Total'):
    return data[data[column] > 0]


def get_expenses(data, column='Total'):
    return data[data[column] < 0]


def get_month_data(data, month, year):
    data = data[data['Year'] == year]
    return data[data['Month'] == month]


def get_year_data(data, year):
    return data[data['Year'] == year]


def render_income(context, data_month: pd.DataFrame):
    def get_income_fields(column, field_abbr):
        total_sum = total[column].sum()
        context[f'income_{field_abbr}_m'] = np.round(total_sum, 2)

    total = get_incomes(data_month)
    total_sum_cats = total.groupby(['Category']).agg({'Total': 'sum', 'Nick': 'sum', 'Tabea': 'sum'})
    total_sum_cats = total_sum_cats.sort_values('Total', ascending=False)
    total_sum_cats = np.round(total_sum_cats, 2)
    total_sum_cats['Category'] = total_sum_cats.index
    context['income_m'] = total_sum_cats.to_dict(orient='records')

    get_income_fields('Total', 'tot')
    get_income_fields('Tabea', 'ta')
    get_income_fields('Nick', 'ni')

    return context


def render_expenses(doc, context, tmpdirname, data_month: pd.DataFrame):
    def get_expense_fields(column, field_abbr):
        total_sum = total[column].sum()
        cats_total_dict = dict(cats_total[column])
        cats_total_dict['Expense'] = total_sum

        for k, v in cats_total_dict.items():
            context[f'{str.lower(k)}_{field_abbr}_m'] = np.round(v, 2)

    total = get_expenses(data_month)
    cats_total = total.groupby(['Category_Norm']).agg({'Total': 'sum', 'Nick': 'sum', 'Tabea': 'sum'})
    get_expense_fields('Total', 'tot')
    get_expense_fields('Nick', 'ni')
    get_expense_fields('Tabea', 'ta')

    # Plot pie charts
    cats_total = cats_total * -1  # Convert expenses to positive
    cats_total = cats_total[['Total', 'Tabea', 'Nick']]  # Rearrange columns
    n = get_num_categories()
    colors = [plt.cm.Blues(float(i) / (n + 1)) for i in range(1, n + 1)]
    cats_total.plot.pie(subplots=True, figsize=(15, 5), legend=True, title=['Total', 'Tabea', 'Nick'],
                        labeldistance=None, ylabel='', colors=colors, fontsize=80)
    plt.tight_layout()
    img_name = os.path.join(tmpdirname, 'expense_pie_charts.png')
    plt.savefig(img_name)
    context['expense_pie_charts'] = InlineImage(doc, img_name, width=Mm(160))

    return context


def render_balance(context, full_data: pd.DataFrame, year_data: pd.DataFrame, month_data: pd.DataFrame):
    def get_balance_fields(column, field_abbr):
        total_sum = full_data[column].sum()
        year_sum = year_data[column].sum()
        month_sum = month_data[column].sum()

        context[f'balance_{field_abbr}_m'] = np.round(month_sum, 2)
        context[f'balance_{field_abbr}_y'] = np.round(year_sum, 2)
        context[f'balance_{field_abbr}_tot'] = np.round(total_sum, 2)

    get_balance_fields('Total', 'tot')
    get_balance_fields('Nick', 'ni')
    get_balance_fields('Tabea', 'ta')

    return context


def plot_trend_regression(tmpdirname, data, title, filename):
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

    plt.figure(figsize=(10, 4))
    plt.grid()
    plt.plot(indices, X[:, 1], label='Expenses')
    plt.plot(indices_clean, y_pred, label=f'Regression Line: m = {str(np.round(reg.coef_[0][0], 2))} EUR')
    plt.legend()
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(os.path.join(tmpdirname, filename))


def render_trends(doc, context, data, tmpdirname):
    def render_trends_separate(column, field_abbr):
        cats_total_column = cats_total[column]

        cat = 'Total'
        plot_trend_regression(tmpdirname, cats_total_column.sum(axis=1), f'{column} {cat}',
                              f'trend_{field_abbr}_{cat}.png')
        context[f'trend_{field_abbr}_{str.lower(cat)}'] = InlineImage(doc, os.path.join(tmpdirname,
                                                                                        f'trend_{field_abbr}_{cat}.png'),
                                                                      width=Mm(160))

        for cat in get_trend_cats():
            plot_trend_regression(tmpdirname, cats_total_column[cat], f'{column} {cat}',
                                  f'trend_{field_abbr}_{cat}.png')
            context[f'trend_{field_abbr}_{str.lower(cat)}'] = InlineImage(doc, os.path.join(tmpdirname,
                                                                                            f'trend_{field_abbr}_{cat}.png'),
                                                                          width=Mm(160))

    total = get_expenses(data)
    cats_total = total.groupby(['Year', 'Month', 'Category_Norm']).agg({'Total': 'sum', 'Nick': 'sum', 'Tabea': 'sum'})
    cats_total = cats_total.unstack(level=-1)
    cats_total = cats_total * -1

    render_trends_separate('Total', 'tot')
    render_trends_separate('Nick', 'ni')
    render_trends_separate('Tabea', 'ta')

    return context


def render_trend_balance(doc, context, data, tmpdirname):
    def render_trend_balance_separate(costs_merged, column, field_abbr, suffix):
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111)
        costs_merged[column + '_expense'].plot.bar(ax=ax, color=['maroon'], alpha=0.4)
        costs_merged[column + '_income'].plot.bar(ax=ax, color=['olivedrab'], alpha=0.4)
        costs_merged[column].plot.bar(ax=ax, grid=True)

        patches_per_cat = len(ax.patches) * 2 // 3
        for p in ax.patches[patches_per_cat:]:
            if p.get_height() > 0:
                ax.annotate(str(round(p.get_height(), 2)), (p.get_x(), p.get_height() + 80))
            else:
                ax.annotate(str(round(p.get_height(), 2)), (p.get_x(), p.get_height() - 200))

        plt.xticks(rotation=45)
        plt.tight_layout()

        name = f'trend_balance_{field_abbr}_{suffix}'
        plt.savefig(os.path.join(tmpdirname, name + '.png'))
        context[name] = InlineImage(doc, os.path.join(tmpdirname, name + '.png'), width=Mm(160))

    expenses_m = get_expenses(data).groupby(['Year', 'Month']).agg({'Total': 'sum', 'Nick': 'sum', 'Tabea': 'sum'})
    incomes_m = get_incomes(data).groupby(['Year', 'Month']).agg({'Total': 'sum', 'Nick': 'sum', 'Tabea': 'sum'})
    net_m = data.groupby(['Year', 'Month']).agg({'Total': 'sum', 'Nick': 'sum', 'Tabea': 'sum'})
    costs_m_merged = expenses_m.merge(incomes_m, how='outer', left_index=True, right_index=True,
                                      suffixes=("_expense", "_income")) \
        .merge(net_m, how='outer', left_index=True, right_index=True)
    expenses_y = get_expenses(data).groupby(['Year']).agg({'Total': 'sum', 'Nick': 'sum', 'Tabea': 'sum'})
    incomes_y = get_incomes(data).groupby(['Year']).agg({'Total': 'sum', 'Nick': 'sum', 'Tabea': 'sum'})
    net_y = data.groupby(['Year']).agg({'Total': 'sum', 'Nick': 'sum', 'Tabea': 'sum'})
    costs_y_merged = expenses_y.merge(incomes_y, how='outer', left_index=True, right_index=True,
                                      suffixes=("_expense", "_income")) \
        .merge(net_y, how='outer', left_index=True, right_index=True)

    render_trend_balance_separate(costs_m_merged, 'Total', 'tot', 'm')
    render_trend_balance_separate(costs_m_merged, 'Tabea', 'ta', 'm')
    render_trend_balance_separate(costs_m_merged, 'Nick', 'ni', 'm')

    render_trend_balance_separate(costs_y_merged, 'Total', 'tot', 'y')
    render_trend_balance_separate(costs_y_merged, 'Tabea', 'ta', 'y')
    render_trend_balance_separate(costs_y_merged, 'Nick', 'ni', 'y')

    return context


def render_top_expenses(context, full_data: pd.DataFrame, year_data: pd.DataFrame, month_data: pd.DataFrame, n=10):
    def sort_drop_dup_take_first_n_to_dict(df: pd.DataFrame, column):
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
        return df[:n].to_dict(orient='records')

    def render_top_expenses_separate(column, field_abbr):
        context[f'top_expenses_{field_abbr}_tot'] = sort_drop_dup_take_first_n_to_dict(full_data, column)
        context[f'top_expenses_{field_abbr}_y'] = sort_drop_dup_take_first_n_to_dict(year_data, column)
        context[f'top_expenses_{field_abbr}_m'] = sort_drop_dup_take_first_n_to_dict(month_data, column)

    def datetime_to_date(row):
        row['Date'] = row['Date_Time'].strftime("%d.%m.%Y")
        return row

    full_data = full_data.apply(datetime_to_date, axis=1)
    year_data = year_data.apply(datetime_to_date, axis=1)
    month_data = month_data.apply(datetime_to_date, axis=1)

    render_top_expenses_separate('Total', 'tot')
    render_top_expenses_separate('Nick', 'ni')
    render_top_expenses_separate('Tabea', 'ta')

    return context


def render_template(month, year, data, reports_dir, template_name="Report_Template.docx"):
    with tempfile.TemporaryDirectory() as tmpdirname:
        doc = DocxTemplate(template_name)
        context = {'current_month': str(month),
                   'current_year': str(year)}

        df = filter_past_only(month, year, data)

        context, col = normalize_categories(context, df['Category'])
        df['Category_Norm'] = col
        # Get month data
        current_month_data = get_month_data(df, month, year)
        current_year_data = get_year_data(df, year)

        context = render_income(context, current_month_data)
        context = render_expenses(doc, context, tmpdirname, current_month_data)
        context = render_balance(context, df, current_year_data, current_month_data)
        context = render_trends(doc, context, df, tmpdirname)
        context = render_trend_balance(doc, context, df, tmpdirname)
        context = render_top_expenses(context, df, current_year_data, current_month_data)

        doc.render(context)
        doc.save(os.path.join(reports_dir, f"Report_{year}_{month}.docx"))


if __name__ == '__main__':
    data_dir = 'C:\\Users\\Nick\\OneDrive\\Dokumente\\Persönlich\\Buntentor\\02_Financials\\01_monthly\\'
    reports_dir = 'C:\\Users\\Nick\\OneDrive\\Dokumente\\Persönlich\\Buntentor\\02_Financials\\01_monthly\\reports\\'
    month = 6
    year = 2023
    data = read_tricount_data(data_dir)
    data.to_excel(data_dir + f"{year}{month:02}.xlsx")
    render_template(month, year, data, reports_dir)

