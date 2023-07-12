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


def hex_to_rgba(hex: str, opacity=0.5):
    hex = hex.lstrip('#')
    r, g, b = tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))
    return f'rgba({r}, {g}, {b}, {opacity})'


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


def get_color_map(data_dir, file_name="color_mapping.xlsx", parent_category_sheet="parent_category",
                  tricount_sheet="tricount_category"):
    norm_color_map = pd.read_excel(os.path.join(data_dir, file_name), sheet_name=parent_category_sheet)
    norm_color_map = norm_color_map.set_index('parent_category')

    tricount_color_map = pd.read_excel(os.path.join(data_dir, file_name), sheet_name=tricount_sheet)
    tricount_color_map = tricount_color_map.set_index('tricount_category')

    return norm_color_map, tricount_color_map


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
    return ['Charges', 'Groceries', 'Shopping', 'Leisure', 'Urlaub']  # TODO In Excel sheet


def normalize_categories(context, s_categories, norm_color_map, tricount_color_map):
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
                value = tricount_color_map.loc[cell]['parent_category']
            except KeyError:
                misc_set.add(cell)
                value = 'Misc'
        cell = value
        return cell

    s_categories_norm = s_categories.apply(insert_norm_category)

    # Get distinct parent categories from norm_color_map
    parent_categories = norm_color_map.index.unique()
    for parent_category in parent_categories:
        # All values in tricount_color_map that have parent_category as parent_category
        context[f'{parent_category.lower()}_list'] = tricount_color_map[
            tricount_color_map['parent_category'] == parent_category].index.tolist()

    context['urlaub_list'] = list(urlaub_set)
    context['misc_list'] = list(misc_set)

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


def render_sankey(doc, context, tmpdirname, data_month: pd.DataFrame, tricount_color_map: pd.DataFrame):
    def render_sankey_separate(column, field_abbr):
        incomes = get_incomes(data_month, column=column)
        expenses = get_expenses(data_month, column=column)

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
        node_colors = [tricount_color_map.loc[label]['color'] if label in tricount_color_map.index else '#909090' for label in names_list]
        link_colors = [hex_to_rgba(c) for c in node_colors]

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
                target=[total_node] * len(income_labels) + [i + len(income_labels) for i in range(len(expense_labels))],
                value=income_sum_cats[column].tolist() + expense_sum_cats[column].tolist(),
                color=link_colors
            ))])

        filename = f'{tmpdirname}/sankey_{field_abbr}.png'
        fig.write_image(filename, width=800, height=600, scale=2)
        context[f'sankey_{field_abbr}'] = InlineImage(doc, filename, width=Mm(160))

    render_sankey_separate('Nick', 'ni')
    render_sankey_separate('Tabea', 'ta')

    return context


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


def render_expenses(doc, context, tmpdirname, data_month: pd.DataFrame, norm_color_map: pd.DataFrame):
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

    def pie_separate(column, field_abbr, cats):
        cats_single = cats[[column]]
        cats_single = cats_single.sort_values(by=[column], ascending=False)
        # Get colors as list
        colors = list(cats_single.index.map(lambda x: norm_color_map.loc[x, 'color']))
        cats_single.plot.pie(subplots=True, figsize=(5, 5), legend=True, title=column,
                             labeldistance=None, ylabel='', colors=colors, fontsize=80)
        plt.tight_layout()
        img_name = os.path.join(tmpdirname, f'expense_pie_charts_{field_abbr}.png')
        plt.savefig(img_name)
        context[f'expense_pie_charts_{field_abbr}'] = InlineImage(doc, img_name, width=Mm(50))

    pie_separate('Total', 'tot', cats_total)
    pie_separate('Nick', 'ni', cats_total)
    pie_separate('Tabea', 'ta', cats_total)

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


def render_template(month, year, data, norm_color_map, tricount_color_map, reports_dir,
                    template_name="Report_Template.docx"):
    with tempfile.TemporaryDirectory() as tmpdirname:
        doc = DocxTemplate(template_name)
        context = {'current_month': str(month),
                   'current_year': str(year)}

        # Get the earliest date
        earliest_date = data['Date_Time'].min()
        # Counts the number of months between the earliest date and the current date
        count_months = (year - earliest_date.year) * 12 + (month - earliest_date.month)
        context['count_months'] = count_months

        df = filter_past_only(month, year, data)

        context, col = normalize_categories(context, df['Category'], norm_color_map, tricount_color_map)
        df['Category_Norm'] = col
        # Get month data
        current_month_data = get_month_data(df, month, year)
        current_year_data = get_year_data(df, year)

        context = render_sankey(doc, context, tmpdirname, current_month_data, tricount_color_map)

        context = render_income(context, current_month_data)
        context = render_expenses(doc, context, tmpdirname, current_month_data, norm_color_map)
        context = render_balance(context, df, current_year_data, current_month_data)
        context = render_trends(doc, context, df, tmpdirname)
        context = render_trend_balance(doc, context, df, tmpdirname)
        context = render_top_expenses(context, df, current_year_data, current_month_data)

        doc.render(context)
        doc.save(os.path.join(reports_dir, f"Report_{year}_{month}.docx"))


if __name__ == '__main__':
    base_dir = 'C:\\Users\\Nick\\OneDrive\\Dokumente\\Persönlich\\Buntentor\\02_Financials\\01_monthly\\'
    reports_dir = base_dir + 'reports\\'
    data_dir = base_dir + 'data\\'
    csv_name = 'Tricount_BuntentorAxiom.csv'
    month = 5
    year = 2023

    norm_color_map, tricount_color_map = get_color_map(data_dir)
    data = read_tricount_data(data_dir, filename=csv_name)
    data.to_excel(base_dir + f"{year}{month:02}.xlsx")
    render_template(month, year, data, norm_color_map, tricount_color_map, reports_dir)
