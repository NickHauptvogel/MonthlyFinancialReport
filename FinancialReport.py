from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from matplotlib.backends.backend_pdf import PdfPages

data_dir = 'C:\\Users\\Nick\\OneDrive\\Dokumente\\Persönlich\\Buntentor\\02_Financials\\01_monthly\\'
reports_dir = 'C:\\Users\\Nick\\OneDrive\\Dokumente\\Persönlich\\Buntentor\\02_Financials\\01_monthly\\reports\\'
months = ['10']
current_year = '2022'


def read_csv(month):
    df = pd.read_csv(data_dir + 'data.csv')
    # Delete last row (Tricount export info)
    df = df[:-1]

    df['Date & time'] = pd.to_datetime(df['Date & time'])
    df['Month'] = df['Date & time'].dt.month
    df['Year'] = df['Date & time'].dt.year

    # Only past expenses
    next_year = int(current_year)
    next_month = int(month) + 1
    if next_month > 12:
        next_month = next_month % 12
        next_year += 1

    last_date = datetime(next_year, next_month, 1)
    df = df[df['Date & time'] < last_date]

    # Drop unnecessary columns
    df = df.drop(
        columns=['Amount', 'Currency', 'Exchange rate', 'Attachment URL', 'Paid by', 'Paid by Nick', 'Paid by Tabea'])
    df = df.rename(columns={"Amount in default currency (EUR)": "EUR"})

    expenses = df[df['Transaction type'] == 'Expense']
    incomes = df[df['Transaction type'] == 'Income']

    return expenses, incomes


def plot_cost_and_regression(data, title, pp):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    points = [item for sublist in data.values.tolist() for item in sublist]

    X = np.array([[i, points[i]] for i in range(len(points))])
    reg = LinearRegression().fit(X[:, 0].reshape(-1, 1), X[:, 1].reshape(-1, 1))
    y_pred = reg.predict(X[:, 0].reshape(-1, 1))
    data['Regression'] = y_pred
    data[data.columns[0]].plot(ax=ax, style='-')
    data['Regression'].plot(ax=ax, title=title, style='k--', figsize=(15, 10), grid=True)
    ax.annotate(str(np.round(reg.coef_[0][0], 2)), (len(points) - 1, y_pred[len(points) - 1] + 10))
    for x, y in zip([i for i in range(len(points))], points):
        ax.annotate(str(round(y, 2)), (x, y), c='C0')

    pp.savefig(fig)


def main():
    groceries_list = ['Groceries', 'Penny', 'Rossmann', 'Rewe']
    charges_list = ['Rent & Charges', 'Abonnements', 'Geschäftsreise', 'Healthcare', 'Internet & Telefon', 'Shares & Fonds', 'Green']
    shopping_list = ['Accommodation', 'Shopping', 'Presents', 'Hobby']
    leisure_list = ['Leisure', 'Date', 'Eating out', 'Sinnlos Aktivitäten']
    education_list = ['Education', 'Denmark Preparation', 'Studies']

    for current_month in months:

        expenses, incomes = read_csv(current_month)

        pp = PdfPages(f'{reports_dir}Report_{current_year}_{current_month}.pdf')

        fig = plt.figure()
        ax = fig.add_subplot(111)
        cats_cost = expenses \
            .groupby(['Year', 'Month', 'Category']) \
            .agg({'EUR': 'sum'})
        cats_cost_agg = cats_cost.unstack(level=-1)['EUR']
        vacation = cats_cost_agg.filter(regex='^Urlaub', axis=1).sum(axis=1)
        cats_cost_agg = cats_cost_agg.filter(regex='^(?!Urlaub).+')
        cats_cost_agg['Urlaub'] = vacation
        groceries = cats_cost_agg[groceries_list].sum(axis=1)
        cats_cost_agg = cats_cost_agg.drop(columns=groceries_list)
        cats_cost_agg['Groceries'] = groceries
        charges = cats_cost_agg[charges_list].sum(axis=1)
        cats_cost_agg = cats_cost_agg.drop(columns=charges_list)
        cats_cost_agg['Charges, Costs'] = charges
        shopping = cats_cost_agg[shopping_list].sum(axis=1)
        cats_cost_agg = cats_cost_agg.drop(columns=shopping_list)
        cats_cost_agg['Shopping'] = shopping
        leisure = cats_cost_agg[leisure_list].sum(axis=1)
        cats_cost_agg = cats_cost_agg.drop(columns=leisure_list)
        cats_cost_agg['Leisure'] = leisure
        education = cats_cost_agg[education_list].sum(axis=1)
        cats_cost_agg = cats_cost_agg.drop(columns=education_list)
        cats_cost_agg['Education'] = education
        cats_cost_agg.plot.bar(ax=ax, title='Categories cost trend', stacked=True,
                                             figsize=(15, 10))
        pp.savefig(fig)

        overall_cost = expenses.groupby(['Year', 'Month']).agg({'EUR': 'sum'})
        plot_cost_and_regression(overall_cost, 'Overall cost trend', pp)

        overall_cost = expenses.groupby(['Year', 'Month']).agg({'Impacted to Tabea': 'sum'})
        # Turn income from - to +
        overall_cost = overall_cost * -1
        plot_cost_and_regression(overall_cost, 'Tabea - Overall cost trend', pp)

        overall_cost = expenses.groupby(['Year', 'Month']).agg({'Impacted to Nick': 'sum'})
        # Turn income from - to +
        overall_cost = overall_cost * -1
        plot_cost_and_regression(overall_cost, 'Nick - Overall cost trend', pp)

        groceries_cost = expenses[expenses['Category'].isin(groceries_list)]
        groceries_cost_tabea = groceries_cost.groupby(['Year', 'Month']).agg({'Impacted to Tabea': 'sum'})
        groceries_cost_nick = groceries_cost.groupby(['Year', 'Month']).agg({'Impacted to Nick': 'sum'})
        # Turn income from - to +
        groceries_cost_tabea = groceries_cost_tabea * -1
        groceries_cost_nick = groceries_cost_nick * -1
        plot_cost_and_regression(groceries_cost_tabea, 'Tabea - Groceries cost trend', pp)
        plot_cost_and_regression(groceries_cost_nick, 'Nick - Groceries cost trend', pp)

        current_month_cost = cats_cost.loc[int(current_year), int(current_month)].reset_index()
        current_month_cost = current_month_cost.sort_values(by=['EUR'], ascending=False)
        current_month_title = f'Cost structure {current_month}/{current_year}'

        fig1, ax1 = plt.subplots(figsize=(15, 13))
        explode = [0.05 * i for i in range(len(current_month_cost))]
        ax1.pie(current_month_cost['EUR'], labels=current_month_cost['Category'],
                autopct='%1.1f%%', startangle=0, pctdistance=0.8, explode=explode)
        ax1.set_title(current_month_title)
        # draw circle
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)  # Equal aspect ratio ensures that pie is drawn as a circle
        ax1.axis('equal')
        plt.tight_layout()
        pp.savefig(fig1)

        fig = plt.figure(figsize=(15, 5))
        ax = fig.add_subplot(111)
        top_expenses = expenses[(expenses['Year'] == int(current_year)) & (expenses['Month'] == int(current_month))] \
            .sort_values(by=['EUR'], ascending=False).head(n=10)
        top_expenses = top_expenses.drop(columns=['Date & time', 'Year', 'Month'])

        ax.table(cellText=top_expenses.values,
                 colLabels=top_expenses.columns,
                 loc='center')
        ax.set_title("Top 10 Expenses")
        ax.axis('off')
        plt.tight_layout()
        pp.savefig(fig)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        expenses_per_person = expenses.groupby(['Year', 'Month']).agg(
            {'Impacted to Nick': 'sum', 'Impacted to Tabea': 'sum'})
        incomes_per_person = incomes.groupby(['Year', 'Month']).agg(
            {'Impacted to Nick': 'sum', 'Impacted to Tabea': 'sum'})
        netto = expenses_per_person.add(incomes_per_person)
        costs_per_person = expenses_per_person.merge(incomes_per_person, how='outer', left_index=True, right_index=True) \
            .merge(netto, how='outer', left_index=True, right_index=True)
        costs_per_person[costs_per_person.columns[0:2]].plot.bar(ax=ax, color=['maroon', 'indianred'], alpha=0.4,
                                                                 grid=True, figsize=(15, 10))
        costs_per_person[costs_per_person.columns[2:4]].plot.bar(ax=ax, color=['olivedrab', 'yellowgreen'], alpha=0.4,
                                                                 grid=True)
        costs_per_person[costs_per_person.columns[4:6]].plot.bar(ax=ax, grid=True)
        savings_total = costs_per_person[costs_per_person.columns[4:6]].sum().to_numpy()
        ax.legend(['Expense Nick', 'Expense Tabea', 'Income Nick', 'Income Tabea', 'Delta Nick', 'Delta Tabea'])
        ax.set_title('Cost deltas per month and person')

        patches_per_cat = len(ax.patches) // 6
        for p in ax.patches[4 * patches_per_cat:]:
            if p.get_height() > 0:
                ax.annotate(str(round(p.get_height(), 2)), (p.get_x() - 0.12, p.get_height() + 50))
            else:
                ax.annotate(str(round(p.get_height(), 2)), (p.get_x() - 0.12, p.get_height() - 230))

        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        # place a text box in upper left in axes coords
        ax.text(0.05, 0.95, f"Total savings \n--------------\nNick: {savings_total[0]}€\nTabea: {savings_total[1]}€",
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        # plt.show()
        pp.savefig(fig)

        pp.close()

        expenses.to_excel(data_dir + current_year + current_month + '.xlsx')


if __name__ == '__main__':
    main()
