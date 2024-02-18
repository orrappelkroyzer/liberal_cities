import os, sys
from pathlib import Path
local_python_path = os.path.sep.join(__file__.split(os.path.sep)[:-1])
if local_python_path not in sys.path:
    sys.path.append(local_python_path)

from utils.utils import load_config, get_logger
logger = get_logger(__name__)
config = load_config(Path(local_python_path) / "config.json")
from utils.plotly_utils import fix_and_write
from scipy import stats
import numpy as np
import plotly.graph_objects as go
import pandas as pd

def news_bar_group_function(df):
    tf_cols = [col for col in df.columns if col.startswith('News Consumtion:')]
    tf_cols_dict = {x : x[17:] for x in tf_cols}
    df = df.rename(columns=tf_cols_dict)
    tf_cols = list(tf_cols_dict.values())
    return df.groupby('Group')[tf_cols].sum().unstack().reset_index().rename(columns={'level_0' : 'News Source', 0 : 'Count'})
    
groups = ['Conservative', 'Municipalist', 'Liberal']
colors = ['blue', 'green', 'red']

def grouped_bar_chart(df, column, column_sort=None, group_function = None):
    
    if group_function is not None:
        grouped_counts = group_function(df)
    else:
        grouped_counts = df.groupby(['Group', column]).size().reset_index(name='Count')

    

    probs = df['Group'].value_counts(normalize=True)
    # Calculating the standard error (SE) for each group and applying the formula for the 5% confidence interval (CI)
    # Assuming a normal distribution, the Z value for 95% confidence is 1.96
    z_score = 1.96
    grouped_counts = grouped_counts.merge(probs, on='Group')
    grouped_counts['SE'] = np.sqrt(grouped_counts['proportion'] * (1 - grouped_counts['proportion']) / grouped_counts['Count']) 
    grouped_counts = grouped_counts.join(grouped_counts.groupby(column)['Count'].sum().rename(f'Count_by_{column}'), on=column)
    grouped_counts['Count'] /= grouped_counts[f'Count_by_{column}']
    grouped_counts['Count'] *= 100
    grouped_counts['SE'] *= grouped_counts['Count']

    # Calculate confidence intervals using the normal approximation for the Poisson distribution
    grouped_counts['CI_lower'] = grouped_counts['Count'] - z_score * grouped_counts['SE']
    grouped_counts['CI_upper'] = grouped_counts['Count'] + z_score * grouped_counts['SE']

    # Ensure CI bounds do not go below 0
    grouped_counts['CI_lower'] = grouped_counts['CI_lower'].clip(lower=0, upper=100)
    if column_sort is None:
        column_sort = sorted(grouped_counts[column].unique())
    
    grouped_counts = grouped_counts.set_index(['Group', column]).unstack().fillna(0)

    
    
    
    
    # Create a grouped bar chart using Plotly
    fig = go.Figure()
    for i, group in enumerate(groups):
        fig.add_trace(go.Bar(
            x=column_sort,
            y=grouped_counts['Count'].loc[group][column_sort],
            name=group,
            error_y=dict(
                type='data',
                symmetric=False,
                array=grouped_counts['CI_upper'].loc[group][column_sort] - grouped_counts['Count'].loc[group][column_sort],
                arrayminus=grouped_counts['Count'].loc[group][column_sort] - grouped_counts['CI_lower'].loc[group][column_sort]
            ),
            marker_color=colors[i % len(colors)]
        ))

    # Update the layout
    fig.update_layout(
        barmode='group',
        title=f"Share of Respondents by Group in {column}",
        xaxis_title=column,
        yaxis_title="%",
        legend_title="Group"
    )

    # Show the figure
    fix_and_write(fig, filename=f"{column}_by_group")

def bar_chart(df, col):
    # Group by 'col' and calculate mean and standard error
    grouped = df.groupby('Group')[col].agg(['mean', 'count', 'std'])

    # Calculate the 95% confidence interval
    confidence = 0.95
    grouped['ci95_high'] = grouped['mean'] + stats.t.ppf((1 + confidence) / 2., grouped['count'] - 1) * grouped['std'] / np.sqrt(grouped['count'])
    grouped['ci95_low'] = grouped['mean'] - stats.t.ppf((1 + confidence) / 2., grouped['count'] - 1) * grouped['std'] / np.sqrt(grouped['count'])

    # Creating the plot with Plotly
    fig = go.Figure()

    for i, index in enumerate(groups):
        row = grouped.loc[index]
        fig.add_trace(go.Bar(name=index, x=[index], y=[row['mean']],
                             marker_color=colors[i % len(colors)],
                            error_y=dict(type='data',
                                        array=[row['ci95_high']-row['mean']],
                                        arrayminus=[row['mean']-row['ci95_low']])))
        fig.add_trace(go.Scatter(
            x=[index],
            y=[row['mean']/2],
            text=[f"{int(row['count'])} respondents"],
            mode="text",
        ))
    fig.update_layout(barmode='group',
        title=f"Average {col} by Group with 95% CI",
        xaxis_title='Group',
        yaxis_title=col,
        showlegend=False)
    fix_and_write(fig, col)

iv = ["Position on Economy LR", 
      'Income', 
      'Outlook on Israel (Combined Score)', 
      'Expectations from Municipaliity']
categoric_iv = ['Ethnicity', 'Religiosity', 'Section', 'Childhood Residence']
column_sort = {'Party' : ["Ra'am", 'Chadash-Taal', 'Meretz', 'Labor','Yesh Atid', 'HaMachane HaMamlachti', 
                          'Yisrael Beiteinu', 'Shas', 'United Torah Judaism', 'Likud', 
                          'Jewish Home', 'Religious Zionism']}

def main():
    df = pd.read_excel(Path(config['output_dir']).parent / "with_scores.xlsx")
    for col in categoric_iv + ['Quarter', 'Ability to influence Government Policy', 'Party', 'Voted for Coalition']:
        grouped_bar_chart(df, col, column_sort=column_sort.get(col, None))
    grouped_bar_chart(df, 'News Source', group_function=news_bar_group_function)
    for col in iv:
        bar_chart(df, col)

if __name__ == "__main__":
    main()