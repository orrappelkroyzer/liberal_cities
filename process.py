import os, sys
from pathlib import Path
local_python_path = os.path.sep.join(__file__.split(os.path.sep)[:-1])
if local_python_path not in sys.path:
    sys.path.append(local_python_path)

from utils.utils import load_config, get_logger
logger = get_logger(__name__)
config = load_config(Path(local_python_path) / "config.json")
from datetime import datetime
import pandas as pd
from translation_dict import content_dict, titles_dict, similar_columns, neighborhood2quarter, quarter2section, coalition_parties
from stepmix.stepmix import StepMix
from sklearn.decomposition import PCA
import plotly.express as px
from utils.plotly_utils import fix_and_write, HTML
from extended_utils.analyses.lr.basic import coeffs_analysis
import plotly.graph_objects as go
pd.options.mode.chained_assignment = None
from itertools import combinations
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import scipy.stats as stats
import statsmodels.api as sm

def read_input():
    logger.info("Reading input")
    df1 = pd.read_excel(Path(config['input_dir']) / "raw input.xlsx", sheet_name="גברים")
    df1 = df1.loc[df1.index[1:]]
    df2 = pd.read_excel(Path(config['input_dir']) / "raw input.xlsx", sheet_name="נשים")
    df2 = df2.loc[df1.index[1:]]
    df = pd.concat([df1, df2]).set_index('panelistid')
    for k, v in similar_columns.items():
        df[k] = df[k].fillna(df[v])
    df = df.drop(columns=similar_columns.values())
    df = df.rename(columns=titles_dict)
    df = df.replace(content_dict)
    df = df.join(pd.Series(neighborhood2quarter, name='Quarter'), on='Neighborhood')
    df = df.join(pd.Series(quarter2section, name='Section'), on='Quarter')
    df['Income'] = df['Wage'].fillna(df["Family Income"])
    political_outlook_vars = ["Status of Israel", "Concern regarding Ability to Keep Lifestyle", "Outlook on Israeli Future"]
    expectation_from_municipality = ["Expect Municipality to be More Active in National Areas of Concern to the City", "Expect Municipality to be Expand it Activities"]
    for col in political_outlook_vars + expectation_from_municipality + ['Ability to influence Government Policy']:
        df[col] = df[col].replace("Don't know", np.nan)
        df[col] = df[col].replace("I don't know", np.nan)
    locs = df[political_outlook_vars].dropna().index
    df.loc[locs, 'Outlook on Israel (Combined Score)'] = PCA(n_components=1).fit_transform(df.loc[locs, political_outlook_vars])
    locs = df[expectation_from_municipality].dropna().index
    df.loc[locs, "Expectations from Municipaliity"] = PCA(n_components=1).fit_transform(df.loc[locs, expectation_from_municipality])
    df['Voted for Coalition'] = df["Party"].isin(coalition_parties)
    df.loc[df["Party"] == "Did not vote", 'Voted for Coalition'] = None
    return df

explained_variables = [
    'Importance of Open Spaces',
    'Importance of Walkability',
    'Importance of Culture',
    'Importance of Shabbat Services',
    'Importance of Arab-Jewish Relations',
    'Importance of Gay Rights',
    'Budget - Open Spaces',
    'Budget - Walkability',
    'Budget - Culture',
    'Budget - Shabbat Services',
    'Budget - Arab-Jewish Relations',
    'Budget - Gay Rights',
    'Shabbat Services vs. Open Spaces',
    'Shabbat Services vs. Walkability',
    'Shabbat Services vs. Culture',
    'Gay Rights vs. Walkability',
    'Gay Rights vs. Culture',
    'Gay Rights vs. Open Spaces',
    'Gay Rights vs. Walkability',
    'Arab-Jewish Relations vs. Culture',
    'Arab-Jewish Relations vs. Walkability',
    'Arab-Jewish Relations vs. Open Spaces',
]

# def lca(df, data_prefix, num_groups):
#     logger.info("Running LCA")
#     data = df[explained_variables]
#     model = StepMix(n_components=num_groups, measurement='continuous_nan', random_state=123, abs_tol=1e-5, rel_tol=1e-5)
#     model.fit(data)
#     df['lca_outcome'] = model.predict(data)
#     df['lca_outcome'] = df['lca_outcome'].apply(int)
#     (config['output_dir']  / data_prefix / f"{num_groups}_groups").mkdir(exist_ok=True, parents=True)
#     df.to_excel(Path(config['output_dir'])  / data_prefix / f"{num_groups}_groups" / f"lca_outcome.xlsx")
#     data['lca_outcome'] = df['lca_outcome']
#     a = data.groupby('lca_outcome').mean().T
#     a.to_excel(Path(config['output_dir'])  / data_prefix / f"{num_groups}_groups"/ f"lca_outcome_means.xlsx")
#     fig = go.Figure(data=[
#         # go.Bar(name=x, x=a[x], y=a.index,  orientation='h') for x in a.columns
#         go.Bar(name=x, y=a[x], x=a.index) for x in a.columns
#     ])
#     fig.update_xaxes(title_font=dict(size=4))
#     fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='Black')
#     fix_and_write(fig, output_dir=config['output_dir'] / data_prefix / f"{num_groups}_groups", filename=f"lca_outcome_means")
#     return df

# def present_on_pca(df, data_prefix, num_groups):
#     logger.info("Presenting LCA on PCA outcome")
#     data = df[explained_variables]
#     data_for_pca = data[[x for x in data.columns if "vs. " not in x]]
#     for x in data_for_pca.columns:
#         data_for_pca[x] = data_for_pca[x].fillna(data_for_pca[x].mean())
#     pca = PCA(n_components=2)
#     data_from_pca = pd.DataFrame(pca.fit_transform(data_for_pca), index=data_for_pca.index)
#     data_from_pca['lca_outcome'] = df['lca_outcome']
#     df['name'] = df['var555'].fillna("") + " " + df['var556'].fillna("")
#     data_from_pca['lca_outcome'] = data_from_pca['lca_outcome'].apply(str)
#     fig = px.scatter(data_from_pca, x=0, y=1, 
#                     color='lca_outcome', 
#                     hover_name=data_from_pca.index)
#     fix_and_write(fig, output_dir=config['output_dir'] / data_prefix / f"{num_groups}_groups", filename=f"lca_on_pca", output_type=HTML)


issues = ['Open Spaces', 'Walkability', 'Culture', 'Shabbat Services', 'Arab-Jewish Relations', 'Gay Rights']
def sanity_check(i_df):
    df = i_df.copy(deep=True)
    logger.info("Running sanity check")
    cs_df = pd.DataFrame(index=df.index)
    df['consistency_score_sharp_diffs'] = 0
    df['consistency_score_soft_diffs'] = 0
    for x in combinations(issues[::-1], 2):
        raw_diffs = pd.DataFrame({'importnace' : (df[f"Importance of {x[0]}"]-df[f"Importance of {x[1]}"]) / 10,
                              'budget' : (df[f"Budget - {x[0]}"]-df[f"Budget - {x[1]}"]) / 100})
        viniettes_col = f"{x[0]} vs. {x[1]}"
        if viniettes_col in df.columns:
            raw_diffs['viniette'] = df[viniettes_col].dropna() / 2
        diffs = pd.DataFrame(dict(zip([f"{y[0]}/{y[1]}" for y in combinations(raw_diffs, 2)], 
                                      [(((raw_diffs[y[0]] > 0) & (raw_diffs[y[1]] < 0)) | 
                                        ((raw_diffs[y[0]] < 0) & (raw_diffs[y[1]] > 0))) * \
                                            (raw_diffs[y[0]] + raw_diffs[y[1]]).apply(abs)  *
                                        raw_diffs[list(y)].notnull().T.all()
                                       for y in combinations(raw_diffs, 2)])))
        cs_df[f"{x[0]}/{x[1]} mismatch (sharp)"] = diffs.sum(axis=1) 
        df['consistency_score_sharp_diffs'] += cs_df[f"{x[0]}/{x[1]} mismatch (sharp)"]
        diffs = pd.DataFrame(dict(zip([f"{y[0]}/{y[1]}" for y in combinations(raw_diffs, 2)], 
                                      [(((raw_diffs[y[0]] >= 0) & (raw_diffs[y[1]] <= 0)) | 
                                        ((raw_diffs[y[0]] <= 0) & (raw_diffs[y[1]] >= 0))) * \
                                            (raw_diffs[y[0]] + raw_diffs[y[1]]).apply(abs)  *
                                        raw_diffs[list(y)].notnull().T.all()
                                       for y in combinations(raw_diffs, 2)])))
        cs_df[f"{x[0]}/{x[1]} mismatch (soft)"] = diffs.sum(axis=1).fillna(0)
        df['consistency_score_soft_diffs'] += cs_df[f"{x[0]}/{x[1]} mismatch (soft)"]
        
    return df

def conservatibe_liberal_tal_score(df):
    logger.info("conservative_liberal_score")
    liberal_issues = issues[3:]
    core_issues = issues[:3]
    scores = pd.DataFrame(index=df.index)
    budget_answers = pd.concat([df[[f"Budget - {x}" for x in liberal_issues]], -df[[f"Budget - {x}" for x in core_issues]]], axis=1).fillna(0)
    scores['Budget Score'] = budget_answers.sum(axis=1)
    importace_answers = pd.concat([df[[f"Importance of {x}" for x in liberal_issues]], -df[[f"Importance of {x}" for x in core_issues]]], axis=1).fillna(0)
    scores['Importance Score'] = importace_answers.sum(axis=1)
    viniettes_answers = df[[x for x in df.columns if "vs. " in x]].apply(lambda row: pd.Series(row.dropna().values), axis=1)
    scores['Vignette Score'] = viniettes_answers.sum(axis=1)
    boundaries = {
        'Budget Score' : [-100, 100],
        'Importance Score' : [-30, 30],
        'Vignette Score' : [-6, 6]
    }
    for col in scores.columns:
        s = scores[col]
        bounds = boundaries[col]
        normalized_s = (s - bounds[0]) / (bounds[1] - bounds[0])
        scores[col] = 2*(normalized_s - 0.5)
    for x, y in combinations(scores.columns, 2):
        fig = px.scatter(scores, x=x, y=y, hover_name=scores.index)
        x_data = scores[x]
        y_data = scores[y]
        # Compute the linear regression line using np.polyfit
        m, b = np.polyfit(x_data, y_data, 1)

        # Add the trendline to the scatter plot
        fig.add_trace(go.Scatter(x=x_data, y=m*x_data + b, mode='lines', name='Trendline'))

        # Calculate the R^2 value
        residuals = y_data - (m * x_data + b)
        ss_total = np.sum((y_data - np.mean(y_data)) ** 2)
        ss_residual = np.sum(residuals ** 2)
        r_squared = 1 - (ss_residual / ss_total)
        fig.update_layout(
            annotations=[
                go.layout.Annotation(
                    text=f'Equation: y = {round(m, 2)}x + {round(b, 2)}, R^2={round(r_squared, 2)}',
                    xref="paper", yref="paper",
                    x=0.05, y=0.05, showarrow=False,
                    font=dict(size=20)
                )
                ]
        )
        # fix_and_write(fig, f"Scores_{x}_{y}", output_type=HTML)
    

    df['Combined Score'] = PCA(n_components=1).fit_transform(scores)
    df.loc[df['Combined Score'] <= -0.497007912, 'Group'] = "Conservative"
    df.loc[df['Combined Score'] >= 0.478071363, 'Group'] = 'Liberal'
    df['Group'] = df['Group'].fillna('Municipalist')
    fig = px.histogram(df['Combined Score'], nbins=150, title="Conservative-Liberal Score Distribution")
    fix_and_write(fig, filename="score_distribution")
    fig = px.scatter_3d(df.join(scores), 
                        x='Vignette Score', 
                        y='Budget Score', 
                        z='Importance Score', 
                        color='Group', hover_name=df.index)
    fix_and_write(fig, filename="score_3d", output_type=HTML)
    
    return df.join(scores)

categoric_iv = ['Ethnicity', 'Religiosity', 'Section', 'Childhood Residence']
def bar_charts(df):
    logger.info("bar_charts")
    for col in categoric_iv + ['Quarter', 'Ability to influence Government Policy', 'Party', 'Voted for Coalition']:
        # Group by 'col' and calculate mean and standard error
        grouped = df.groupby(col)['Combined Score'].agg(['mean', 'count', 'std'])

        # Calculate the 95% confidence interval
        confidence = 0.95
        grouped['ci95_high'] = grouped['mean'] + stats.t.ppf((1 + confidence) / 2., grouped['count'] - 1) * grouped['std'] / np.sqrt(grouped['count'])
        grouped['ci95_low'] = grouped['mean'] - stats.t.ppf((1 + confidence) / 2., grouped['count'] - 1) * grouped['std'] / np.sqrt(grouped['count'])

        # Creating the plot with Plotly
        fig = go.Figure()

        for index, row in grouped.iterrows():
            fig.add_trace(go.Bar(name=index, x=[index], y=[row['mean']],
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
            title=f"Conservative-Liberal score according to {col}",
            xaxis_title=col,
            yaxis_title="Conservative-Liberal score",
            showlegend=False)
        fix_and_write(fig, f"Scores_{col}")

def news_bar_chart(df):
    logger.info("news_bar_chart")
    tf_cols = [col for col in df.columns if col.startswith('News Consumtion:')]
    fig = go.Figure()

    for col in tf_cols:
        # Create a sub-dataframe where the True/False column is True
        sub_df = df[df[col]]

        # Calculate mean, count and standard deviation
        mean = sub_df['Combined Score'].mean()
        count = sub_df['Combined Score'].count()
        std = sub_df['Combined Score'].std()

        # Calculate the 95% confidence interval
        confidence = 0.95
        ci95_high = mean + stats.t.ppf((1 + confidence) / 2., count - 1) * std / np.sqrt(count)
        ci95_low = mean - stats.t.ppf((1 + confidence) / 2., count - 1) * std / np.sqrt(count)

        # Add bar trace
        fig.add_trace(go.Bar(name=col, x=[col], y=[mean],
                            error_y=dict(type='data',
                                        array=[ci95_high-mean],
                                        arrayminus=[mean-ci95_low])))

    # Add count (number of observations) as text annotation inside each bar
    fig.add_trace(go.Scatter(
        x=[col],
        y=[mean],
        text=[str(count)],
        mode="text",
    ))
    fig.update_layout(barmode='group',
            title=f"Conservative-Liberal score according to News Source",
            xaxis_title="News Source",
            yaxis_title="Conservative-Liberal score",
            showlegend=False)
    fix_and_write(fig, f"Scores_News_Source")

iv = ["Position on Economy LR", 
      'Income', 
      'Outlook on Israel (Combined Score)', 
      'Expectations from Municipaliity']

def scatter_plots(df):
    logger.info("scatter_plots")
    for col in iv:
        t_df = df[[col, 'Combined Score']]
        t_df[col] = t_df[col].replace("I don't know", np.nan)
        t_df = t_df.dropna()
        x_data = t_df[col]
        y_data = t_df['Combined Score']

 

        # Standardize the input data
        scaler = MinMaxScaler()
        x_data = scaler.fit_transform(np.array(x_data).reshape(-1, 1))
        y_data = scaler.fit_transform(np.array(y_data).reshape(-1, 1))
        t_df = pd.DataFrame({col:x_data.reshape(1, -1)[0], 'Combined Score' : y_data.reshape(1, -1)[0]})
        fig = px.scatter(t_df, x=col, y='Combined Score', hover_name=t_df.index)


        # Add a constant column to the independent variable
        x_data = sm.add_constant(x_data)

        # Fit the ordinary least squares (OLS) model
        model = sm.OLS(y_data, x_data)
        results = model.fit()
        m, b = results.params[1], results.params[0]
        r_squared = results.rsquared


        # Add the trendline to the scatter plot
        fig.add_trace(go.Scatter(x=x_data[:, 1], y=m*x_data[:, 1] + b, mode='lines', name='Trendline'))

        fig.update_layout(
            annotations=[
                go.layout.Annotation(
                    text=f'Equation: y = {round(m, 2)}x + {round(b, 2)}, R^2={round(r_squared, 2)}',
                    xref="paper", yref="paper",
                    x=0.05, y=0.05, showarrow=False,
                    font=dict(size=28)

                )
                ])
        fix_and_write(fig, f"Scores_{col}")

def multivariate_regression(df):
    logger.info("multivariate_regression")
    coeffs_analysis(df=df, 
                    output_dir=config['output_dir'],
                    dependent_variables=['Combined Score'],
                    independent_variables=iv + \
                                          [col for col in df.columns if col.startswith('News Consumtion:')] + \
                                          ['Ability to influence Government Policy', 'Voted for Coalition'],
                    categoric_independent_variables=categoric_iv,
                    filename='multivariate_regression',
                    height_factor=1.5)

def multivariate_regression_partial_iv(df):
    logger.info("multivariate_regression")
    coeffs_analysis(df=df, 
                    output_dir=config['output_dir'],
                    dependent_variables=['Combined Score'],
                    independent_variables=[ 'Outlook on Israel (Combined Score)', 
                                            'Voted for Coalition', 'Income', 
                                           ],
                    categoric_independent_variables= ['Ethnicity', 'Religiosity'],
                    uninteresting_independent_variables=['const', 'Ethnicity', 'Income'],
                    filename='multivariate_regression_v1_1',
                    sort_values=False,
                    height_factor=1.5)

    coeffs_analysis(df=df, 
                    output_dir=config['output_dir'],
                    dependent_variables=['Combined Score'],
                    independent_variables=['Ability to influence Government Policy',
                                            'Voted for Coalition', 'Income', 
                                           ],
                    categoric_independent_variables= ['Ethnicity', 'Religiosity'],
                    uninteresting_independent_variables=['const', 'Ethnicity', 'Income'],
                    filename='multivariate_regression_v1_2',
                    sort_values=False,
                    height_factor=1.5)


def main():
    df = read_input()
    df = sanity_check(df)
    df.to_csv(Path(config['output_dir']) / "sanity_check.csv")
    t_df = df[df['consistency_score_soft_diffs'] <= df['consistency_score_soft_diffs'].quantile(0.9)]
    t_df = conservatibe_liberal_tal_score(t_df)
    t_df.to_excel(Path(config['output_dir']) / "with_scores.xlsx")
    t_df = pd.read_excel(Path(config['output_dir']) / "with_scores.xlsx")


    # bar_charts(t_df)
    # news_bar_chart(t_df)
    # scatter_plots(t_df)
    # multivariate_regression(t_df)
    multivariate_regression_partial_iv(t_df)


if __name__ == "__main__":
    main()