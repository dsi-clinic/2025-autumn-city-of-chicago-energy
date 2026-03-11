# Notebooks

This folder contains:
* `building_composition_over_time.ipynb` contains visualizations showing how building composition changes over time
* `change_in_energy_use.ipynb` contains building-level filtration of dataset, visualizations of year-over-year change in average and building-level and cumulative change-from-baseline.
* `change_in_energy_by_property_type.ipynb` contains property-type-level analysis of energy metrics, visualization of year-over-year change energy persistence, and categorization of property types into 3 COVID-impact-categories.
* `change_in_energy_by_year_buildings_and_community.ipynb` contains visualizations for year-over-year change energy persistence for year built, # of buildings, and community area.
* `data_exploration.ipynb` contains primary data exploration of descriptive statistics of key metrics showing energy performance.
* `energy_trends_by_building_type.ipynb` contains visualizations on how energy trends differ based on primary property type
* `exploratory_choropleth_maps.ipynb` contains choropleth maps of educational and demographic variables
* `exploratory_did_analysis_I.ipynb` and `exploratory_did_analysis_II.ipynb` contains difference-in-differences analysis and related visualizations
* `generational_mobility_visualization.ipynb` contains visualizations of intergenerational income and education mobility
* `interaction_with_covid_impact_and_weather.ipynb` contains exploration of weather normalized energy metrics, OLS model training that examines the interaction between property type and post_placard, covid-impact-category and post_placard.
* `other_cities_comparison.ipynb` contains comparison of Chicago energy usage with national trends.
* `visualization_by_year_and_exploratory_correlation.ipynb` contains visualization of metrics by year and correlation summary
* `within_building_variation.ipynb` contains visualizations of building-level energy change over year, distribution of pre-2019 and post-2019, and within-building fixed effect model, estimating the energy changes within building over time.
* `year_built_groupings.ipynb` contains grouping of year built as well as visualizations for year-over-year changes

Within fix_effects_notebooks: Folder was made to group fixed effect testing
* `fixed_effects.ipynb` contains initial exploration of using fixed effects for explination site EUI with Chicago Star Rating
* `fixed_effects_validation.ipynb` contains process of trying to validate the findings of fixed_effects.ipynb by adding more variables
* `fixed_effects_energy_rating.ipynb` Better visualizing the graphs in fixed_effects_validation.ipynb to show coeffiecent differences and further understanding trends
* `time_series_data_quality.ipynb` Exploring the quality of the data that would be used in the time series notebooks
* `time_series_naive.ipynb` Starting initial time series with naive structure 
* `time_series_features.ipynb` Using the naive time series model and added more features to better understand what explains energy use
* `time_series_local_change.ipynb` Created a new variable to better capture a buildings energy use in comparisons to the city trend
* `time_series_local_change_interaction.ipynb` Used the time series local change model to explore which features of a building explains the buildings energy trends
* `cities_vs_chicago_I.ipynb` contains exploratory comparison of Chicago energy trends with peer cities, including initial data preparation and visualizations of cross-city Site EUI patterns.
* `city_vs_chicago_II.ipynb` contains extended analysis comparing Chicago with other major cities, including trend visualizations and preliminary difference-in-differences exploration.
* `four_cities_comparison.ipynb` contains comparative analysis of energy use trends across Chicago and three peer cities, including visualizations of Site EUI changes and cross-city trend comparisons used to support the difference-in-differences analysis.