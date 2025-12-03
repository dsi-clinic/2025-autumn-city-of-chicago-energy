# Notebooks

This folder contains:
* `data_exploration.ipynb` contains primary data exploration of descriptive statistics of key metrics showing energy performance.
* `building.ipynb` contains building-level filtration of dataset, visualizations of year-over-year change in average and building-level and cumulative change-from-baseline.
* `building_new.ipynb` contains property-type-level analysis of energy metrics, visualization of year-over-year change energy persistence, and categorization of property types into 3 COVID-impact-categories.
* `interaction.ipynb` contains exploration of weather normalized energy metrics, OLS model training that examines the interaction between property type and post_placard, covid-impact-category and post_placard.
* `in_building.ipynb` contains visualizations of building-level energy change over year, distribution of pre-2019 and post-2019, and within-building fixed effect model, estimating the energy changes within building over time.
* `by_year_vis_and_correlation.ipynb` contains visualization of metrics by year and correlation summary

All scripts assume the cleaned dataset is located in '/output' or '/data/chicago_energy_benchmarking'. All scripts assume the helper functions are located in '/utils'.