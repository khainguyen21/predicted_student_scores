# Student Score Analysis

A simple data analysis project that processes student scoring data to explore correlations between math, reading, and writing scores. The code also demonstrates basic data preprocessing techniques using scikit-learn pipelines.

## Overview

This project analyzes student performance data with a focus on:
- Correlation between different subject scores
- Data preprocessing using scikit-learn pipelines
- Handling categorical education level data with ordinal encoding

## Features

- Data loading and exploration
- Correlation analysis between math, reading, and writing scores
- Train/test data splitting
- Preprocessing pipelines for handling missing values
- Ordinal encoding for education level categories

## Requirements

- Python 3.x
- pandas
- scikit-learn
- statistics

## Usage

1. Place your "StudentScore.xls" file in the same directory as the script
2. Run the script to perform the analysis
3. View the correlation results between subject scores
4. See the before/after results of ordinal encoding for education levels

## Data Preprocessing

The code includes two preprocessing approaches:
- Numerical data transformation (currently commented out)
- Categorical data transformation for education levels using ordinal encoding

## Note

The code includes commented-out sections for ydata_profiling which can be uncommented if you wish to generate a comprehensive data profile report.
