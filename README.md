# Running telemetry processor as well as running analytics engine, web interface to view run statistics

Key features:
    processes running telemetry from garmin watch in the form of .fit files using fitparse
    storage system using sqlite database engine with supabase storage free tier
    streamlit extension turns python data into readable web pages
    uses processed run data to calculate a variety of long term fitness metrics using TRIMP, 42-day/7-day exponential moving averages of CTL, ATL, and TSB statistics
    additionally has a single activity viewer showing statistics and a dynamic map from an activity
    shows a heatmap on a global map of every recorded activity


Files:
    app.py:
        frontend app controller: entry point controlled by the streamlit extension customizing everything about the page, including the password, font and theme, buttons, and computes activity statistics in real time
    aggregate_runs.py
        ingests the .fit files, filters out non-running ones, callibrates the occasional treadmill run, extracts gps samples and writes everything to the sqlite database
    hr_analytics.py
        math and modeling engine:
            dynamic karvonen heart rate zone calculator based on either auto or manually inputted heart rate data
            Banister TRIMP: algorithm quantifying workout strain
            Performance management chart: builds on weighted average of many statistics
            allows for manual pr overrides
    route_heatmap.py
        uses pydeck to render a heatmap of recorded runs, with dynamic coloring and intensity features
    cloud_vault.py
        bridges local sqlite with the supabase storage plan using requests extension, uses .toml for instructions
    fit_parser.py
        parser utalizing fitparse to clean up .fit files and extract usable statistics
    requirements.txt
        helper for streamlit to install necessary python libraries streamlit, pandas, numpy, matplotlib, fitparse, pydeck, requests

Services used:
    Github: hosts python codebase
    Streamlit Community cloud: hosts the web app and python environment from requirements.txt and app.py
    Supabase storage: private value bucket stores binary version of .fit files
    Openstreetmap: converts run latitude and longitude onto a usable map


