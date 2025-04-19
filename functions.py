import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import date, timedelta, datetime,timezone
import requests
import pandas
import pytz
from scipy.optimize import curve_fit
from zoneinfo import ZoneInfo
import os


def time_since(timestr: str) -> float:
    """
    Given a string like "Time: 2025-04-17 03:19:42 EDT",
    return how many hours ago that was.
    """
    # Define EDT as UTC-4
    edt = timezone(timedelta(hours=-4))
    
    # Extract and parse the datetime
    dt_str = timestr.replace("Time:", "").replace("EDT", "").strip()
    dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=edt)
    
    # Get current time in EDT
    now = datetime.now(edt)
    
    # Time difference in hours
    delta = now - dt
    return delta.total_seconds() / 3600

def write_csv(csv_file,row):
    """
    The current log is in csv_file 
    row is a python dictionary with new information that goes into a single line
    """
    new_df = pandas.DataFrame([row])
    new_df['time'] = pandas.to_datetime(new_df['time']).dt.tz_convert('America/Toronto')
    
    if os.path.exists(csv_file):
        df = pandas.read_csv(csv_file)
        # Convert 'time' column to datetime with timezone if not already
        df['time'] = pandas.to_datetime(df['time']).dt.tz_convert('America/Toronto')
    

        if (df['time'] == row['time']).any():
            print("Row with this timestamp already exists.")
        else:
            df = pandas.concat([df, new_df], ignore_index=True)
            df.to_csv(csv_file, index=False)            
    else:
        # File doesn't exist, create it
        new_df.to_csv(csv_file, index=False)


def get_time(first_tick = 0,
             next_tick = 1):
    "start and end of the time for the fridge query"
    now = datetime.now(timezone.utc)

    first_tick = now - timedelta(hours=first_tick)
    next_tick = now - timedelta(hours=next_tick)

    formatted_first_tick = first_tick.strftime("%Y-%m-%dT%H:%M:%S")
    formatted_next_tick  = next_tick.strftime("%Y-%m-%dT%H:%M:%S")
    return (formatted_first_tick, formatted_next_tick)


def get_config(file_path
                   ):
    "Keep the password in the a separate place"
    credentials = {}
    with open(file_path, 'r') as file:
        for line in file:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                credentials[key.strip()] = value.strip()
    return credentials


get_url = lambda url,table_name,tick_info: (f'{url}'
                                            f'table={table_name}&'
                                            f'startDate={tick_info[0]}&'
                                            f'endDate={tick_info[1]}'
                                           )

def get_fridge_data(tick_info,
                    table_name = 'fridge',
                    credential_file = 'credentials.txt'
                   ):
    "get fridge data"

    config = get_config(credential_file)
     
    url = get_url(config['url'],
                  table_name,
                  tick_info)
    print(url)
    
    response = requests.get(url, auth=(config['username'], config['password']))    
    
    data = response.json()
    pd_df = pandas.DataFrame(data)
    pd_df['timestamp'] = pandas.to_datetime(pd_df['timestamp'], utc=True)
    pd_df['timestamp'] = pd_df['timestamp'].dt.tz_convert('America/Toronto')
    pd_df['timestamp_int'] = pd_df['timestamp'].astype('int64')
    pd_df['timestamp_s'] = pd_df['timestamp_int'] / 10**9
    pd_df['timestamp_h'] = pd_df['timestamp_s'] / 3600
    return pd_df

def get_step_index(I_array):
    "Find the indices beyond which the MC temperature changes."
    
    I_array = np.array(I_array)  # if it's not already
    diffs = I_array[1:] != I_array[:-1]
    index_list = np.where(diffs)[0].tolist()

    return index_list


def make_plot(data, step_index):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3),dpi = 150)

    # Extract tick positions
    tick_positions = data['timestamp_h'].iloc[step_index].values
    tick_labels = data['timestamp'].iloc[step_index].dt.strftime('%m/%d-%H:%M')
    # tick_labels = [f"{x:.2f}" for x in tick_positions]  # Format as needed

    # Plot data
    ax[0].plot(data['timestamp_h'], data['I3'], c = 'black')
    ax[0].set_ylabel('MC-current [mA]')
    ax[0].set_xlabel('Time')
    ax[0].set_xticks(tick_positions)  # Set only desired ticks
    ax[0].set_xticklabels(tick_labels, rotation=90, ha='center')

    
    ax[1].plot(data['timestamp_h'], data['T3'], c = 'black')
    ax[1].set_ylabel('MC-temperature [mK]')
    ax[1].set_xlabel('Time')
    ax[1].set_xticks(tick_positions)  # Set only desired ticks
    ax[1].set_xticklabels(tick_labels, rotation=90, ha='center')

    # Add vertical lines
    for index in step_index:
        x = data['timestamp_h'].iloc[index]
        ax[0].axvline(x, linestyle=':', color='r')
        ax[1].axvline(x, linestyle=':', color='r')
    return fig, ax

pram_to_record = {'time': 'timestamp',
                  'Still current [mA]': 'I0',
                  'MC current [mA]': 'I3',
                  'MC [mK](RuOx)': 'T3',
                  'MC [mK](CMN)': 'T5',
                  'ST [mK]': 'T2',
                  'CP [mK]': 'T4',
                  '4K [mK]': 'T1',
                  'Flow [ml/min]': 'Flow',
                  'PCOMP [mBar]': 'Pc',
                  'ST P [mBar]': 'MG2',
                  '4He Dump [mBar]': 'P6',
                  '3He Dump [mBar]': 'P7'}


get_available_keys = lambda df, pram_to_record: {tag:key for tag, key in pram_to_record.items() if key in df}
get_row_data = lambda row, avialable_keys: {key: row[mapped_key] for key, mapped_key in avialable_keys.items()}

def get_info_matching_rows(df1,df2,mapping):
    'find rows that match between df1 and df2 and return their info'
    avialable_keys1 = get_available_keys(df1,mapping)
    queries = {}
    for index, row in df1.iterrows():
        queries[row['timestamp_int']] = get_row_data(row,avialable_keys1)

    avialable_keys2 = get_available_keys(df2,mapping)
    for time in queries:
        closest_index = df2['timestamp_int'].sub(time).abs().idxmin()
        closest_row = df2.loc[closest_index]
        info = get_row_data(closest_row,avialable_keys2)   
        queries[time] = queries[time] | info
        
    return queries

def get_temp_info(n_hour=(23,10), mapping=pram_to_record):
    "organize the fridge information at times that should be recorded into a dictionary"
    
    time_array = get_time(*n_hour)
    
    df1 = get_fridge_data(time_array, table_name='temp')
    df2 = get_fridge_data(time_array)
    
    step_index = get_step_index(df1['I3'])
    reduced_df1 = df1.iloc[step_index]
    
    queries = get_info_matching_rows(reduced_df1,df2,mapping)
    



    fig, ax = make_plot(df1, step_index)
    He4, He3, ST_current = [list(queries.values())[0][i] for i in ['4He Dump [mBar]',
                                        '3He Dump [mBar]',
                                        'Still current [mA]']]
    
    fig.suptitle(f'He4: {He4:.1f} mBar - He3: {He3:.1f} mBar\nStill current: {ST_current} mA - sensor: RuOx' )
    fig.tight_layout()
    fig.savefig(f'He4{He4:.1f}-He3{He3:.1f}-StillCurrent{ST_current}-RuOx.jpg')
    return queries
    
def save_to_text(queries,file_name = 'output_data.txt'):
    with open(file_name, "w") as f:
        for i in queries.values():
            # Convert Timestamp to string for readability
            time_str = i['time'].strftime("%Y-%m-%d %H:%M:%S %Z")
            f.write(f"Time: {time_str}\n")
            for key, value in i.items():
                if key != 'time':
                    f.write(f"-- {key}: {value}\n")
            f.write("\n") 


def time_diff_from_log(filepath):
    eastern = ZoneInfo("America/Toronto")  # or "America/New_York" for U.S. Eastern time

    with open(filepath, "r") as file:
        lines = file.readlines()

    # Extract all timestamps
    timestamps = [line.strip().split("Time: ")[1] for line in lines if line.startswith("Time:")]

    # Parse into timezone-aware datetime objects
    datetimes = [
        datetime.strptime(ts[:-4], "%Y-%m-%d %H:%M:%S").replace(tzinfo=eastern)
        for ts in timestamps
    ]

    if not datetimes:
        raise ValueError("No timestamps found in the file.")

    now = datetime.now(eastern)

    def hours_ago(t):
        delta = now - t
        return delta.total_seconds() / 3600

    #start from 3 hours before to 1 hours after a full scan of ST temp
    return hours_ago(datetimes[0])+3, hours_ago(datetimes[-1])-1


def get_temp_data(time_array):

    #load the data from DB
    df1 = get_fridge_data(time_array, table_name='temp')
    df2 = get_fridge_data(time_array)

    
    step_index = get_step_index(df1['I3'])
    #Dropping the first element.
    reduced_df1 = df1.iloc[step_index]
    
    
    queries_temp_up = get_info_matching_rows(reduced_df1.iloc[1:],df2,pram_to_record)
    
    get_ranges = lambda array: [[i,j] for i , j in zip(array[:-1], array[1:])] 
    windows = [df1.iloc[i:j] for i, j  in get_ranges(step_index)]

    temp_up = {(entry['4He Dump [mBar]'],
                entry['3He Dump [mBar]'],
                entry['Still current [mA]'],
                entry['MC current [mA]']) : data for entry,data in zip(queries_temp_up.values(), windows)}

    queries_temp_down = get_info_matching_rows(reduced_df1.iloc[:1],df2,pram_to_record)
    
    windows = [df1.iloc[:step_index[0]]]
    temp_down = {(entry['4He Dump [mBar]'],
                entry['3He Dump [mBar]'],
                entry['Still current [mA]'],
                entry['MC current [mA]']) : data for entry,data in zip(queries_temp_down.values(), windows)}
    
    return temp_down, temp_up

switch_names = {'RuOx':'T3',
                'CMN': 'T5'}
get_xy = lambda window, sensor: ((window['timestamp_s']-window['timestamp_s'].min()).to_numpy(),window[switch_names[sensor]].to_numpy())

def saturating_exp(x, a, b, c):
    return a-b*np.exp(c*x)

def do_fit(fit_func, 
           x,
           y,
           ax):
    # do fit
    params, pcov = curve_fit(fit_func, x, y,bounds=([0.001,0.001,-0.01],[max(y)+50,200,-0.00001]))    

    # plot the fit
    
    ax.plot(x,fit_func(x, *params), label = 'Fit function', c = 'k')
    # start of the fit
    ax.axvline(x[0],label='Start of fit',c = 'r')

    # show the temp difference
    temp_unc = params[0]-y[-1]
    
    ax.text(x[0]+100, y[-1]-10,                            # x, y in axis fraction
             f'limit - measured = {temp_unc:.2f} [mK]',  # your text
            )

    # find the amount of time needed to reach the limit
    y_max = y[-1]
    x_max = x[-1]
    temp_limit = 1
    x_extrapolation = []
    y_extrapolation = []
    while params[0]-y_max>temp_limit:
        x_max+=5
        y_max = saturating_exp(x_max,*params)
        x_extrapolation.append(x_max)
        y_extrapolation.append(y_max)
    extra_time = x_max-x[-1]
    ax.plot(x_extrapolation,y_extrapolation,c = 'b', label = 'Extrapolation')
    
    # show the amount of time needed to reach the limit
    ax.text(x[0]+100, y[-1]-15,                            # x, y in axis fraction
             f'Extra_time( $\\Delta T$<{temp_limit} mK) = {extra_time} [s]',  # your text
            )

    a, b, c = params
    eq_str = f"$f(x) = {a:.2f} - {b:.2f} \\, e^{{{c:.4f} \\, x}}$"
    
    # Show on plot
    ax.text(x[0]+100, y[-1]-20,            
            eq_str,
           )

    ax.axhline(params[0],alpha=0.4,c= 'b',label='Fit limit')
    
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('MC temperature [mK]')
    
    
    return params, extra_time, temp_unc


def do_fit_downward(fit_func, 
                    x,
                    y,
                    ax):
    
    # do fit
    params, pcov = curve_fit(fit_func, x, y,bounds=([0,-200,-0.001],[min(y),-0.0001,-0.00001]))

    # plot the fit
    
    ax.plot(x,fit_func(x, *params), label = 'Fit function', c = 'k')
    # start of the fit
    ax.axvline(x[0],label='Start of fit',c = 'r')

    # show the temp difference
    temp_unc = params[0]- y[-1]
    
    ax.text(x[0]-1000, y[-1]+9,                            # x, y in axis fraction
             f'limit - measured = {temp_unc:.2f} [mK]',  # your text
            )

    # find the amount of time needed to reach the limit
    y_max = y[-1]
    x_max = x[-1]
    temp_limit = 1
    x_extrapolation = []
    y_extrapolation = []
    while y_max-params[0]>temp_limit:
        x_max+=5
        y_max = saturating_exp(x_max,*params)
        x_extrapolation.append(x_max)
        y_extrapolation.append(y_max)
    extra_time = x_max-x[-1]
    ax.plot(x_extrapolation,y_extrapolation,c = 'b', label = 'Extrapolation')
    
    # show the amount of time needed to reach the limit
    ax.text(x[0]-1000, y[-1]+6,                            # x, y in axis fraction
             f'Extra_time( $\\Delta T$<{temp_limit} mK) = {extra_time} [s]',  # your text
            )

    a, b, c = params
    eq_str = f"$f(x) = {a:.2f} - ({b:.2f}) \\, e^{{{c:.4f} \\, x}}$"
    
    # Show on plot
    ax.text(x[0]-1000, y[-1]+3,            
            eq_str,
           )

    ax.axhline(params[0],alpha=0.4,c= 'b',label='Fit limit')
    
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('MC temperature [mK]')
    ax.set_ylim(5, 30)    
    return params, extra_time, temp_unc

def extract_mc_values(filepath,sensor='RuOx'):
    """
    Extracts all 'MC [mK](...)' values from a text file.

    Parameters:
        filepath (str): Path to the text file.

    Returns:
        List[float]: A list of MC [mK](CMN) values as floats.
    """
    values = []
    
    with open(filepath, 'r') as file:
        for line in file:
            if f'-- MC [mK]({sensor}):' in line:
                # Extract the value after the colon and strip whitespace
                value_str = line.split(':')[1].strip()
                try:
                    value = float(value_str)
                    values.append(value)
                except ValueError:
                    print(f"Warning: could not convert '{value_str}' to float.")
    
    return values







def fit_quadratic_and_plot(x, 
                           y, 
                           x_eval,
                           title,
                           ax,):
    
    # Fit a 2nd-order polynomial
    coeffs = np.polyfit(x, y, deg=2)
    poly = np.poly1d(coeffs)
    
    # Evaluate at x_eval
    y_eval = poly(x_eval)
    
    # Generate a smooth curve for plotting
    x_fit = np.linspace(min(x), 110, 500)
    y_fit = poly(x_fit)
    
    # Plot
    ax.plot(x_fit, y_fit, label='Quadratic Fit', color='blue')
    ax.scatter(x, y, label='Data Points', color='red')
    ax.scatter(x_eval, y_eval, color='black', zorder=5, label=f'Fit')
    ax.axvline(x_eval, linestyle='--', color='gray', alpha=0.5)
    ax.axhline(y_eval, linestyle='--', color='gray', alpha=0.5)
    ax.text(20,y_eval,f'{y_eval:.02f} $\\mu$W')
    ax.set_xlabel('MC temprature [mK]')
    ax.set_ylabel('MC power [$\\mu$W]')
    ax.set_ylim(0,500)
    ax.set_xlim(0,150)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return y_eval


def make_pretty_table(csv_file):
# Load the CSV
    df = pandas.read_csv(csv_file)
    
    # Convert all data to strings for formatting
    df_str = df.astype(str)
    df_str.columns = df_str.columns.map(str)
    
    # Determine max width for each column (header vs content)
    col_widths = {
        col: max(len(col), df_str[col].map(len).max())
        for col in df_str.columns
    }
    
    # Build header row
    header = "| " + " | ".join(f"{col:<{col_widths[col]}}" for col in df_str.columns) + " |"
    
    # Build separator row
    separator = "|-" + "-|-".join("-" * col_widths[col] for col in df_str.columns) + "-|"
    
    # Build each data row
    rows = [
        "| " + " | ".join(f"{val:<{col_widths[col]}}" for val, col in zip(row, df_str.columns)) + " |"
        for row in df_str.values
    ]
    
    # Combine full table
    table = "\n".join([header, separator] + rows)
    
    # Write to file
    with open("pretty_table.txt", "w") as f:
        f.write(table)
    print("✔️ Table with dynamic column spacing saved to 'pretty_table.txt'")
