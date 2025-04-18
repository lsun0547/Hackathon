import random

import pandas as pd
from faker import Faker

fake = Faker()

# Sample data pools
location_options = ['North Gate', 'Old Oak Tree', 'Mausoleum', 'Grave 42', 'Chapel', 'Back Fence']
weather_options = ['Clear', 'Foggy', 'Rainy', 'Stormy']
moon_options = ['New Moon', 'Young Moon', 'Waxing Crescent', 'Waxing Quarter', 'Waxing Gibbous', 'Full Moon',
                'Waning Gibbous', 'Waning Quarter', 'Waning Crescent', 'Old Moon']
temp_options = ['Very Cold', 'Cold', 'Mild', 'Warm', 'Hot']
time_options = [f"{hour}:00" for hour in range(0, 24)]
day_options = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']


def generate_entry():
    time = random.choice(time_options)
    location = random.choice(location_options)
    weather = random.choice(weather_options)
    moon = random.choice(moon_options)
    temp = random.choice(temp_options)
    day = random.choice(day_options)

    # Haunting logic based on likely to be haunted areas, based on weather, time, moon phase, etc
    haunted = 0
    if (weather in ['Foggy', 'Stormy'] or int(time.split(':')[0]) in range(0, 5)
            or location in ['Mausoleum', 'Chapel']
            or temp in ['Very Cold', 'Cold']):
        haunted = random.choices([1, 0], weights=[0.85, 0.15])[0]
    elif (moon in ['Full Moon'] and int(time.split(':')[0]) in range(20, 5) or (
            location in ['Grave 42'] and day in ['Friday']) or (
                  temp in ['Very Cold'] and weather in ['Foggy', 'Stormy'])):
        haunted = random.choices([1, 0], weights=[0.9, 0.1])[0]
    else:
        haunted = random.choices([0, 1], weights=[1, 0])[0]

    return {
        'time': time,
        'location': location,
        'weather': weather,
        'temperature': temp,
        'day': day,
        'moon': moon,
        'haunted': haunted
    }


# Generate dataset
dataset = [generate_entry() for _ in range(1000000)]
df = pd.DataFrame(dataset)

print(df.head())

# Save to CSV
df.to_csv('haunting_dataset.csv', index=False)

# If printing full data set
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#     print(df)

# print('Haunted Dataset:\n'+df.head())
