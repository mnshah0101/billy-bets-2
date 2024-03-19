import pandas as pd
import json
import requests

# Assuming 'df' is your DataFrame
# sports_data_key should be your actual API key
sports_data_key = '66b9a43385ed4dc981818ad925d0efb9'
season = '2023'
URL = f"https://api.sportsdata.io/v3/cbb/scores/json/TeamSeasonStats/{season}"
data = requests.get(URL, headers={'Ocp-Apim-Subscription-Key': sports_data_key})

# Ensure data from API is converted to a DataFrame
df = pd.DataFrame(data.json())

# Create the mapping from the 'Name' column to the 'TeamID' column
team_name_to_id = df.set_index('Name')['TeamID'].to_dict()

# Initialize updated_data dictionary
updated_data = {}

# Modify the mapping to include both the full name and the name minus the last word
for key, value in team_name_to_id.items():
    # Add original mapping
    updated_data[key] = value
    # Split the key by spaces to remove the last word
    new_key_parts = key.split(' ')[:-1]  # Remove the last word
    # Only add a new entry if there is more than one word in the original key
    if new_key_parts:
        new_key = ' '.join(new_key_parts)  # Rejoin the remaining words
        updated_data[new_key] = value  # Add the new mapping

# Convert the updated mapping to JSON, with indentation for readability
updated_json = json.dumps(updated_data, indent=4)



# Write the updated mapping to a different file
with open('team_id.json', 'w') as f:
    json.dump(updated_data, f, indent=4)
