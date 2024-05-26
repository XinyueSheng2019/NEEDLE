import os
import json
import lasair


token = 'XXXXXXXXXXXX'

def get_json(ztf_id, path):
    # Construct the save path for the JSON file
    save_path = os.path.join(path, f"{ztf_id}.json")

    # Check if JSON file already exists
    if not os.path.exists(save_path):
        # Initialize Lasair client with token
        L = lasair.lasair_client(token)

        # Fetch data for the given ZTF ID
        c = L.objects([ztf_id])[0]

        try:
            # Remove non-detections
            temp_list = [cd for cd in c['candidates'] if 'candid' in cd.keys()]
            c['candidates'] = temp_list
        except Exception as e:
            print(f"Error processing ZTF ID {ztf_id}: {e}")

        # Convert data to JSON format
        json_object = json.dumps(c, indent=4)

        # Save JSON data to file
        with open(save_path, "w") as outfile:
            outfile.write(json_object)
    else:
        print(f"JSON file already exists for ZTF ID {ztf_id}")

