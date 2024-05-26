import lasair
import pandas as pd 

TOKEN = 'XXXXXXXXXXXXXXX'

def get_potential_host(obj, ra, dec, ori_df_path):
    """Retrieve potential host information for a given object."""
    print(obj)
    try:
        client = lasair.lasair_client(TOKEN)
        result = client.sherlock_position(ra, dec)
        crossmatches = result.get("crossmatches", [])
        if crossmatches:
            top_host = crossmatches[0]
            top_host['object_id'] = obj
            top = pd.DataFrame([top_host])
            original_df = pd.read_csv(ori_df_path)
            if obj not in original_df['object_id'].tolist():
                df = pd.concat([original_df, top], ignore_index=True)
                df.to_csv('test.csv')
            top_ra, top_dec = top_host['raDeg'], top_host['decDeg']
        else:
            top_ra, top_dec = None, None
    except Exception as e:
        print(f"Error retrieving potential host for object {obj}: {e}")
        top_ra, top_dec = None, None
    return top_ra, top_dec

def get_multiple_hosts(table, ori_df_path):
    """Retrieve potential hosts for multiple objects."""
    df = pd.read_csv(ori_df_path)
    try:
        client = lasair.lasair_client(TOKEN)
        for ztf_id, ra, dec in zip(table['ztf_id'], table['ra'], table['dec']):
            if ztf_id in df['object_id'].tolist():
                print(f'Object {ztf_id}\'s host is already recorded!\n')
                continue
            else:
                result = client.sherlock_position(ra, dec)  
                crossmatches = result.get("crossmatches", [])
                if crossmatches:
                    top_host = crossmatches[0]
                    top_host['object_id'] = ztf_id
                    top = pd.DataFrame([top_host])
                    df = pd.concat([df, top], ignore_index=True)
                    print(f'Object {ztf_id}\'s host is added!\n')
                else:
                    print(f'Object {ztf_id}\'s host is not found!\n')
        df.to_csv(ori_df_path)
    except Exception as e:
        print(f"Error retrieving hosts: {e}")
