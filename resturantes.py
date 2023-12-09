import time
import googlemaps # pip install googlemaps
import pandas as pd # pip install pandas

def miles_to_meters(miles):
    try:
        return miles * 1_609.344
    except:
        return 0
        
API_KEY = 'AIzaSyAqUGuZe5tLuog68E-PRYVA9Xnmv4frNPw'
map_client = googlemaps.Client(API_KEY)

address = 'HUB-Robeson Center, University Park, PA'
geocode = map_client.geocode(address=address)

(lat, lng) = map(geocode[0]['geometry']['location'].get, ('lat', 'lng'))


distance = miles_to_meters(2)
business_list = []
response = map_client.places_nearby(
    location=(lat, lng),
    radius=distance,
    type='restaurant'
)   

business_list.extend(response.get('results'))



next_page_token = response.get('next_page_token')

while next_page_token:
    time.sleep(2)
    response = map_client.places_nearby(
        location=(lat, lng),
        radius=distance,
        type='restaurant',
        page_token=next_page_token
    )   
    business_list.extend(response.get('results'))
    next_page_token = response.get('next_page_token')

for place in business_list:
    print(place["name"])
#
df = pd.DataFrame(business_list)
df['url'] = 'https://www.google.com/maps/place/?q=place_id:' + df['place_id']
df.to_csv('{0}.csv'.format("NearbyResturantes"), index=False)
