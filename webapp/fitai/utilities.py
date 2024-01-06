# imports
import os
import json
import psycopg2
from openai import OpenAI
from flask_login import current_user

def get_completion(user, item):
    """Method to connect to OpenAI API"""
    #print(prompt)
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"""I need a meal choice categorized based on how it fits my current 
                               fitness level and my fitness goals.
                               I am {user}.
                               The meal choice is {item}.
                               Category 1 is the following menu item is a good choice. 
                               Category 2 is the following menu item is an average choice. 
                               Category 3 is the following item is a bad choice.
                               Reply with a JSON object in the following format:
                               {{"category": integer, "reason": string}}""",
            }
        ],
        model="gpt-3.5-turbo",
    )

    response = chat_completion.choices[0].message.content
    print(response)
    return response

def get_db_connection():
    """Method to connect to the database"""
    con = psycopg2.connect(
        host=os.environ['DB_IP'],
        database=os.environ['DB_NAME'],
        user=os.environ['DB_USERNAME'],
        password=os.environ['DB_PASSWORD'])
    return con

def get_restaurants():
    """Method to get restaurants"""
    restaurants = {}
    with get_db_connection() as con:
        cur = con.cursor()
        cur.execute("""SELECT * FROM restaurants""")
        restaurant_data = cur.fetchall()
        if restaurant_data:
            for row in restaurant_data:
                restaurants[row] = {
                    'id': row[0],
                    'name': row[2]
                }
    return restaurants

def get_restaurant_menu(in_restaurant_id):
    """Method to get restaurant menu items"""
    restaurant_menu = {}
    try:
        restaurant_id = int(in_restaurant_id)
    except ValueError:
        restaurant_id = 0
    with get_db_connection() as con:
        cur = con.cursor()
        cur.execute('SELECT * FROM restaurants WHERE id = %s',(restaurant_id,))
        restaurant_data = cur.fetchone()
        if restaurant_data:
            restaurant_menu = {
                'id': restaurant_data[0],
                'name': restaurant_data[2]
            }
        cur.execute('SELECT * FROM fooditems WHERE restaurant_id = %s',(restaurant_id,))
        menu_data = cur.fetchall()
        if menu_data:
            restaurant_menu['menu'] = {}
            for row in menu_data:
                restaurant_menu['menu'][row[0]] = {
                    'item': row[3],
                    'nutrition': {
                        'cal': row[4],
                        'fat': row[5],
                        'carbs': row[6],
                        'protein': row[7]
                    },
                    'quality': 0,
                    'reason': ''
                }
    return restaurant_menu

def analyze_menu(in_menu):
    """Method to get AI analysis for menu"""
    menu = in_menu

    for item in menu['menu']:
        client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        )

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"""I need menu options categorized based on how it fits my current 
                                fitness level and my fitness goals.
                                I am {current_user}.
                                The meal is {menu['menu'][item]}.
                                Category 1 is the following menu item is a good choice. 
                                Category 2 is the following menu item is an average choice. 
                                Category 3 is the following item is a bad choice.
                                Reply with a JSON object in the following format:
                               {{"category": integer, "reason": string}}""",
                }
            ],
            model="gpt-3.5-turbo",
        )

        response = json.loads(chat_completion.choices[0].message.content)
        menu['menu'][item]['quality'] = response['category']
        menu['menu'][item]['reason'] = response['reason']

    return menu
