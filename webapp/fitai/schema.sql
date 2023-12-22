DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS restaurants;
DROP TABLE IF EXISTS fooditems;

CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    first_name TEXT NOT NULL,
    birthdate TEXT NOT NULL,
    height INTEGER NOT NULL,
    weight INTEGER NOT NULL,
    sex TEXT NOT NULL,
    fitness_goal TEXT NOT NULL
);

CREATE TABLE restaurants (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    name TEXT UNIQUE NOT NULL
);

CREATE TABLE fooditems (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    restaurant_id INTEGER NOT NULL,
    name TEXT UNIQUE NOT NULL,
    calories INTEGER NOT NULL,
    fat INTEGER NOT NULL,
    carbs INTEGER NOT NULL,
    protein INTEGER NOT NULL,
    FOREIGN KEY (restaurant_id)
       REFERENCES restaurants (id)
);

INSERT INTO users (username,password,first_name,birthdate,height,weight,sex,fitness_goal) 
VALUES ('test','$2b$12$Gokl1daL9hYTa4a2S1qqU.ZfmVaRRNlhTjQGxZqi1VYWaX3OrLsHO',
'test','1992-11-28',68,150,'male','lose_weight');

INSERT INTO restaurants (name) VALUES ('Chick-fil-a');
INSERT INTO restaurants (name) VALUES ('McDonalds');
INSERT INTO restaurants (name) VALUES ('Taco Bell');

INSERT INTO fooditems (restaurant_id, name, calories, fat, carbs, protein) 
VALUES ((SELECT id FROM restaurants WHERE name = 'Chick-fil-a'),'Spicy Chicken Sandwich',450,19,45,28);
INSERT INTO fooditems (restaurant_id, name, calories, fat, carbs, protein) 
VALUES ((SELECT id FROM restaurants WHERE name = 'Chick-fil-a'),'Grilled Chicken Sandwich',390,12,44,28);

INSERT INTO fooditems (restaurant_id, name, calories, fat, carbs, protein) 
VALUES ((SELECT id FROM restaurants WHERE name = 'McDonalds'),'Big Mac',590,34,46,25);
INSERT INTO fooditems (restaurant_id, name, calories, fat, carbs, protein) 
VALUES ((SELECT id FROM restaurants WHERE name = 'McDonalds'),'10 Piece Chicken McNuggets',410,24,26,23);

INSERT INTO fooditems (restaurant_id, name, calories, fat, carbs, protein) 
VALUES ((SELECT id FROM restaurants WHERE name = 'Taco Bell'),'Cheesy Gordita Crunch',490,28,41,20);
INSERT INTO fooditems (restaurant_id, name, calories, fat, carbs, protein) 
VALUES ((SELECT id FROM restaurants WHERE name = 'Taco Bell'),'Quesadilla - Chicken',520,26,41,26);
