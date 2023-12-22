"""
    Main Application File
"""

import os
from sqlite3 import DatabaseError
from flask import Flask, render_template, request, redirect, flash, url_for
from flask_bcrypt import Bcrypt
from flask_cors import CORS
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from dotenv import load_dotenv
from forms import LoginForm, SignUpForm
from users import User
from utilities import get_db_connection, get_restaurant_menu, get_restaurants, analyze_menu

# Load environment variables from .env file
load_dotenv()

# Configure App
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes and origins
bcrypt= Bcrypt(app)
login_manager = LoginManager(app)
login_manager.init_app(app)
login_manager.login_view = "login"

# Configure Flask
app.config['DEBUG'] = os.environ.get('FLASK_DEBUG')
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')
app.config['SESSION_COOKIE_NAME'] = 'fitai_session'
app.config['SESSION_COOKIE_SECURE'] = False
app.config['SESSION_COOKIE_HTTPONLY'] = True

# Connect database
con = get_db_connection()
cur = con.cursor()

@app.route('/', methods=['POST', 'GET'])
def index():
    """Default Route"""
    return render_template('index.html')

@app.route('/restaurants')
@login_required
def restaurants():
    """Restaurant List page"""
    restaurant_list = get_restaurants()
    return render_template('restaurants.html', data=restaurant_list)

@app.route('/menu/<restaurant_id>', methods=['GET', 'POST'])
@login_required
def menu(restaurant_id):
    """Menu page"""
    restaurant_menu = get_restaurant_menu(restaurant_id)
    smart_menu = analyze_menu(restaurant_menu)
    return render_template('menu.html', data=smart_menu)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """User signup"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = SignUpForm()
    return render_template('signup.html', form=form)

@app.route('/createuser', methods=['GET', 'POST'])
def create_user():
    """User Creation"""
    if request.method == 'POST':
        try:
            username = request.form['username']
            password = request.form['password']
            hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
            first_name = request.form['first_name']
            birthdate = request.form['birthdate']
            height = request.form['height']
            weight = request.form['weight']
            sex = request.form['sex']
            fitness_goal = request.form['fitness_goal']

            cur.execute("""INSERT INTO users (username,password,first_name,birthdate,
                        height,weight,sex,fitness_goal) VALUES (?,?,?,?,?,?,?,?)""",
                        (username,
                            hashed_password,
                            first_name,
                            birthdate,
                            height,
                            weight,
                            sex,
                            fitness_goal))
            con.commit()
            print("User Creation Successful!")
        except DatabaseError:
            con.rollback()
            flash("User Creation Failed")
        return redirect(url_for('index'))
    return redirect('index')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User Login Method"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = LoginForm()
    if request.method == 'POST':
        try:
            username = request.form['username']
            password = request.form['password']
            cur.execute("""SELECT * FROM users WHERE username=?""",
                                            (username,))
            user_data = cur.fetchone()
            user_obj = User(user_data)
            if bcrypt.check_password_hash(user_obj.password, password):
                login_user(user_obj)
                print("User Login Successful!")
            else:
                flash('Incorrect Username or Password.')
                return redirect(url_for('login'))
        except DatabaseError:
            flash("User Login Failed")
        return redirect(url_for('restaurants'))
    return render_template('login.html', form=form)

@app.route("/logout")
@login_required
def logout():
    """User Logout Method"""
    logout_user()
    return redirect('/')

@login_manager.user_loader
def load_user(user_id):
    """Flask Login user loader"""
    cur.execute('SELECT * FROM users WHERE id = ?', (user_id,))
    user_data = cur.fetchone()
    if user_data:
        user = User(user_data)
        user.is_authenticated = True
        return user
    return None

if __name__ == '__main__':
    app.run()
