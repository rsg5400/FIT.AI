# imports
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, DateField, IntegerField, SelectField
from wtforms.validators import DataRequired, EqualTo, Length, ValidationError
from utilities import get_db_connection

class SignUpForm(FlaskForm):
    """
        User Signup Form
    """
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=8)])
    confirm_password = PasswordField('Confirm Password', 
        validators=[DataRequired(), EqualTo('password')])
    first_name = StringField('First Name', validators=[DataRequired()])
    birthdate = DateField('Birthdate', validators=[DataRequired()])
    height = IntegerField('Height (in)', validators=[DataRequired()])
    weight = IntegerField('Weight (lbs)', validators=[DataRequired()])
    sex = SelectField('Sex:', choices=[('male', 'Male'), ('female', 'Female')], 
        validators=[DataRequired()])
    fitness_goal = SelectField('Fitness Goal:', choices=[
        ('lose_weight', 'Lose Weight'),
        ('burn_fat', 'Burn Fat'),
        ('gain_muscle', 'Gain Muscle'),
        ('sports_performance', 'Perform Better in Sports')
    ], validators=[DataRequired()])

class LoginForm(FlaskForm):
    """
        User Login Form
    """
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=8)])

    def validate_username(self, username):
        """Method to validate usernames"""
        with get_db_connection() as con:
            curs = con.cursor()
            curs.execute("SELECT username FROM users where username = (?)",[username.data])
            valuser = curs.fetchone()
            if valuser is None:
                raise ValidationError(
                    'This Username is not registered. Please register before login')
