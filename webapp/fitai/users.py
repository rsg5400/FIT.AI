""" User Model"""

from flask_login import UserMixin

class User(UserMixin): # (too many instance attributes) pylint: disable=R0902
    """Flask-login user class"""

    def __init__(self, user_data):
        self.id = user_data[0] # id
        self.username = user_data[2] # username
        self.password = user_data[3] # password (hashed)
        self.first_name = user_data[4] # first_name
        self.birthdate = user_data[5] # birthdate
        self.height = user_data[6] # height
        self.weight = user_data[7] # weight
        self.sex = user_data[8] # sex
        self.fitness_goal = user_data[9] # fitness_goal
        self._is_authenticated = False
        self._is_active = True
        self._is_anoymous = False

    @property
    def is_authenticated(self):
        return self._is_authenticated

    @is_authenticated.setter
    def is_authenticated(self, val):
        self._is_authenticated = val

    @property
    def is_active(self):
        return self._is_active

    @is_active.setter
    def is_active(self, val):
        self._is_active = val

    @property
    def is_anonymous(self):
        return self._is_anoymous

    @is_anonymous.setter
    def is_anonymous(self, val):
        self._is_anonymous = val

    def get_id(self):
        return self.id

    def __str__(self):
        """Method to print string representation of object"""
        str_user = f"""
            self.id: {self.id}
            self.username: {self.username}
            self.password: {self.password}
            self.first_name: {self.first_name}
            self.birthdate: {self.birthdate}
            self.height: {self.height}
            self.weight: {self.weight}
            self.sex: {self.sex}
            self.fitness_goal: {self.fitness_goal}
            self._is_authenticated: {self._is_authenticated}
            self._is_active: {self._is_active}
            self._is_anoymous: {self._is_anoymous}
        """
        return str_user
