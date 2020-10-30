

class Foo:
    def __init__(self, name, age=42):
        self.name       = name
        self.age        = age
        self.bar        = None

    def do_stuff(self):
        return f"I am {self.name} & I am {self.age} yo"

    def add_bar(self, country, city):
        self.bar        = Bar(country, city)


class Bar:
    def __init__(self, country, city):
        self.country    = country
        self.city       = city

    def where_am_i(self):
        return f"I am in {self.city}, {self.country}"
