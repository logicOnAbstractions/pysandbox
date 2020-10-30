import click
from application import Foo, Bar

@click.group(name="main")
def main_menu():
    pass

@click.command()
@click.option('--age', default=42, help='Your age - defaults to 42 for some reasons')
@click.option('--name', prompt='your name', help='The person to greet.')
def main_menu(name, age):
    """ a simple programs that plays with objects & allows navigation in a menu"""
    foo = Foo(name, age)
    click.echo(foo.do_stuff())

# add the cmds to the group
# main_menu.add_command(foo)

if __name__ == '__main__':
    main_menu()