import click

@click.group(invoke_without_command=True)
@click.pass_context
def main_group(ctx):
    """ Lists all the submenu options available"""
    cmds    = main_group.list_commands(ctx)
    click.echo(f"Available options:")
    for idx, cmd_str in enumerate(cmds):
        click.echo(f"{idx}:{cmd_str}")

    click.echo(f"Now that you know all the options, let's make a selection:")
    ctx.invoke(main_group.get_command(ctx, "selection"))

@main_group.command()
@click.option('--next_cmd', prompt='Next command:', help="Enter the number corresponding to the desired command")
@click.pass_context
def selection(ctx, next_cmd):
    click.echo(f"You've selected {next_cmd}")

    # check that selection is valid

    # invoke the desired command

    # return to parent previous command


@main_group.command()
def submenu_1():
    click.echo('A submenu option ')

@main_group.command()
def submenu_2():
    click.echo('Another option')

@main_group.command()
def application_code():
    click.echo('running some application code.... ')
    click.echo('done.')

if __name__ == '__main__':
    main_group()
    # selection()

