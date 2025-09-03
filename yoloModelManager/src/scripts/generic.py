import click


@click.command()
@click.option(
    '--width',
    '-w',
    'w',
    type= click.IntRange(
        min= 0,
        min_open= True
    ),
    defaut= 1280,
    help= 'Width of the screen.'
)
@click.option(
    '--height',
    '-h',
    'h',
    type= click.IntRange(
        min= 0,
        min_open= True
    ),
    defaut= 720,
    help= 'Height of the screen.'
)
def move_cursor(
    w: int,
    h: int
) -> None:
    import random
    import time
    import platform
    
    while True:
        x: int = random.randint(0, w)
        y: int = random.randint(0, h)
        if platform.system() == 'Windows':
            import ctypes
            ctypes.windll.user32.SetCursorPos(x, y) # type: ignore
        time.sleep(30)
