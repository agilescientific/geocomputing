#!/usr/bin/env python
# -*- coding: utf-8 -*-
import click


@click.command()
@click.argument('name')
def hello(name):
    click.secho(f'Hello {name}!', fg='green')
    return


if __name__ == '__main__':
    hello()
