import json
import os
import re

import requests


def hide_cells(notebook, tags=None):
    """
    Finds the tags in each cell and removes it.

    Returns dict without 'hide' tagged cells.
    """
    if tags is None:
        tags = ['hide']

    clean = []
    for cell in notebook['cells']:
        if not set(tags).intersection(cell['metadata'].get('tags', list())):
            clean.append(cell)

    notebook['cells'] = clean

    return notebook


def empty_cells(notebook):
    """
    Finds the tag 'empty' in each cell and removes its content

    Returns dict with empty cells
    """

    clean = []
    for cell in notebook['cells']:
        try:
            tags = cell['metadata']['tags']
            if True in map(lambda x: x.lower().startswith('empty'), tags):
                cell['source'] = []
                clean.append(cell)

        except KeyError:
            clean.append(cell)

    notebook['cells'] = clean

    return notebook


def style_cells(notebook, style):
    """
    Finds the tags 'exercise', 'advanced' or 'info' in each cell and applies HTML template.

    Returns dict with template cells.
    """
    styles = {
        'exercise': "<div style=\"background: #e0ffe0; border: solid 2px #d0f0d0; border-radius:3px; padding: 1em; color: darkgreen\">\n",
        'advanced': "<div style=\"background: #fff0e0; border: solid 2px #ffe7d0; border-radius:3px; padding: 1em; color: chocolate\">\n",
        'info': "<div style=\"background: #e0f0ff; border: solid 2px #d0e0f0; border-radius:3px; padding: 1em; color: navy\">\n",
    }

    clean = []
    wraphead = [styles[style], ]
    wraptail = ["\n</div>", ]

    for cell in notebook['cells']:
        try:
            tags = cell['metadata']['tags']
            if True in [t.lower().startswith(style[:3]) for t in tags]:
                src = cell['source']
                src = [re.sub(r"^#+? (.+)\n", r"<h3>\1</h3>\n", s) for s in src]
                cell['source'] = wraphead + src + wraptail
        except KeyError:
            pass
        clean.append(cell)
    notebook['cells'] = clean

    return notebook


def hide_code(notebook):
    """
    Finds the tags '#!--' and '#--! in each cell and removes
    the lines in between.

    Returns dict
    """

    for i, cell in enumerate(notebook['cells']):
        istart = 0
        istop = -1
        for idx, line in enumerate(cell['source']):
            if '#!--' in line:
                istart = idx
            if '#--!' in line:
                istop = idx

        notebook['cells'][i]['source'] = cell['source'][:istart] + cell['source'][istop+1:]

    return notebook


def hide_toolbar(notebook):
    """
    Finds the display toolbar tag and hides it
    """

    if 'celltoolbar' in notebook['metadata']:
        del(notebook['metadata']['celltoolbar'])
    return notebook


def process_notebook(infile,
                     outfile,
                     clear_input=False,  # Don't touch the input file.
                     clear_output=True,
                     exercise=True,
                     advanced=True,
                     info=True,
                     hidecode=True,
                     demo=False,  # If demo, remove exercises and enable demos.
                     data_url_stem=None,
                    ):
    """
    Loads an 'ipynb' file as a dict and performs cleaning tasks

    Writes cleaned version
    """
    if clear_input:
        _ = os.system("nbstripout {}".format(infile))

    with open(infile, encoding='utf-8') as f:
        notebook = json.loads(f.read())

    if demo:
        notebook = hide_cells(notebook, tags=['exercise', 'solution'])
    else:
        notebook = hide_cells(notebook, tags=['hide', 'demo'])

    if exercise:
        notebook = style_cells(notebook, 'exercise')
    if advanced:
        notebook = style_cells(notebook, 'advanced')
    if info:
        notebook = style_cells(notebook, 'info')

    if hidecode:
        notebook = hide_code(notebook)

    notebook = hide_toolbar(notebook)

    text = json.dumps(notebook)
    images = re.findall(r"\.\./images/([-_.a-zA-Z0-9]+)", text)
    # print(infile, images)

    if data_url_stem is None:
        data_url_stem = r"https://geocomp\.s3\.amazonaws\.com/data/"
    data_urls = re.findall(fr"({data_url_stem}[-_.a-zA-Z0-9]+)", text)
    # print(infile, data_urls)

    with open(outfile, 'w') as f:
        _ = f.write(text)

    if clear_output:
        _ = os.system("nbstripout {}".format(outfile))

    return images, data_urls

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clean a notebook")
    parser.add_argument('infile', help='Input notebook')
    parser.add_argument('outfile', help='Output notebook')
    parser.add_argument('--clear-input', action='store_true', help='Remove input cells')
    parser.add_argument('--clear-output', action='store_true', help='Remove output cells')

    process_notebook(**vars(parser.parse_args()))
