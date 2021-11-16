# {{title}}

Welcome to the **{{title}}** class!


## Set up an environment

To build an environment called {{env}}, type the following commands inside the Anaconda Prompt (or terminal) **in the same directory where you placed the repo** (i.e. so that the `environment.yaml` file is in the working directory):

```
conda env create
conda activate {{env}}
python -m ipykernel install --user --name {{env}}
```

Note that you only need to do that last line once per environment. In future you can activate this environment with `conda activate {{env}}` and start working.

Now you can start a notebook server with `jupyter notebook` and you're all set!


## Curriculum

{% for day, items in curriculum.items() %}### Day {{day}}

{% for item in items %}{% if '.ipynb' in item %}- [{{ item|replace('_', ' ')|replace('.ipynb', '') }}](notebooks/{{ item }}){% else%}- {{ item }}{% endif %}
{% endfor %}
{% endfor %}
## Extras

{% for item in extras %}- [{{ item|replace('_', ' ')|replace('.ipynb', '') }}](notebooks/{{ item }})
{% endfor %}

---

agilescientific.com