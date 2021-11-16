{% for day, items in curriculum.items() %}### Day {{day}}

    {% for item in items %}{% if '.ipynb' in item %}- [{{ item|replace('_', ' ')|replace('.ipynb', '') }}](notebooks/{{ item }}){% else%}- {{ item }}{% endif %}
{% endfor %}{% endfor %}