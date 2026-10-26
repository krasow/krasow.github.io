---
layout: layouts/home.njk
permalink: /index.html
templateEngineOverride: njk
greeting: Hello World! I'm David Krasowska.
skills: [MPI, CUDA, C, C++, Python, Julia, LLVM, HPC]
---
{% import "partials/home.njk" as home %}
{% import "partials/ui.njk" as ui %}

{{ home.greeting(greeting) }}
{# Bio from content/summary.md, education from content/cv.md, contact from content/_data/site.json #}
{{ home.terminal(site, collections.snippets | first, (collections.cv | first).data.education) }}
{{ ui.divider() }}
{# content/projects/ (only those with a `card:`) #}
{{ home.projects(collections.projects) }}
{{ ui.divider() }}
{# content/experience/ #}
{{ home.experience(collections.experience) }}
{{ ui.divider() }}
{# content/publications/ #}
{{ home.publications(collections.publications) }}
{{ home.skills(skills) }}
