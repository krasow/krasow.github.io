---
layout: layouts/base.njk
title: "Presentations"
permalink: /pages/presentations.html
templateEngineOverride: njk
---
{% import "partials/presentations.njk" as talks %}
{% import "partials/ui.njk" as ui %}

{# content/presentations/, split by each talk's `kind`; `poster:` ids link into content/posters/ #}
{{ talks.list("Conference Talks", collections.presentations, "conference", collections.posters) }}
{{ ui.divider() }}
{{ talks.list("Invited Talks", collections.presentations, "invited", collections.posters) }}
{{ ui.divider() }}
{# content/posters/ #}
{{ talks.posters("Posters", collections.posters) }}
