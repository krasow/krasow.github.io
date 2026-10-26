---
layout: layouts/base.njk
title: Presentations
permalink: /pages/presentations.html
# The body is a page outline (Nunjucks), not Markdown prose.
templateEngineOverride: njk
---
{% import "partials/presentations.njk" as talks %}
{% import "partials/ui.njk" as ui %}

{# content/presentations/, split by each talk's `kind` #}
{{ talks.list("Conference Talks", collections.presentations, "conference") }}
{{ ui.divider() }}
{{ talks.list("Invited Talks", collections.presentations, "invited") }}
