---
layout: layouts/base.njk
title: News
permalink: /pages/news.html
# The body is a page outline (Nunjucks), not Markdown prose.
templateEngineOverride: njk
---
{% import "partials/news.njk" as news %}

{# content/news/, one file per article #}
{{ news.list("News", collections.news) }}
