---
layout: layouts/base.njk
title: "News"
permalink: /pages/news.html
templateEngineOverride: njk
---
{% import "partials/news.njk" as news %}

{# content/news/, one file per article #}
{{ news.list("News", collections.news) }}
