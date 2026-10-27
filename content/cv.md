---
layout: layouts/cv.njk
title: "CV"
permalink: /pages/cv.html
tags: cv
templateEngineOverride: njk

education:
  - title: "Ph.D. Candidate & M.S. in Computer Science"
    date: 2023 – Now
    org: Northwestern University
    location: Evanston, IL
    desc: Advised by Dr. Peter Dinda, Prescience Lab
  - title: "B.S. in Computer Engineering"
    date: 2019 – 2022
    org: Clemson University
    location: Clemson, SC

awards:
  - title: "DOE Computational Science Graduate Fellowship"
    date: 2023
    org: Department of Energy
  - title: "First Place, ACM Undergraduate Student Research Competition"
    date: 2022
    org: Supercomputing 2022
  - title: "Best Poster (Team), IndySCC22"
    date: 2022
    org: Supercomputing 2022

teaching:
  - title: "Teaching Assistant, CS 343: Operating Systems"
    date: Winter 2025
    org: Northwestern University
  - title: "Zhehao Zhu, Master's Research Mentor"
    date: 2024 – 2025
    org: Northwestern University
  - title: "Nader Rahhal, Undergraduate Research Mentor"
    date: 2025 – 2026
    org: Northwestern University
---
{% import "partials/cv.njk" as cv %}

{{ cv.list("Education", education) }}
{{ cv.list("Awards & Fellowships", awards) }}
{{ cv.experience("Experience", collections.experience) }}
{{ cv.list("Teaching & Mentoring", teaching) }}
{{ cv.publications("Publications", collections.publications) }}
{{ cv.talks("Conference Talks", collections.presentations, "conference") }}
{{ cv.talks("Invited Talks", collections.presentations, "invited") }}
{{ cv.posters("Posters", collections.posters) }}
