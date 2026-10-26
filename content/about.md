---
layout: layouts/about.njk
title: "About"
permalink: /pages/about.html
photo: /assets/images/sc.JPG
photoLink: https://sc22.supercomputing.org/presentation/?id=spostu110&sess=sess227#038;sess=sess227
---

# My background {.title-xlg}

I am a Computer Science Ph.D. candidate at Northwestern University under Dr. Peter Dinda and a recipient of the Department of Energy Computational Science Graduate Fellowship. I've become interested in helping domain scientists run their workloads on distributed, heterogeneous computing systems. This focus is part of the [Constellation Project](https://constellation-project.net) which enables frictionless heterogeneous parallelism. My research introduction began with research within the FTHPC group at Clemson University under Dr. Jon C. Calhoun. Within this group, I collaborated with Argonne National Laboratory as a visiting student on lossy data compression research. Our goal was to determine different statistical predictions that would allow for estimation of lossy compression ratios (more info [here](/projects/compression/)). This had lead to [publications](/index.html#publications) and [awards](/pages/news.html). Also, I participated in the Student Cluster Competition at Super Computing (SC) '21 and INDYSCC at SC '22.

# David's interests?

The end of Moore's law has resulted in development of new specialized architectures for computing [1]. AI companies are driving the current accelerator boom, but I find this question particularly compelling: **How can we leverage emerging AI-focused heterogeneous accelerators to support the workloads of HPC domain scientists?**

I've became interested in utilizing higher-level languages in a performant way. I am working on a project that brings the ability to run Julia applications across a distributed system using NVIDIA GPUs. This takes advantage of Julia's abstract array interface to implement distributed heterogeneous arrays managed by a novel HPC runtime- Legion (more info [here](/projects/cunumeric/)).

Outside of GPUs, I've worked with CGRAs, FPGAs, etc. However, one specific accelerator that has peaked my interest in Processing in Memory (PIM). There has been countless prior work in the area of increasing computational speed using processing in memory (PIM) accelerators. There are a few issues that require a demanding amount of work for adoption for these newer ideas. This challenge stems from the fact that domain scientists are often required to rewrite their code using specialized directives to take advantage of new hardware. Unfortunately, this portability issue isn't unique to PIM- many emerging AI accelerators face the same barrier. Using the same novel runtime mentioned above (Legion), I have made progress towards resolving the complexities of programming PIM systems (more info [here](/projects/legionpim/)).

[1] T. N. Theis and H. . -S. P. Wong, "The End of Moore's Law: A New Beginning for Information Technology," in Computing in Science & Engineering, vol. 19, no. 2, pp. 41-50, Mar.-Apr. 2017, doi: 10.1109/MCSE.2017.29.
{.cite}
