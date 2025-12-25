---
title: "PGA Tour Stats Scraper"
date: 2025-05-11
lastmod: 2025-05-11
tags: ["`Python", "^^All Projects", "Web scrape", "PGA Tour Stats", "Sports Betting", "Sports Data"]
author: "Shaun Yap"
description: "Automate PGA Tour data collection with a Python scraper that extracts tournament-level stats from 2004 to present. Includes full dataset and Jupyter Notebook for analysis." 
summary: "This project centres on building a fully automated web scraper that collects tournament-level statistics from the official PGA Tour website, covering data from 2004 to the present. It overcomes the platform's manual, one-stat-at-a-time download limitation by enabling users to extract structured, high-quality .csv datasets across any year or date range - with support for all available stat codes. The tool is available as both a Python script and an interactive Jupyter Notebook, and the repository also includes a complete pre-scraped dataset (2004–2025) for immediate use." 
cover:
    image: ""
    alt: ""
    relative: false
editPost:
    URL: "https://github.com/shaunyap01/PGA-Tour-Stats-Scraper"
    Text: PGA Tour Stats Scraper GitHub Repository
showToc: true
disableAnchoredHeadings: false

---

*A data project driven by a love for golf, statistics, and a curiosity: can data outsmart the bookmakers?*

---

## **From Tee Box to Text Editor**

I used to play competitive golf, representing **Kuala Lumpur**, Malaysia's capital, at the **national level**. I wasn't the biggest hitter or the most naturally gifted - I was nearly a foot shorter than the average player in my age group (thanks to a late growth spurt and some unfortunate date cutoffs). So I leaned into what I could control: **statistics-based preparation and deliberate course management**.

That edge took me further than talent alone might have.

These days, I'm no longer grinding out 36-hole weekends, but my love for golf - and for data - hasn't gone anywhere. Lately, that's taken a new form: **sports betting**. Nothing serious - just small-stakes wagers with friends - but over the past year, I've seen a surprising **150% return**.

I'll be the first to admit: a good portion of that was luck. But it got me wondering…

> *What would it take to build a model that doesn't just predict results - but consistently outperforms the bookmakers' implied odds?*

---

## **Where Do You Even Start?**

Like any data scientist, I started with the data.

Sites like **DataGolf** and **FantasyNational**, and others frequently cited on Reddit were promising - but behind paywalls. That left me with one major free alternative: the **official PGA Tour statistics site**.

It's a goldmine, offering data back to **2004** - but also a logistical nightmare. Every CSV must be downloaded manually, one at a time, by selecting a **specific stat**, for a **specific tournament**.

To put this in perspective:

* \~350 stats per tournament
* \~39 events per season
* That's **over 13,650 CSV files** for a single year

There weren't any public scrapers available. Most users either paid for third-party access or gave up.

---

## **So I Built One**

To solve this, I built a **web scraper** that automates collection of all **‘Tournament Only' statistics** from the PGA Tour website - across **any season from 2004 to the present**.

It enables flexible, large-scale data collection without needing manual intervention.

### **Key Features**

* Scrape across **any date range or season between 2004 and present**
* Select specific metrics using the **Stat Code Reference List**
  *e.g. `02564` = Strokes Gained: Putting*
* Export clean `.csv` files, organised by:

  * **Stat category**
  * **Season or date range**
* Run as either:

  * **Python script (CLI)**
  * **Jupyter Notebook (interactive)**

In addition, the **GitHub repository includes fully pre-scraped data** covering **01/01/2004 – 20/04/2025**, so you can start exploring immediately - no scraping required.

---

## **Why This Matters**

There's growing interest in using data to inform sports predictions - but too often, the data itself is locked away, incomplete, or poorly documented. By making this tool (and the data) openly available, I hope to support others who want to:

* Build predictive models
* Investigate performance trends
* Analyse course-specific player stats
* Create more transparent, hands-on sports analytics projects

In short, this is about lowering the barrier to entry - and encouraging exploration.

---

## **What's Next?**

This project lays the groundwork by solving the data access problem - but the real challenge (and excitement) lies ahead. Building predictive models using this dataset, exploring whether performance metrics can reliably anticipate outcomes - and how closely those outcomes align (or diverge) from bookmaker odds.

---

## [Explore the GitHub Repository](https://github.com/shaunyap01/PGA-Tour-Stats-Scraper)

*(Includes full codebase, setup instructions, stat references, and a complete pre-scraped dataset (from 2004-01-01 to 2025-04-20))*

---

Saved some money using this instead of buying data elsewhere?  
Consider [supporting via Buy Me a Coffee](https://buymeacoffee.com/shaunyap01) - thank you!
