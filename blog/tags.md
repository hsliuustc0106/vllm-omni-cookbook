---
layout: page
title: Tags
---

{%- assign date_format = site.minima.date_format | default: "%b %-d, %Y" -%}

All post tags, most used first.

{%- comment -%}Liquid cannot sort a hash by value directly; map to "count|tag" strings.{%- endcomment -%}
{%- assign tag_pairs = "" | split: "" -%}
{%- for tag in site.tags -%}
  {%- assign pair = tag[1].size | append: "|" | append: tag[0] -%}
  {%- assign tag_pairs = tag_pairs | push: pair -%}
{%- endfor -%}
{%- assign tag_pairs = tag_pairs | sort | reverse -%}

{%- for pair in tag_pairs -%}
{%- assign count = pair | split: "|" | first | plus: 0 -%}
{%- assign tag = pair | split: "|" | last -%}
## <a id="{{ tag | slugify }}" href="#{{ tag | slugify }}">#{{ tag }}</a> <span class="tag-count">{{ count }}</span>

{%- for post in site.tags[tag] %}
- [{{ post.title | escape }}]({{ post.url | relative_url }}) — {{ post.date | date: date_format }}
{%- endfor %}

{%- endfor -%}
