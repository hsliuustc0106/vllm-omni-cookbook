---
layout: page
title: Tags
---

All post tags, most used first.

{%- comment -%}Liquid cannot sort a hash by value directly; map to "count|tag" strings.{%- endcomment -%}
{%- assign tag_pairs = "" | split: "" -%}
{%- for tag in site.tags -%}
  {%- assign pair = tag[1].size | append: "|" | append: tag[0] -%}
  {%- assign tag_pairs = tag_pairs | push: pair -%}
{%- endfor -%}
{%- assign tag_pairs = tag_pairs | sort | reverse -%}

<div class="tags-grid">
{%- for pair in tag_pairs -%}
{%- assign count = pair | split: "|" | first | plus: 0 -%}
{%- assign tag = pair | split: "|" | last -%}
  <section class="tag-card" id="{{ tag | slugify }}">
    <h2 class="tag-card-title"><a href="#{{ tag | slugify }}">#{{ tag }}</a>
      <span class="sidebar-count">{{ count }}</span>
    </h2>
    <ul class="tag-post-list">
      {%- for post in site.tags[tag] -%}
      <li>
        <a href="{{ post.url | relative_url }}">{{ post.title | escape }}</a>
        <time datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: "%b %-d, %Y" }}</time>
      </li>
      {%- endfor -%}
    </ul>
  </section>
{%- endfor -%}
</div>
