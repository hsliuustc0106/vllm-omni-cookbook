#!/usr/bin/env ruby
# frozen_string_literal: true

# Blog post lint: every blog/_posts/*.md must carry a `summary` front-matter
# field of at most 240 characters (SEO description + index card teaser).
# Minimal reimplementation of the convention vllm.ai enforces in CI.
#
# Usage: ruby scripts/check-blog-summaries.rb [posts_dir]

require "yaml"
require "date"

posts_dir = ARGV[0] || File.join(__dir__, "..", "blog", "_posts")
posts = Dir.glob(File.join(posts_dir, "*.md")).sort

if posts.empty?
  warn "no posts found under #{posts_dir}"
  exit 1
end

config = YAML.safe_load(File.read(File.join(__dir__, "..", "blog", "_config.yml")),
                        permitted_classes: [Date, Time], aliases: false) || {}
feature_slugs = (config["features"] || []).map { |f| f["slug"].to_s.downcase }

bad = []
all_tags = Hash.new { |h, k| [] } # downcased tag -> [original, file]
meta_by_path = {}                # normalized URL -> [meta, filename] (for pair checks)

def normalize_url(u)
  u = u.to_s.strip
  u = u.sub(%r{/+\z}, "")
  u = File.join("", u) unless u.start_with?("/")
  u
end

posts.each do |path|
  begin
    front = File.read(path).split(/^---\s*$/)[1]
    meta = front ? YAML.safe_load(front, permitted_classes: [Date, Time], aliases: false) : nil
  rescue StandardError => e
    bad << "#{File.basename(path)}: unparseable front matter (#{e.class}: #{e.message})"
    next
  end

  summary = meta.is_a?(Hash) ? meta["summary"] : nil
  if summary.nil? || summary.to_s.strip.empty?
    bad << "#{File.basename(path)}: missing `summary` front matter"
  elsif summary.length > 240
    bad << "#{File.basename(path)}: summary is #{summary.length} chars (max 240)"
  end

  (meta.is_a?(Hash) ? meta["tags"] : []).to_a.each do |tag|
    tag = tag.to_s
    all_tags[tag.downcase] << [tag, File.basename(path)]
    if feature_slugs.include?(tag.downcase)
      bad << "#{File.basename(path)}: tag `#{tag}` duplicates a site.features slug — classify via `feature:` front matter instead"
    end
  end

  # URL for pairing checks: explicit permalink, else the default
  # /:year-:month-:day-:title/ derived from the filename.
  own = meta["permalink"] ||
        "/" + File.basename(path).sub(/\.zh\.md\z/, "").sub(/\.md\z/, "")
  meta_by_path[normalize_url(own)] = [meta, File.basename(path)] if meta.is_a?(Hash)
end

# Bilingual pairing: a post declaring a language must point at an existing
# companion whose `pair` points back, and a zh edition must live under /zh/
# mirroring its English canonical URL.
meta_by_path.each_value do |meta, name|
  lang = meta["lang"]
  pair = meta["pair"]
  next if lang.nil? && pair.nil?

  own_url = normalize_url(meta["permalink"] ||
                          "/" + name.sub(/\.zh\.md\z/, "").sub(/\.md\z/, ""))
  if lang == "zh"
    unless own_url.start_with?("/zh/")
      bad << "#{name}: zh edition must use a /zh/... permalink, got #{own_url}"
    end
  end
  if pair.nil?
    bad << "#{name}: `lang` declared without a `pair` companion URL"
    next
  end
  pair_url = normalize_url(pair)
  companion = meta_by_path[pair_url]
  if companion.nil?
    bad << "#{name}: `pair` #{pair} does not resolve to any post"
  elsif normalize_url(companion[0]["pair"]) != own_url
    bad << "#{name}: `pair` is not mutual — #{companion[1]} points at #{companion[0]["pair"].inspect}"
  elsif lang == "zh" && own_url != "/zh" + normalize_url(companion[0]["permalink"] ||
          "/" + companion[1].sub(/\.zh\.md\z/, "").sub(/\.md\z/, ""))
    bad << "#{name}: zh permalink must mirror the English URL under /zh/"
  end
end

all_tags.each do |down, variants|
  originals = variants.map(&:first).uniq
  if originals.size > 1
    bad << "tag casing drift: #{originals.join(" vs ")} (#{variants.map(&:last).uniq.join(", ")}) — use one canonical form"
  end
end

if bad.any?
  warn bad.join("\n")
  exit 1
end

puts "#{posts.size} post(s) OK (summary <= 240 chars)"
